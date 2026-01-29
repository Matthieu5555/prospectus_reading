"""Extraction phase — reads planned pages and pulls structured data from them.

Two extraction modes exist because prospectus layouts vary widely:
1. **Recipe mode** (preferred): uses DocumentLogic and pre-scanned tables to
   decide *per field* whether to do a table lookup or an LLM text extraction.
   This is faster and cheaper because tables are parsed once up front.
2. **Legacy mode**: two-pass LLM extraction with a pilot fund for learning.
   Kept as a fallback when DocumentLogic is unavailable.

The phase orchestrates umbrella extraction (two-pass for large docs),
broadcast-table pre-parsing, parallel per-fund extraction, and optional
critic verification.

**Two-Pass Umbrella Extraction:**
Solves token overflow on large documents (261+ pages) by splitting extraction:
- Pass 1: Entity info (name, depositary, management_company) from bounded
  intro+outro pages (~20 pages max)
- Pass 2: Constraints (investment restrictions, leverage) using TOC-guided
  sections or search-based fallback
"""

import asyncio
import logging
from dataclasses import dataclass, field

from extractor.core.config import ExtractionConfig as ExtractConfigConstants, UmbrellaConfig
from extractor.phases.phase_base import PhaseRunner
from extractor.pydantic_models import CriticResult
from extractor.pydantic_models.recipe_models import FundExtractionRecipe
from extractor.agents import (
    extract_umbrella,
    extract_with_search,
    apply_broadcast_data,
    verify_and_correct,
    verify_fund_extraction,
    discover_bonus_fields,
    generate_recipes,
)
from extractor.core import (
    ExtractionError, ErrorCategory, ErrorSeverity,
    QuestionPriority, CONSTRAINT_FIELDS, is_not_found, is_actionable_not_found,
    TableExtractor, ParsedTable,
    KnowledgeConsolidator,
    SourceType,
    select_umbrella_pages,
    check_token_limit,
)
from extractor.core.field_strategy import KnowledgeContext
from extractor.core.value_helpers import get_raw_value

logger = logging.getLogger(__name__)


@dataclass
class ExtractionResult:
    """Result from the extraction phase."""

    umbrella_data: dict
    funds_data: list[dict]
    broadcast_data: dict[str, list[dict]]
    critic_results: list[CriticResult]
    failed_funds: list[str] = field(default_factory=list)


class ExtractionPhase(PhaseRunner[ExtractionResult]):
    """Phase 3-5: Extract umbrella, broadcast tables, and funds.

    Uses recipe-based extraction with DocumentLogic:
    1. Umbrella extraction
    2. Broadcast table pre-parsing
    3. Parallel fund extraction with table lookups
    4. Critic verification (optional)
    """

    name = "Extraction"

    async def run(self) -> ExtractionResult:
        """Execute extraction phase.

        Returns:
            ExtractionResult with all extracted data.
        """
        if not self.context.plan:
            self.log("No plan available, skipping extraction", "warning")
            return ExtractionResult(
                umbrella_data={},
                funds_data=[],
                broadcast_data={},
                critic_results=[],
            )

        # Phase 3a: Umbrella
        await self._extract_umbrella()

        # Get DocumentLogic (always available from LogicPhase)
        document_logic = getattr(self.context.state, 'document_logic', None)
        if not document_logic:
            self.log("No DocumentLogic available - using minimal extraction", "warning")
            return ExtractionResult(
                umbrella_data=self.context.umbrella_data,
                funds_data=[],
                broadcast_data={},
                critic_results=[],
            )

        # Recipe-based extraction
        return await self._run_recipe_extraction(document_logic)

    async def _run_recipe_extraction(self, document_logic) -> ExtractionResult:
        """Execute recipe-based extraction using DocumentLogic.

        This is the v3 extraction path that uses table lookup for
        consolidated tables and dispatches based on field strategies.
        """
        # Generate recipes from plan, document logic, and knowledge graph
        # Knowledge graph has discovered table locations from exploration
        recipes = generate_recipes(
            self.context.plan,
            document_logic,
            self.context.knowledge,
        )
        self.log(f"Generated {len(recipes.fund_recipes)} fund recipes")

        if recipes.broadcast_tables:
            self.log(f"Broadcast tables to parse: {[t.table_type for t in recipes.broadcast_tables]}")
        else:
            self.log("No broadcast tables in recipes", "warning")

        # Use pre-scanned tables from TableScanPhase
        parsed_tables: dict[str, ParsedTable] = {}
        scanned = self.context.state.scanned_tables

        for table_source in recipes.broadcast_tables:
            if not table_source.table_pages:
                continue

            # Find matching pre-scanned tables by page
            matching = [t for t in scanned if t.page in table_source.table_pages]
            if matching:
                columns = matching[0].columns
                rows = []
                for t in matching:
                    rows.extend(t.rows)
                parsed_tables[table_source.table_type] = ParsedTable(
                    columns=columns,
                    rows=rows,
                    source_pages=(table_source.table_pages[0], table_source.table_pages[-1]),
                    raw_html="",
                )
                self.log(f"Using pre-scanned {table_source.table_type}: {len(rows)} rows")
            else:
                self.log(f"No pre-scanned table for {table_source.table_type}", "warning")

        # Initialize table extractor (for query functionality only)
        table_extractor = TableExtractor(use_llm=False)

        # Track how many broadcast tables were successfully parsed (for stats)
        self.context.parsed_broadcast_tables = len(parsed_tables)

        # Extract funds using recipes
        await self._extract_funds_with_recipes(
            recipes.fund_recipes,
            table_extractor,
            parsed_tables,
        )

        return ExtractionResult(
            umbrella_data=self.context.umbrella_data,
            funds_data=self.context.funds_data,
            broadcast_data=self.context.broadcast_data,
            critic_results=self.context.critic_results,
            failed_funds=self.context.errors.failed_entities,
        )

    async def _extract_umbrella(self):
        """Extract umbrella-level information using two-pass approach.

        Uses smart_model for umbrella extraction since this is a high-impact
        phase - umbrella data propagates to all funds.

        **Two-Pass Strategy (for large documents):**
        1. Pass 1: Entity info (name, depositary, management_company) from
           bounded intro+outro pages (~20 pages max, always safe)
        2. Pass 2: Constraints (investment restrictions, leverage) from
           TOC-guided sections or search-based fallback

        This guarantees we never exceed token limits while maximizing coverage.
        """
        plan = self.context.plan
        self.logger.start_phase("Umbrella", 1, self.context.smart_model)

        page_count = self.context.pdf.page_count

        # Get skeleton for TOC-guided constraint extraction
        skeleton = getattr(self.context.state, 'skeleton', None)

        # Compute smart page plan using two-pass approach
        page_plan = select_umbrella_pages(skeleton, page_count)
        self.log(f"Umbrella page plan: {page_plan.summary()}")

        # =========================================================
        # Pass 1: Entity info (always bounded, never overflows)
        # =========================================================
        entity_pages = page_plan.entity_pages
        self.log(f"Pass 1: Entity info from {len(entity_pages)} pages")

        entity_text = self._read_page_blocks(entity_pages)
        self.log(f"Entity text: {len(entity_text)} chars")

        try:
            # Extract entity info (name, depositary, management_company, etc.)
            entity_data = await extract_umbrella(
                entity_text,
                model=self.context.smart_model,
                cost_tracker=self.context.cost_tracker,
            )
            self.context.umbrella_data = entity_data

        except Exception as e:
            self.log(f"Entity extraction failed: {e}", "error")
            self.context.umbrella_data = {
                "name": {
                    "value": plan.umbrella_name,
                    "source_page": None,
                    "source_quote": None,
                    "rationale": "Umbrella name from plan (entity extraction failed)",
                    "confidence": 1.0,
                }
            }

        # =========================================================
        # Pass 2: Constraints (TOC-guided or search fallback)
        # =========================================================
        constraint_data = await self._extract_umbrella_constraints(
            page_plan.constraint_pages,
            skeleton,
            page_count,
        )

        # Merge constraint data into umbrella data
        # (Only fill fields that are NOT_FOUND or missing in entity extraction)
        for field_name, value in constraint_data.items():
            existing = self.context.umbrella_data.get(field_name)
            if not existing or is_not_found(existing):
                self.context.umbrella_data[field_name] = value

        # =========================================================
        # Critic verification (on combined data)
        # =========================================================
        if self.context.use_critic:
            # Use entity text for critic (smaller, more focused)
            self.log("Running critic verification")
            self.context.umbrella_data, critic_result = await verify_and_correct(
                "umbrella",
                plan.umbrella_name,
                self.context.umbrella_data,
                entity_text,
                model=self.context.critic_model,
                cost_tracker=self.context.cost_tracker,
            )
            self.context.critic_results.append(critic_result)
            self.log(f"Critic confidence: {critic_result.overall_confidence:.0%}")

        # Ensure name is set with provenance
        existing_name = self.context.umbrella_data.get("name")
        name_value = get_raw_value(existing_name, None) if existing_name else None
        if not name_value or name_value == "NOT_FOUND":
            self.context.umbrella_data["name"] = {
                "value": plan.umbrella_name,
                "source_page": None,
                "source_quote": None,
                "rationale": "Umbrella name from plan (fallback)",
                "confidence": 1.0,
            }

        # Log umbrella result (extract raw value since name might be provenance dict)
        umbrella_name = get_raw_value(self.context.umbrella_data.get("name"), "Unknown")
        self.logger.phase_result("Umbrella", umbrella_name)

    async def _extract_umbrella_constraints(
        self,
        toc_constraint_pages: list[int],
        skeleton,
        total_pages: int,
    ) -> dict:
        """Extract umbrella-level constraints using TOC-guided pages or search.

        Args:
            toc_constraint_pages: Pages from TOC matching constraint patterns.
            skeleton: DocumentSkeleton for page resolution.
            total_pages: Total document pages.

        Returns:
            Dict with constraint fields (investment_restrictions, leverage, etc.)
        """
        from extractor.core.llm_client import LLMClient
        from extractor.prompts.reader_prompt import UMBRELLA_CONSTRAINT_PROMPT
        from extractor.pydantic_models.constraints import CONSTRAINT_FIELD_DESCRIPTIONS

        # Decide which pages to read
        if toc_constraint_pages:
            # TOC-guided: use pre-identified constraint sections
            pages_to_read = toc_constraint_pages
            self.log(f"Pass 2: Constraints from {len(pages_to_read)} TOC-guided pages")
        else:
            # Search fallback: find constraint-related pages via pattern search
            pages_to_read = await self._find_constraint_pages_via_search()
            if pages_to_read:
                self.log(f"Pass 2: Constraints from {len(pages_to_read)} search-found pages")
            else:
                self.log("Pass 2: No constraint pages found, skipping", "warning")
                return {}

        # Token limit check
        is_safe, estimated_tokens = check_token_limit(len(pages_to_read))
        if not is_safe:
            # Truncate to safe limit
            max_safe_pages = UmbrellaConfig.MAX_UMBRELLA_INPUT_TOKENS * UmbrellaConfig.CHARS_PER_TOKEN // 3000
            self.log(
                f"Constraint pages ({len(pages_to_read)}) would exceed token limit "
                f"(~{estimated_tokens:,} tokens), truncating to {max_safe_pages}",
                "warning"
            )
            pages_to_read = pages_to_read[:max_safe_pages]

        # Read constraint pages
        constraint_text = self._read_page_blocks(pages_to_read)
        self.log(f"Constraint text: {len(constraint_text)} chars")

        if not constraint_text.strip():
            return {}

        # Build fields description for prompt
        constraint_fields = [
            "investment_restrictions",
            "leverage_policy",
            "derivatives_usage",
            "borrowing_limit",
        ]
        fields_desc = "\n".join(
            f"- {field}: {CONSTRAINT_FIELD_DESCRIPTIONS.get(field, field)}"
            for field in constraint_fields
        )

        # Extract constraints
        client = LLMClient(cost_tracker=self.context.cost_tracker)
        try:
            response = await client.complete(
                system_prompt=UMBRELLA_CONSTRAINT_PROMPT.format(fields_desc=fields_desc),
                user_prompt=f"Extract umbrella-level constraints from:\n\n{constraint_text}",
                model=self.context.smart_model,
                agent="umbrella_constraints",
            )
            return response.content
        except Exception as e:
            self.log(f"Constraint extraction failed: {e}", "error")
            return {}

    async def _find_constraint_pages_via_search(self) -> list[int]:
        """Find constraint-related pages using pattern search.

        Fallback when no TOC sections match constraint patterns.

        Returns:
            List of page numbers containing constraint content.
        """
        search_context = self.context.create_search_context()

        # Search for constraint-related patterns
        all_hits = []
        for category in ["restriction", "leverage", "derivative"]:
            hits = search_context.search_patterns(category, max_results=15)
            all_hits.extend(hits)

        if not all_hits:
            return []

        # Dedupe and sort by page
        pages = sorted(set(hit["page"] for hit in all_hits))

        # Cap at configured limit
        max_pages = UmbrellaConfig.CONSTRAINT_SEARCH_FALLBACK_PAGES
        if len(pages) > max_pages:
            pages = pages[:max_pages]

        return pages

    def _read_page_blocks(self, pages: list[int]) -> str:
        """Read pages in contiguous blocks for efficiency.

        Groups consecutive pages and reads them as ranges to minimize
        PDF reader calls.

        Args:
            pages: List of page numbers (1-indexed).

        Returns:
            Concatenated text from all pages.
        """
        if not pages:
            return ""

        sorted_pages = sorted(pages)
        blocks: list[tuple[int, int]] = []
        start = sorted_pages[0]
        end = sorted_pages[0]

        for page in sorted_pages[1:]:
            if page == end + 1:
                end = page
            else:
                blocks.append((start, end))
                start = page
                end = page
        blocks.append((start, end))

        # Read each block
        texts = []
        for block_start, block_end in blocks:
            text = self.context.pdf.read_pages(block_start, block_end)
            texts.append(text)

        return "\n\n".join(texts)

    async def _extract_funds_with_recipes(
        self,
        recipes: list[FundExtractionRecipe],
        table_extractor: TableExtractor,
        parsed_tables: dict[str, ParsedTable],
    ):
        """Extract funds using recipe-based dispatch.

        For each fund:
        1. Iterate through field strategies
        2. Table lookup fields: query pre-parsed tables
        3. Text extraction fields: call LLM
        4. Cross-reference fields: read target pages and extract

        Args:
            recipes: Per-fund extraction recipes.
            table_extractor: Initialized TableExtractor with cache.
            parsed_tables: Pre-parsed broadcast tables.
        """
        # Limit funds if max_funds is set
        if self.context.max_funds and len(recipes) > self.context.max_funds:
            self.log(f"Limiting to {self.context.max_funds} funds (of {len(recipes)} total)")
            recipes = recipes[:self.context.max_funds]

        self.logger.start_phase("Fund Extraction (Recipe)", len(recipes), self.context.reader_model)

        # Create search context for fallback searches
        search_context = self.context.create_search_context()

        # Create knowledge context for knowledge-first extraction
        knowledge_ctx = KnowledgeContext(self.context.knowledge, self.context.store)

        # Create consolidator for conflict-aware fact recording
        consolidator = KnowledgeConsolidator(
            self.context.store,
            cost_tracker=self.context.cost_tracker,
            verification_model=self.context.reader_model,
        )

        async def extract_fund_with_recipe(recipe: FundExtractionRecipe) -> dict | None:
            async with self.context.semaphore:
                try:
                    if not recipe.dedicated_pages:
                        # No pages - return minimal data
                        return {
                            "name": {
                                "value": recipe.fund_name,
                                "source_page": None,
                                "source_quote": None,
                                "rationale": "Fund name from recipe (no dedicated pages)",
                                "confidence": 1.0,
                            },
                            "share_classes": []
                        }

                    # Read fund's dedicated pages
                    pages_text = self.context.pdf.read_pages(
                        recipe.dedicated_pages[0],
                        recipe.dedicated_pages[-1]
                    )

                    # Use extract_with_search for proper extraction (knowledge-first)
                    fund_data, share_classes = await extract_with_search(
                        recipe.fund_name,
                        pages_text,
                        search_context,
                        model=self.context.reader_model,
                        gleaning_passes=self.context.gleaning_passes,
                        cost_tracker=self.context.cost_tracker,
                        knowledge_ctx=knowledge_ctx,
                    )
                    fund_data["share_classes"] = share_classes

                    # Enhance share classes with table lookups using KG-RAG hints
                    table_lookup_count = 0
                    isin_table = parsed_tables.get("isin")
                    fee_table = parsed_tables.get("fee")

                    # Get graph hints for this fund
                    fund_context = self.context.store.get_fund_context(recipe.fund_name)
                    isin_hint = fund_context.isin_table if fund_context else None
                    fee_hint = fund_context.fee_table if fund_context else None

                    if isin_table or fee_table:
                        for sc in share_classes:
                            sc_name = get_raw_value(sc.get("name"), "")
                            if not sc_name:
                                continue

                            # ISIN lookup - use graph hints for column names
                            if isin_table and is_not_found(sc.get("isin")):
                                lookup_col = isin_hint.lookup_column if isin_hint else "Share Class"
                                # Try multiple column variants
                                row = (
                                    table_extractor.query(isin_table, lookup_col, sc_name) or
                                    table_extractor.query(isin_table, "Share Class", sc_name) or
                                    table_extractor.query(isin_table, "Class", sc_name)
                                )
                                if row:
                                    for isin_col in ["ISIN", "Isin", "ISIN Code"]:
                                        if isin_col in row and row[isin_col]:
                                            sc["isin"] = {
                                                "value": row[isin_col],
                                                "source_page": isin_table.source_pages[0] if hasattr(isin_table, 'source_pages') else None,
                                                "source_quote": None,
                                                "rationale": f"ISIN from table lookup (col: {lookup_col})",
                                                "confidence": 0.95,
                                                "source_type": "table",
                                            }
                                            # Record fact via consolidator (handles conflicts)
                                            await consolidator.add_fact(
                                                f"share_class:{recipe.fund_name}:{sc_name}",
                                                "isin",
                                                row[isin_col],
                                                source_page=isin_table.source_pages[0] if hasattr(isin_table, 'source_pages') else None,
                                                source_type=SourceType.TABLE,
                                                extraction_phase="extraction",
                                                confidence=0.95,
                                                pdf_reader=self.context.pdf,
                                            )
                                            table_lookup_count += 1
                                            break

                            # Fee lookup - use graph hints for column names
                            if fee_table and is_not_found(sc.get("management_fee")):
                                lookup_col = fee_hint.lookup_column if fee_hint else "Share Class"
                                row = (
                                    table_extractor.query(fee_table, lookup_col, sc_name) or
                                    table_extractor.query(fee_table, "Share Class", sc_name) or
                                    table_extractor.query(fee_table, "Class", sc_name)
                                )
                                if row:
                                    for fee_col in ["Management Fee", "Mgmt Fee", "Annual Fee", "TER"]:
                                        if fee_col in row and row[fee_col]:
                                            sc["management_fee"] = {
                                                "value": row[fee_col],
                                                "source_page": fee_table.source_pages[0] if hasattr(fee_table, 'source_pages') else None,
                                                "source_quote": None,
                                                "rationale": f"Fee from table lookup (col: {lookup_col})",
                                                "confidence": 0.95,
                                                "source_type": "table",
                                            }
                                            # Record fact via consolidator (handles conflicts)
                                            await consolidator.add_fact(
                                                f"share_class:{recipe.fund_name}:{sc_name}",
                                                "management_fee",
                                                row[fee_col],
                                                source_page=fee_table.source_pages[0] if hasattr(fee_table, 'source_pages') else None,
                                                source_type=SourceType.TABLE,
                                                extraction_phase="extraction",
                                                confidence=0.95,
                                                pdf_reader=self.context.pdf,
                                            )
                                            table_lookup_count += 1
                                            break

                    # Record extraction facts to graph (for LLM-extracted data, not just table lookups)
                    await self._record_extraction_facts(
                        consolidator, recipe.fund_name, fund_data, share_classes
                    )

                    # Post per-fund questions for NOT_FOUND fields so resolver knows what was tried
                    searched_pages = recipe.dedicated_pages or []
                    for sc in share_classes:
                        sc_name = get_raw_value(sc.get("name"), "")
                        if not sc_name:
                            continue
                        for fn, val in sc.items():
                            if fn == "name":
                                continue
                            if is_actionable_not_found(val):
                                self.context.knowledge.ask_question(
                                    question=f"Could not find {fn} for {sc_name}",
                                    field_name=fn,
                                    entity_name=recipe.fund_name,
                                    source_agent=f"extraction_{recipe.fund_name}",
                                    priority=QuestionPriority.LOW,
                                    pages_searched=searched_pages,
                                )

                    # Apply any additional broadcast data
                    apply_broadcast_data(fund_data, self.context.broadcast_data, recipe.fund_name)

                    # Critic verification (pass parsed tables as ground truth)
                    if self.context.use_critic:
                        fund_data, critic_results = await verify_fund_extraction(
                            recipe.fund_name,
                            fund_data,
                            pages_text,
                            share_classes,
                            model=self.context.critic_model,
                            cost_tracker=self.context.cost_tracker,
                            parsed_tables=parsed_tables,
                        )
                        self.context.critic_results.extend(critic_results)

                    self.logger.tick(f"{recipe.fund_name[:30]}")
                    return fund_data

                except Exception as e:
                    self.log(f"[{recipe.fund_name}] Recipe extraction failed: {e}", "error")
                    import json as _json
                    error_cat = ErrorCategory.LLM_PARSE if isinstance(e, (_json.JSONDecodeError, ValueError)) else ErrorCategory.LLM_API
                    self.context.errors.add(ExtractionError(
                        category=error_cat,
                        severity=ErrorSeverity.ERROR,
                        message=str(e),
                        phase="Fund Extraction (Recipe)",
                        entity_name=recipe.fund_name,
                    ))
                    return {
                        "name": {
                            "value": recipe.fund_name,
                            "source_page": recipe.dedicated_pages[0] if recipe.dedicated_pages else None,
                            "source_quote": None,
                            "rationale": "Fund name from recipe (extraction failed)",
                            "confidence": 1.0,
                        },
                        "share_classes": [],
                        "_error": str(e)
                    }

        # Run in parallel
        extraction_tasks = [extract_fund_with_recipe(recipe) for recipe in recipes]
        results = await asyncio.gather(*extraction_tasks, return_exceptions=True)

        # Collect results
        self.context.funds_data = []
        for i, r in enumerate(results):
            if isinstance(r, Exception):
                recipe = recipes[i]
                self.log(f"[{recipe.fund_name}] Gather exception: {r}", "error")
                self.context.funds_data.append({
                    "name": {
                        "value": recipe.fund_name,
                        "source_page": recipe.dedicated_pages[0] if recipe.dedicated_pages else None,
                        "source_quote": None,
                        "rationale": "Fund name from extraction recipe (extraction failed)",
                        "confidence": 1.0,
                    },
                    "share_classes": [],
                    "_error": str(r)
                })
            elif r is not None:
                self.context.funds_data.append(r)

        # Post-process
        self._enrich_with_knowledge_graph()

        # Run discovery if enabled
        if self.context.discover_bonus:
            await self._run_discovery(recipes)

        # Log results
        failed_count = len([r for r in results if isinstance(r, dict) and r.get("_error")])
        self.logger.phase_result(
            "Fund Extraction (Recipe)",
            f"{len(self.context.funds_data)} funds",
            failed=failed_count,
        )

        # Log KG-RAG graph stats
        graph_stats = self.context.store.stats()
        self.log(f"KG-RAG graph after extraction: {graph_stats['facts']} facts recorded")

        # Reconciliation checkpoint
        report = consolidator.reconcile()
        if not report.is_healthy():
            for issue in report.issues:
                self.log(f"RECONCILIATION ISSUE: {issue}", "warning")
        for warning in report.warnings:
            self.log(f"Reconciliation warning: {warning}", "debug")

        # Log conflict resolution summary
        if consolidator.conflict_resolutions:
            self.log(f"Resolved {len(consolidator.conflict_resolutions)} conflicts during extraction")
        if consolidator.entity_merges:
            self.log(f"Merged {len(consolidator.entity_merges)} duplicate entities")

        # Store audit trail for debugging
        self.context.state.extraction_audit = consolidator.get_audit_summary()

    def _enrich_with_knowledge_graph(self):
        """Enrich extracted data with knowledge graph information.

        - Marks external references for fields known to be in external docs
        - Records questions for fields that ACTUALLY need investigation
        """
        knowledge = self.context.knowledge

        # Get fields that are known to be external - don't create questions for these
        external_fields = {ref.field_name for ref in knowledge.external_refs}

        # Track ACTIONABLE NOT_FOUND counts (excludes external, not_applicable, etc.)
        actionable_not_found_counts: dict[str, int] = {}

        for fund_data in self.context.funds_data:
            # Check share classes for NOT_FOUND values
            for share_class in fund_data.get("share_classes", []):
                for field_name, value in share_class.items():
                    if field_name in ("name",):
                        continue

                    if is_not_found(value):
                        # Check if we have an external reference for this field
                        ext_ref = knowledge.get_external_ref_for_field(field_name)
                        if ext_ref:
                            # Update the value to include external reference info
                            if isinstance(value, dict):
                                value["not_found_reason"] = "in_external_doc"
                                value["external_reference"] = ext_ref.external_doc
                                if not value.get("rationale"):
                                    value["rationale"] = f"Documented in {ext_ref.external_doc}: '{ext_ref.source_quote[:50]}...'"
                        # Only count if actionable (not external, not_applicable, etc.)
                        elif is_actionable_not_found(value):
                            actionable_not_found_counts[field_name] = actionable_not_found_counts.get(field_name, 0) + 1

            # Check fund-level fields too
            for field_name in CONSTRAINT_FIELDS:
                if field_name in external_fields:
                    continue  # Skip external fields
                value = fund_data.get(field_name)
                if is_actionable_not_found(value):
                    actionable_not_found_counts[field_name] = actionable_not_found_counts.get(field_name, 0) + 1

        # Record questions ONLY for fields that actually need investigation
        total_funds = len(self.context.funds_data)
        if total_funds > 0:
            for field_name, count in actionable_not_found_counts.items():
                # Skip fields known to be external
                if field_name in external_fields:
                    continue
                # If more than 50% of funds are missing this field, record a question
                if count > total_funds * ExtractConfigConstants.PRIORITY_MEDIUM_THRESHOLD:
                    is_high_priority = count > total_funds * ExtractConfigConstants.PRIORITY_HIGH_THRESHOLD
                    priority = QuestionPriority.HIGH if is_high_priority else QuestionPriority.MEDIUM
                    # Collect all pages that were used across fund extractions
                    all_searched = set()
                    for fd in self.context.funds_data:
                        prev = self.context.knowledge.get_pages_already_searched(field_name, get_raw_value(fd.get("name"), None))
                        all_searched.update(prev)
                    knowledge.ask_question(
                        question=f"Where can {field_name} be found? Missing in {count}/{total_funds} funds.",
                        field_name=field_name,
                        entity_name="multiple_funds",
                        source_agent="extraction_phase",
                        priority=priority,
                        pages_searched=sorted(all_searched),
                    )

        # Log knowledge graph summary
        stats = knowledge.stats()
        unresolved = stats["unresolved_questions"]
        if unresolved > 0:
            self.log(f"Knowledge graph: {unresolved} unresolved questions")

    async def _run_discovery(self, recipes: list[FundExtractionRecipe]):
        """Discover PMS-relevant fields beyond the standard schema.

        Runs discovery on a sample of funds to find bonus fields.
        Results are stored in context.discovered_fields and context.schema_suggestions.

        Args:
            recipes: Fund extraction recipes (used for page info).
        """
        self.log("Running bonus field discovery...")

        # Sample first 5 funds for discovery
        for i, recipe in enumerate(recipes[:5]):
            if not recipe.dedicated_pages:
                continue

            pages_text = self.context.pdf.read_pages(
                recipe.dedicated_pages[0],
                recipe.dedicated_pages[-1]
            )

            # Get already-extracted fund data
            fund_data = {}
            if i < len(self.context.funds_data):
                fund_data = self.context.funds_data[i]

            try:
                discovered, suggestions = await discover_bonus_fields(
                    recipe.fund_name,
                    pages_text,
                    fund_data,
                    model=self.context.reader_model,
                    cost_tracker=self.context.cost_tracker,
                )
                self.context.discovered_fields.extend(discovered)
                self.context.schema_suggestions.extend(suggestions)
            except Exception as e:
                self.log(f"Discovery failed for {recipe.fund_name}: {e}", "warning")

        if self.context.discovered_fields:
            self.log(f"Discovered {len(self.context.discovered_fields)} bonus fields")

    async def _record_extraction_facts(
        self,
        consolidator: KnowledgeConsolidator,
        fund_name: str,
        fund_data: dict,
        share_classes: list[dict],
    ) -> None:
        """Record extraction facts to the knowledge graph.

        Registers share class entities then records LLM-extracted facts.
        """
        fund_fields = [
            "investment_objective", "benchmark", "leverage_limit",
            "derivative_exposure_limit", "borrowing_limit",
        ]

        share_class_fields = [
            "management_fee", "entry_fee", "exit_fee", "ongoing_charges",
            "performance_fee", "currency", "distribution_policy",
        ]

        # Record fund-level facts
        for field in fund_fields:
            value = fund_data.get(field)
            if value and isinstance(value, dict) and value.get("value") and not value.get("not_found_reason"):
                await consolidator.add_fact(
                    entity_key=f"fund:{fund_name}",
                    field_name=field,
                    value=value.get("value"),
                    source_page=value.get("source_page"),
                    source_quote=value.get("source_quote"),
                    source_type=SourceType.BODY_TEXT if not value.get("source_type") else SourceType(value.get("source_type")),
                    confidence=value.get("confidence", 0.8),
                    extraction_phase="extraction",
                    pdf_reader=self.context.pdf,
                )

        # Register share class entities, then record their facts
        for sc in share_classes:
            sc_name = sc.get("name", {})
            if isinstance(sc_name, dict):
                sc_name = sc_name.get("value", "")
            if not sc_name:
                continue

            # Register entity with fund-scoped key so facts aren't orphaned.
            # Facts use entity_key="share_class:{fund_name}:{sc_name}",
            # so the entity id must match: "{fund_name}:{sc_name}".
            scoped_name = f"{fund_name}:{sc_name}"
            consolidator.resolve_entity(
                entity_type="share_class",
                name=scoped_name,
                confidence=0.8,
                properties={"fund": fund_name, "class_name": sc_name},
            )

            for field in share_class_fields:
                value = sc.get(field)
                if value and isinstance(value, dict) and value.get("value") and not value.get("not_found_reason"):
                    await consolidator.add_fact(
                        entity_key=f"share_class:{fund_name}:{sc_name}",
                        field_name=field,
                        value=value.get("value"),
                        source_page=value.get("source_page"),
                        source_quote=value.get("source_quote"),
                        source_type=SourceType.BODY_TEXT if not value.get("source_type") else SourceType(value.get("source_type")),
                        confidence=value.get("confidence", 0.8),
                        extraction_phase="extraction",
                        pdf_reader=self.context.pdf,
                    )
