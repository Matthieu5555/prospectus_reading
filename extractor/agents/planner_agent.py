"""Planner agent — decides which PDF pages belong to which fund.

The planner exists so each downstream reader agent gets a focused context
window containing only the pages relevant to its fund.  Without planning,
the reader would have to scan the entire document, wasting tokens and
increasing the chance of cross-fund confusion.

It synthesizes exploration notes into a concrete extraction plan: a list of
funds, the page ranges each fund lives on, and pointers to any consolidated
tables (ISIN / fee) that should be parsed once and looked up per fund.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from extractor.core.document_knowledge import DocumentKnowledge

from extractor.core.config import DEFAULT_MODELS, ExtractionConfig
from extractor.core.cost_tracker import CostTracker
from extractor.core.fund_names import (
    FundNameMapping,
    deduplicate_from_exploration_notes,
    deduplicate_heuristic,
    strip_umbrella_prefix,
)
from extractor.core.llm_client import LLMClient
from extractor.pydantic_models.pipeline import (
    ExplorationNotes,
    FundExtractionTask,
    PageLookup,
    PlannerOutput,
)
from extractor.pydantic_models.exploration_models import DocumentSkeleton
from extractor.pydantic_models.logic_models import DocumentLogic
from extractor.pydantic_models.recipe_models import (
    TableLookupSource,
    TextExtractionSource,
    FieldStrategy,
    FundExtractionRecipe,
    ExtractionRecipeSet,
)
from extractor.prompts.planner_prompt import PLANNER_SYSTEM_PROMPT, build_planner_prompt


async def run_planner(
    exploration_notes: list[ExplorationNotes],
    total_pages: int,
    model: str = DEFAULT_MODELS["planning"],
    cost_tracker: CostTracker | None = None,
    canonical_funds: list[str] | None = None,
    fund_name_variants: dict[str, str] | None = None,
) -> PlannerOutput:
    """Run the planner to create an extraction plan.

    Args:
        exploration_notes: All exploration notes from Phase 1.
        total_pages: Total pages in the document.
        model: LLM model to use (needs good reasoning).
        cost_tracker: Optional tracker to record token usage.
        canonical_funds: Authoritative fund list from EntityResolution. If provided,
            the planner will use this instead of discovering funds itself.
        fund_name_variants: Name->canonical mapping from EntityResolution.

    Returns:
        PlannerOutput with extraction plan.
    """
    # Use EntityResolution's canonical funds if provided, otherwise fall back to
    # the old behavior of discovering funds via LLM deduplication
    if canonical_funds is not None and fund_name_variants is not None:
        # EntityResolution has already done the work - use its results
        name_mapping = FundNameMapping(
            canonical_names=canonical_funds,
            name_to_canonical=fund_name_variants,
            duplicates_found=len(fund_name_variants) - len(canonical_funds),
        )
    else:
        # Legacy path: deduplicate fund names from exploration using LLM
        name_mapping = await deduplicate_from_exploration_notes(
            exploration_notes, use_llm=True, model=model, cost_tracker=cost_tracker
        )

    notes_json = json.dumps([note.model_dump() for note in exploration_notes], indent=2)

    client = LLMClient(cost_tracker=cost_tracker)

    try:
        plan = await client.complete_structured(
            system_prompt=PLANNER_SYSTEM_PROMPT,
            user_prompt=build_planner_prompt(notes_json, total_pages, len(exploration_notes)),
            model=model,
            response_model=PlannerOutput,
            agent="planner",
        )

        # Post-process: normalize fund names using our deduplication
        plan = _normalize_plan_fund_names(plan, name_mapping)

        # Ensure ALL canonical funds have tasks (planner may have missed some)
        if canonical_funds:
            existing_task_funds = {task.fund_name for task in plan.fund_tasks}
            for fund_name in canonical_funds:
                if fund_name not in existing_task_funds:
                    plan.fund_tasks.append(FundExtractionTask(
                        fund_name=fund_name,
                        dedicated_pages=[],
                    ))
                if fund_name not in plan.fund_names:
                    plan.fund_names.append(fund_name)
            plan.total_funds = len(plan.fund_names)

        return plan

    except Exception as e:
        # Return a minimal plan on error
        fund_list = canonical_funds if canonical_funds else []
        if not fund_list:
            # Fall back to heuristic deduplication
            names_with_pages = []
            for note in exploration_notes:
                for fund in note.funds_mentioned:
                    names_with_pages.append((fund.name, fund.page))
            fallback_mapping = deduplicate_heuristic(names_with_pages)
            fund_list = fallback_mapping.canonical_names
            name_mapping = fallback_mapping

        return PlannerOutput(
            umbrella_name="Unknown",
            total_funds=len(fund_list),
            fund_names=fund_list,
            umbrella_pages=[1, 2, 3],  # Guess first pages - see ExtractionConfig for rationale
            fund_tasks=[
                FundExtractionTask(fund_name=name, dedicated_pages=[])
                for name in fund_list
            ],
            fund_name_variants=name_mapping.name_to_canonical if name_mapping else {},
            observations=[
                f"Planner error: {e}",
            ],
        )


def _normalize_plan_fund_names(plan: PlannerOutput, name_mapping: FundNameMapping) -> PlannerOutput:
    """Normalize fund names in the plan using pre-computed deduplication.

    The LLM planner may return names that are duplicates or variants.
    This function normalizes them to canonical forms.

    Args:
        plan: The planner output to normalize.
        name_mapping: Pre-computed name deduplication mapping.

    Returns:
        Plan with normalized fund names.
    """
    # Normalize fund_names list
    seen_canonical = set()
    normalized_names = []
    for name in plan.fund_names:
        canonical = name_mapping.name_to_canonical.get(name, name)
        if canonical not in seen_canonical:
            seen_canonical.add(canonical)
            normalized_names.append(canonical)

    # Normalize fund_tasks
    seen_task_names = set()
    normalized_tasks = []
    for task in plan.fund_tasks:
        canonical = name_mapping.name_to_canonical.get(task.fund_name, task.fund_name)
        if canonical not in seen_task_names:
            seen_task_names.add(canonical)
            # Create new task with canonical name
            normalized_tasks.append(FundExtractionTask(
                fund_name=canonical,
                dedicated_pages=task.dedicated_pages,
                isin_lookup=task.isin_lookup,
                fee_lookup=task.fee_lookup,
                share_class_lookup=task.share_class_lookup,
            ))

    # Update plan
    plan.fund_names = normalized_names
    plan.fund_tasks = normalized_tasks
    plan.total_funds = len(normalized_names)
    plan.fund_name_variants = name_mapping.name_to_canonical

    # Add observation about deduplication
    if name_mapping.duplicates_found > 0:
        plan.observations.append(
            f"Deduplicated {name_mapping.duplicates_found} duplicate fund name variants"
        )

    return plan


def compute_page_ranges(
    exploration_notes: list[ExplorationNotes],
    total_pages: int,
    skeleton: DocumentSkeleton | None = None,
    default_section_size: int = ExtractionConfig.UMBRELLA_MIN_GAP,
) -> dict[str, list[int]]:
    """Compute page ranges for funds using skeleton as primary source.

    Strategy (in order of preference):
    1. TOC skeleton - section boundaries from document structure (most reliable)
    2. Explorer page_index - pages tagged with fund_name (filtered to first block)
    3. funds_mentioned - infer ranges from first mention of each fund

    Args:
        exploration_notes: All exploration notes.
        total_pages: Total pages in document.
        skeleton: DocumentSkeleton with TOC-derived section boundaries.
        default_section_size: Default pages per fund if can't infer.

    Returns:
        Dict mapping fund name to list of pages.
    """
    page_ranges: dict[str, list[int]] = {}

    # Strategy 1: Use skeleton (TOC-based) - most reliable
    if skeleton and skeleton.sections:
        page_ranges = _compute_ranges_from_skeleton(skeleton)
        if page_ranges:
            return page_ranges

    # Strategy 2: Use page_index from explorers (with first-contiguous-block filter)
    page_ranges = _compute_ranges_from_page_index(exploration_notes)
    if page_ranges:
        return page_ranges

    # Strategy 3: Fall back to funds_mentioned (legacy heuristic)
    page_ranges = _compute_ranges_from_funds_mentioned(
        exploration_notes, total_pages, default_section_size
    )
    return page_ranges


def _compute_ranges_from_skeleton(skeleton: DocumentSkeleton) -> dict[str, list[int]]:
    """Extract fund page ranges from TOC skeleton.

    Uses section boundaries defined in the document's table of contents.
    This is the most reliable source since it comes from the document itself.

    Returns both exact names and normalized names to help with matching.
    """
    page_ranges: dict[str, list[int]] = {}

    for section in skeleton.sections:
        # Only consider fund sections
        if section.section_type == "fund_section" and section.fund_name:
            fund_name = section.fund_name
            pages = list(range(section.page_start, section.page_end + 1))

            # Store with exact name
            page_ranges[fund_name] = pages

            # Also store with normalized name (helps match variant names)
            # e.g., "Europe Select Equity Fund" from TOC should match
            # "JPMorgan Investment Funds - Europe Select Equity Fund" from planner
            normalized = strip_umbrella_prefix(fund_name)
            if normalized != fund_name:
                page_ranges[normalized] = pages

    return page_ranges




def _compute_ranges_from_page_index(
    exploration_notes: list[ExplorationNotes],
) -> dict[str, list[int]]:
    """Extract fund page ranges from explorer page_index.

    Collects pages tagged with fund_name, then filters to first contiguous block
    to exclude appendix mentions.
    """
    page_ranges: dict[str, list[int]] = {}

    for note in exploration_notes:
        for entry in note.page_index:
            if entry.fund_name:
                fund_name = entry.fund_name
                if fund_name not in page_ranges:
                    page_ranges[fund_name] = []
                if entry.page not in page_ranges[fund_name]:
                    page_ranges[fund_name].append(entry.page)

    # Filter to first contiguous block for each fund
    for fund_name in page_ranges:
        sorted_pages = sorted(page_ranges[fund_name])
        page_ranges[fund_name] = _get_first_contiguous_block(sorted_pages)

    return page_ranges


def _compute_ranges_from_funds_mentioned(
    exploration_notes: list[ExplorationNotes],
    total_pages: int,
    default_section_size: int,
) -> dict[str, list[int]]:
    """Infer page ranges from funds_mentioned entries.

    Uses the first mention of each fund and infers the section extends
    until the next fund starts. Last resort fallback.
    """
    page_ranges: dict[str, list[int]] = {}
    funds = []  # [(name, page)]

    for note in exploration_notes:
        for fund in note.funds_mentioned:
            if fund.has_dedicated_section:
                funds.append((fund.name, fund.page))

    if not funds:
        return {}

    funds.sort(key=lambda x: x[1])

    # Compute page ranges by inferring from next fund's start
    for i, (name, start_page) in enumerate(funds):
        if i + 1 < len(funds):
            next_start = funds[i + 1][1]
            end_page = max(start_page, next_start - 1)
        else:
            end_page = min(start_page + default_section_size - 1, total_pages)

        # Limit section size to avoid reading too many pages
        end_page = min(end_page, start_page + 5)  # Max 6 pages per fund

        page_ranges[name] = list(range(start_page, end_page + 1))

    return page_ranges


def _get_first_contiguous_block(pages: list[int]) -> list[int]:
    """Extract the first contiguous block of pages.

    Given pages [5, 6, 75, 76, 77], returns [5, 6].
    This filters out appendix mentions that aren't part of the fund section.
    """
    if not pages:
        return []

    result = [pages[0]]
    for i in range(1, len(pages)):
        if pages[i] == pages[i - 1] + 1:
            result.append(pages[i])
        else:
            # Gap found - stop at first block
            break
    return result


def compute_table_pages(
    exploration_notes: list[ExplorationNotes],
) -> dict[str, list[int]]:
    """Extract table page locations from page_index.

    Args:
        exploration_notes: All exploration notes.

    Returns:
        Dict mapping table type to list of pages, e.g.:
        {"fee_table": [52, 53], "isin_table": [200, 201]}
    """
    table_pages: dict[str, list[int]] = {}

    for note in exploration_notes:
        for entry in note.page_index:
            if entry.content_type in ("fee_table", "isin_table", "share_class_table"):
                table_type = entry.content_type
                if table_type not in table_pages:
                    table_pages[table_type] = []
                if entry.page not in table_pages[table_type]:
                    table_pages[table_type].append(entry.page)

    # Sort pages for each table type
    for table_type in table_pages:
        table_pages[table_type].sort()

    return table_pages


def _compute_umbrella_pages(
    plan_umbrella_pages: list[int],
    exploration_notes: list[ExplorationNotes],
    total_pages: int,
) -> list[int]:
    """Compute validated umbrella pages from exploration data.

    Strategy:
    1. **Validate planner pages** - reject if excessive (> MAX_UMBRELLA_PAGES)
    2. If planner's umbrella_pages is reasonable (non-empty, <= MAX), use it
    3. Collect umbrella_info_pages from explorers as fallback
    4. If nothing found, fall back to heuristic: first N + last M pages

    This ensures we never read excessive pages for umbrella extraction,
    preventing token explosion errors like BadRequestError (292k tokens).

    **Key insight**: The extraction phase now uses a two-pass approach with
    smart page selection (umbrella_page_selector.py), so this validation is
    a safety net for the planner output, not the primary page selection logic.

    Args:
        plan_umbrella_pages: Pages from LLM planner (may be wrong).
        exploration_notes: Exploration notes with umbrella_info_pages.
        total_pages: Total pages in document.

    Returns:
        Validated list of umbrella pages, capped at MAX_UMBRELLA_PAGES.
    """
    import logging
    logger = logging.getLogger(__name__)

    max_pages = ExtractionConfig.MAX_UMBRELLA_PAGES
    intro_pages = ExtractionConfig.UMBRELLA_INTRO_PAGES
    end_pages = ExtractionConfig.UMBRELLA_END_PAGES

    # Validate planner pages first
    planner_pages = plan_umbrella_pages or []
    planner_page_count = len(planner_pages)

    # CRITICAL: Reject excessive planner output (this indicates a planner bug)
    if planner_page_count > max_pages:
        logger.warning(
            f"Planner returned {planner_page_count} umbrella pages (> {max_pages}), "
            f"rejecting and using heuristic. This indicates a planner bug."
        )
        planner_pages = []
        planner_page_count = 0

    # Additional check: reject if planner returned nearly ALL pages
    # (e.g., [1, 2, 3, ..., 261] for a 261-page document)
    if planner_page_count > 0 and planner_page_count >= total_pages * 0.9:
        logger.warning(
            f"Planner returned {planner_page_count}/{total_pages} pages "
            f"({planner_page_count/total_pages:.0%} of document), "
            f"rejecting as likely planner bug."
        )
        planner_pages = []
        planner_page_count = 0

    # Check for large span (e.g., pages 1-250 out of 261)
    if planner_page_count > 0:
        span = planner_pages[-1] - planner_pages[0] + 1
        if span > total_pages * 0.8:
            logger.warning(
                f"Planner umbrella pages span {span} pages "
                f"(80%+ of {total_pages}-page document), "
                f"rejecting as likely planner bug."
            )
            planner_pages = []
            planner_page_count = 0

    # If planner pages are valid and reasonable, use them
    if planner_page_count > 0 and planner_page_count <= max_pages:
        return sorted(planner_pages)

    # Collect umbrella_info_pages from all exploration notes
    collected_pages: set[int] = set()
    for note in exploration_notes:
        collected_pages.update(note.umbrella_info_pages)

    # Also collect pages with "general_info" content type from page_index
    for note in exploration_notes:
        for entry in note.page_index:
            if entry.content_type == "general_info":
                collected_pages.add(entry.page)

    if collected_pages:
        # Use exploration-discovered umbrella pages
        result = sorted(collected_pages)
        if len(result) > max_pages:
            # Too many - take first chunk + last chunk
            logger.info(
                f"Exploration found {len(result)} umbrella pages, "
                f"truncating to {max_pages}"
            )
            return result[:intro_pages] + result[-(max_pages - intro_pages):]
        return result

    # Fallback heuristic: first N pages + last M pages
    # This covers legal disclaimers at start + appendices at end
    logger.info(
        f"No valid umbrella pages from planner or exploration, "
        f"using heuristic (first {intro_pages} + last {end_pages})"
    )
    fallback: list[int] = []

    # First few pages (legal structure, UCITS status)
    fallback.extend(range(1, min(intro_pages + 1, total_pages + 1)))

    # Last pages (service providers, management company often in appendices)
    end_start = max(total_pages - end_pages + 1, intro_pages + 1)
    if end_start <= total_pages:
        fallback.extend(range(end_start, total_pages + 1))

    return sorted(set(fallback))


def enrich_plan_with_pages(
    plan: PlannerOutput,
    exploration_notes: list[ExplorationNotes],
    total_pages: int,
    skeleton: DocumentSkeleton | None = None,
) -> PlannerOutput:
    """Enrich planner output with computed page ranges and lookup info.

    Uses skeleton (TOC) as primary source, with fallbacks:
    1. Skeleton sections - TOC-derived fund section boundaries (most reliable)
    2. Explorer page_index - pages tagged with fund_name
    3. funds_mentioned - infer ranges from first mention

    Args:
        plan: The planner output to enrich.
        exploration_notes: Exploration notes with page_index.
        total_pages: Total pages in document.
        skeleton: DocumentSkeleton with TOC section boundaries.

    Returns:
        Enriched planner output.
    """
    # Validate and compute umbrella pages
    plan.umbrella_pages = _compute_umbrella_pages(
        plan.umbrella_pages,
        exploration_notes,
        total_pages,
    )

    # Compute page ranges - skeleton is primary source, page_index is fallback
    page_ranges = compute_page_ranges(exploration_notes, total_pages, skeleton)

    # Compute table pages from page_index
    table_pages = compute_table_pages(exploration_notes)

    # Enrich fund tasks
    for task in plan.fund_tasks:
        # ALWAYS use computed page ranges if available (they're more reliable than LLM guesses)
        # Try exact match first, then normalized name (handles "JPM - Fund X" vs "Fund X")
        pages = _lookup_fund_pages(task.fund_name, page_ranges)
        if pages:
            task.dedicated_pages = pages

        # Add fee lookup if missing and we have fee_table pages
        if not task.fee_lookup and "fee_table" in table_pages:
            task.fee_lookup = PageLookup(
                table_pages=table_pages["fee_table"],
                lookup_column="Fund Name",
                lookup_value=task.fund_name,
            )

        # Add ISIN lookup if missing and we have isin_table pages
        if not task.isin_lookup and "isin_table" in table_pages:
            task.isin_lookup = PageLookup(
                table_pages=table_pages["isin_table"],
                lookup_column="Fund Name",
                lookup_value=task.fund_name,
            )

        # Add share class lookup if missing and we have share_class_table pages
        if not task.share_class_lookup and "share_class_table" in table_pages:
            task.share_class_lookup = PageLookup(
                table_pages=table_pages["share_class_table"],
                lookup_column="Fund Name",
                lookup_value=task.fund_name,
            )

    return plan


def _lookup_fund_pages(fund_name: str, page_ranges: dict[str, list[int]]) -> list[int] | None:
    """Look up fund pages, trying both exact and normalized names."""
    # Try exact match
    if fund_name in page_ranges:
        return page_ranges[fund_name]

    # Try normalized name
    normalized = strip_umbrella_prefix(fund_name)
    if normalized in page_ranges:
        return page_ranges[normalized]

    return None


# Recipe Generation

def generate_recipes(
    plan: PlannerOutput,
    document_logic: DocumentLogic,
    knowledge: "DocumentKnowledge | None" = None,
) -> ExtractionRecipeSet:
    """Generate extraction recipes from plan and document logic.

    Prefers knowledge graph for table locations (populated from exploration
    discoveries). Falls back to DocumentLogic when knowledge is unavailable.

    Args:
        plan: PlannerOutput with fund tasks and page assignments.
        document_logic: DocumentLogic with extraction strategies from LLM analysis.
        knowledge: Optional DocumentKnowledge for querying discovered table locations.

    Returns:
        ExtractionRecipeSet with per-fund recipes.
    """
    fund_recipes = []

    for task in plan.fund_tasks:
        recipe = _create_fund_recipe(task, document_logic, knowledge)
        fund_recipes.append(recipe)

    # Create broadcast table sources for upfront parsing
    # Prefer knowledge graph discoveries over DocumentLogic templates
    broadcast_tables = []

    # Get table info from knowledge graph if available
    isin_info = _get_table_info_from_knowledge(knowledge, "isin") if knowledge else None
    fee_info = _get_table_info_from_knowledge(knowledge, "fee") if knowledge else None

    # ISIN broadcast table
    isin_pages = None
    isin_lookup_col = None
    if isin_info:
        # Use knowledge graph discovery
        isin_pages = isin_info["pages"]
        isin_lookup_col = isin_info.get("lookup_column")
    elif document_logic.isin_strategy.location in ("consolidated_table", "appendix"):
        # Fall back to DocumentLogic
        isin_pages = document_logic.isin_strategy.table_pages
        isin_lookup_col = document_logic.isin_strategy.lookup_column

    if isin_pages:
        broadcast_tables.append(TableLookupSource(
            table_type="isin",
            table_pages=isin_pages,
            lookup_column=isin_lookup_col or "Fund Name",
            lookup_value="",
            target_columns=[document_logic.isin_strategy.value_column],
        ))

    # Fee broadcast table
    fee_pages = None
    fee_lookup_col = None
    if fee_info:
        # Use knowledge graph discovery
        fee_pages = fee_info["pages"]
        fee_lookup_col = fee_info.get("lookup_column")
    elif document_logic.fee_strategy.location in ("consolidated_table", "appendix"):
        # Fall back to DocumentLogic
        fee_pages = document_logic.fee_strategy.table_pages
        fee_lookup_col = document_logic.fee_strategy.lookup_column

    if fee_pages:
        broadcast_tables.append(TableLookupSource(
            table_type="fee",
            table_pages=fee_pages,
            lookup_column=fee_lookup_col or "Fund Name",
            lookup_value="",
            target_columns=list(document_logic.fee_strategy.fee_columns.values()),
        ))

    return ExtractionRecipeSet(
        umbrella_name=plan.umbrella_name,
        fund_recipes=fund_recipes,
        broadcast_tables=broadcast_tables,
    )


def _get_table_info_from_knowledge(
    knowledge: "DocumentKnowledge",
    field_name: str,
) -> dict | None:
    """Query knowledge graph for table location info.

    Returns dict with pages, columns, lookup_column if found.
    """
    from extractor.core import FindingType

    # Find table location findings for this field
    table_findings = [
        f for f in knowledge.findings
        if f.finding_type == FindingType.TABLE_LOCATION
        and f.field_name == field_name
        and f.pages
    ]

    if not table_findings:
        return None

    # Combine pages from all findings
    all_pages = set()
    columns = []
    has_fund_name_column = False

    for finding in table_findings:
        all_pages.update(finding.pages)
        if finding.metadata:
            if "columns" in finding.metadata and not columns:
                columns = finding.metadata["columns"]
            if finding.metadata.get("has_fund_name_column"):
                has_fund_name_column = True

    # Infer lookup column from columns if we have a fund name column
    lookup_column = None
    if has_fund_name_column and columns:
        # Look for common fund/share class column names
        for col in columns:
            col_lower = col.lower()
            if any(kw in col_lower for kw in ["fund", "sub-fund", "share class", "class"]):
                lookup_column = col
                break

    return {
        "pages": sorted(all_pages),
        "columns": columns,
        "lookup_column": lookup_column,
        "has_fund_name_column": has_fund_name_column,
    }


def _create_fund_recipe(
    task: FundExtractionTask,
    logic: DocumentLogic,
    knowledge: "DocumentKnowledge | None" = None,
) -> FundExtractionRecipe:
    """Create extraction recipe for one fund based on DocumentLogic.

    Args:
        task: FundExtractionTask with page assignments.
        logic: DocumentLogic with extraction strategies.
        knowledge: Optional knowledge graph for additional field locations.

    Returns:
        FundExtractionRecipe with per-field strategies.
    """
    strategies = []

    # ISIN strategy
    strategies.append(_create_isin_strategy(task, logic))

    # Fee strategies
    strategies.extend(_create_fee_strategies(task, logic))

    # Text extraction fields (always from dedicated pages)
    text_fields = [
        "investment_objective",
        "investment_policy",
        "risk_profile",
        "benchmark",
        "base_currency",
    ]
    for field in text_fields:
        strategies.append(FieldStrategy(
            field_name=field,
            source_type="text_extraction",
            source=TextExtractionSource(
                pages=task.dedicated_pages,
                field_hint=field.replace("_", " "),
            ),
            required=False,
            fallback_to_text=False,
        ))

    # Share class strategy
    share_class_strategy = FieldStrategy(
        field_name="share_classes",
        source_type="text_extraction",
        source=TextExtractionSource(
            pages=task.dedicated_pages,
            field_hint="share classes with names, ISINs, currencies, fees",
        ),
        required=True,
    )

    return FundExtractionRecipe(
        fund_name=task.fund_name,
        dedicated_pages=task.dedicated_pages,
        field_strategies=strategies,
        share_class_strategy=share_class_strategy,
    )


def _create_isin_strategy(task: FundExtractionTask, logic: DocumentLogic) -> FieldStrategy:
    """Create ISIN extraction strategy based on document logic."""
    isin_strat = logic.isin_strategy

    if isin_strat.location == "consolidated_table" and isin_strat.table_pages:
        return FieldStrategy(
            field_name="isin",
            source_type="table_lookup",
            source=TableLookupSource(
                table_type="isin",
                table_pages=isin_strat.table_pages,
                lookup_column=isin_strat.lookup_column or "Fund Name",
                lookup_value=task.fund_name,
                target_columns=[isin_strat.value_column],
            ),
            required=False,
            fallback_to_text=True,
        )

    # Fall back to text extraction
    return FieldStrategy(
        field_name="isin",
        source_type="text_extraction",
        source=TextExtractionSource(
            pages=task.dedicated_pages,
            field_hint="ISIN codes (12 character alphanumeric)",
        ),
        required=False,
    )


def _create_fee_strategies(task: FundExtractionTask, logic: DocumentLogic) -> list[FieldStrategy]:
    """Create fee extraction strategies based on document logic."""
    strategies = []
    fee_strat = logic.fee_strategy

    fee_fields = [
        ("management_fee", "Management Fee"),
        ("performance_fee", "Performance Fee"),
        ("subscription_fee", "Subscription Fee"),
        ("redemption_fee", "Redemption Fee"),
    ]

    if fee_strat.location == "consolidated_table" and fee_strat.table_pages:
        for field_name, default_col in fee_fields:
            # Use column mapping from logic if available
            col_name = fee_strat.fee_columns.get(field_name, default_col)
            strategies.append(FieldStrategy(
                field_name=field_name,
                source_type="table_lookup",
                source=TableLookupSource(
                    table_type="fee",
                    table_pages=fee_strat.table_pages,
                    lookup_column=fee_strat.lookup_column or "Fund Name",
                    lookup_value=task.fund_name,
                    target_columns=[col_name],
                ),
                required=False,
                fallback_to_text=True,
            ))
    else:
        # Text extraction for fees
        for field_name, _ in fee_fields:
            strategies.append(FieldStrategy(
                field_name=field_name,
                source_type="text_extraction",
                source=TextExtractionSource(
                    pages=task.dedicated_pages,
                    field_hint=field_name.replace("_", " "),
                ),
                required=False,
            ))

    return strategies
