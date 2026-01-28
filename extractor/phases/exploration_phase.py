"""Exploration phase - page-level document indexing.

Always runs explorers to create a page-by-page index of document content.
No TOC validation or conditional paths - just straightforward exploration.

Each explorer outputs:
- page_index: What's on each page (fund_section, fee_table, isin_table, etc.)
- funds_mentioned: Fund names found
- tables: Table structures discovered
- cross_references: References to other sections
"""

import asyncio
from dataclasses import dataclass

from extractor.phases.phase_base import PhaseRunner
from extractor.pydantic_models import (
    ExplorationNotes,
    DocumentInventory,
    DocumentSkeleton,
    FieldPresence,
    ExternalFieldRef,
    ExplicitAbsence,
)
from extractor.agents import run_explorer, run_explorer_with_critic, run_explorer_with_gleaning
from extractor.core import (
    ExtractionError, ErrorCategory, ErrorSeverity,
    Finding, FindingType,
    skeleton_from_native_toc,
)
from extractor.core.config import adaptive_chunk_size


@dataclass
class ExplorationResult:
    """Result from the exploration phase."""

    exploration_notes: list[ExplorationNotes]
    skeleton: DocumentSkeleton | None
    total_funds_found: int
    total_tables_found: int
    failed_chunks: int


class ExplorationPhase(PhaseRunner[ExplorationResult]):
    """Page-level document indexing via parallel explorers.

    Always runs explorers - no conditional paths based on TOC quality.
    Each explorer creates a page-by-page index of what exists in their chunk.
    """

    name = "Exploration"

    async def run(self) -> ExplorationResult:
        """Run parallel explorers to create page-level document index.

        Returns:
            ExplorationResult with exploration notes and optional skeleton.
        """
        total_pages = self.context.pdf.page_count

        # Build skeleton from native TOC if available (for cross-ref resolution)
        skeleton = None
        native_toc = self.context.pdf.get_toc()
        if native_toc:
            skeleton = skeleton_from_native_toc(native_toc, total_pages)
            self.log(f"Native TOC: {len(native_toc)} entries (used for cross-ref context)")

        # Use adaptive chunk size for large documents to prevent token overflow
        effective_chunk_size = adaptive_chunk_size(total_pages, base_size=self.context.chunk_size)
        chunks = self.context.pdf.get_page_chunks(effective_chunk_size)

        if effective_chunk_size != self.context.chunk_size:
            self.log(f"Adaptive chunking: reduced from {self.context.chunk_size} to {effective_chunk_size} pages for {total_pages}-page document")
        self.log(f"Exploring {total_pages} pages in {len(chunks)} chunks of {effective_chunk_size} pages")

        self.start(len(chunks), model=self.context.exploration_model)

        # Build skeleton context string for explorers (helps with cross-refs)
        skeleton_context = skeleton.to_explorer_context() if skeleton else ""

        # Check if exploration gleaning is enabled (reuse extraction gleaning setting)
        exploration_gleaning = self.context.gleaning_passes

        async def explore_chunk(start: int, end: int) -> ExplorationNotes | None:
            async with self.context.semaphore:
                try:
                    pages_text = self.context.pdf.read_pages(start, end)

                    # Add table context from pre-scan (if available)
                    table_ctx = self._build_table_context(start, end)
                    combined_context = skeleton_context + table_ctx

                    if self.context.use_critic:
                        notes = await run_explorer_with_critic(
                            pages_text, start, end,
                            model=self.context.exploration_model,
                            critic_model=self.context.critic_model,
                            cost_tracker=self.context.cost_tracker,
                            skeleton_context=combined_context,
                        )
                    elif exploration_gleaning > 1:
                        # Use gleaning to catch missed external references
                        notes = await run_explorer_with_gleaning(
                            pages_text, start, end,
                            model=self.context.exploration_model,
                            cost_tracker=self.context.cost_tracker,
                            skeleton_context=combined_context,
                            gleaning_passes=exploration_gleaning,
                        )
                    else:
                        notes = await run_explorer(
                            pages_text, start, end,
                            model=self.context.exploration_model,
                            cost_tracker=self.context.cost_tracker,
                            skeleton_context=combined_context,
                        )

                    self.logger.tick(f"pages {start}-{end}")
                    return notes

                except Exception as e:
                    self.log(f"Explorer failed for pages {start}-{end}: {e}", "error")
                    self.context.errors.add(ExtractionError(
                        category=ErrorCategory.LLM_API,
                        severity=ErrorSeverity.WARNING,
                        message=str(e),
                        phase=self.name,
                        page_range=(start, end),
                    ))
                    return None

        # Run explorers in parallel
        tasks = [explore_chunk(start, end) for start, end in chunks]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter successful results and track failures
        exploration_notes = []
        failed_ranges = []
        for i, r in enumerate(results):
            if isinstance(r, Exception):
                start, end = chunks[i]
                self.log(f"Explorer gather exception for pages {start}-{end}: {r}", "error")
                failed_ranges.append((start, end))
            elif r is not None:
                exploration_notes.append(r)
            else:
                failed_ranges.append(chunks[i])

        failed_chunks = len(failed_ranges)

        # Handle failures
        if failed_ranges:
            warning = f"WARNING: {failed_chunks} explorer(s) failed for pages: {failed_ranges}. Some funds may be missing."
            self.log(warning, "warning")

            if exploration_notes:
                exploration_notes[0].observations.append(warning)
            else:
                error_note = ExplorationNotes(
                    page_start=1,
                    page_end=self.context.pdf.page_count,
                    observations=[f"CRITICAL: All {failed_chunks} explorers failed.", warning],
                )
                exploration_notes.append(error_note)
                self.log("CRITICAL: All explorers failed!", "error")

        # Resolve cross-references if we have a skeleton
        if skeleton:
            exploration_notes = self._resolve_cross_references(exploration_notes, skeleton)

        # Store in context
        self.context.exploration_notes = exploration_notes

        # Populate knowledge graph
        self._populate_knowledge_graph(exploration_notes)

        if skeleton:
            self.context.knowledge.appendix_map = skeleton.appendix_map

        # Calculate totals
        total_funds = sum(len(n.funds_mentioned) for n in exploration_notes)
        total_tables = sum(len(n.tables) for n in exploration_notes)

        # Log result
        self.logger.phase_result(
            "Exploration",
            f"{len(exploration_notes)} explorers completed",
            funds_found=total_funds,
            tables_found=total_tables,
            failed=failed_chunks,
            toc_source=skeleton.toc_source if skeleton else "none",
        )

        return ExplorationResult(
            exploration_notes=exploration_notes,
            skeleton=skeleton,
            total_funds_found=total_funds,
            total_tables_found=total_tables,
            failed_chunks=failed_chunks,
        )

    def _build_table_context(self, start_page: int, end_page: int) -> str:
        """Build context string describing tables detected in the page range.

        This helps explorers know where tables are located so they can
        accurately report table types and locations.

        Args:
            start_page: First page of the chunk (1-indexed).
            end_page: Last page of the chunk (1-indexed).

        Returns:
            Context string describing detected tables, or empty string if none.
        """
        scanned = getattr(self.context.state, "scanned_tables", [])
        if not scanned:
            return ""

        # Filter to tables in this page range
        tables_in_range = [t for t in scanned if start_page <= t.page <= end_page]
        if not tables_in_range:
            return ""

        ctx = "\n\nTABLES DETECTED ON THESE PAGES:\n"
        for t in tables_in_range:
            # Show first 4 columns
            cols = ", ".join(t.columns[:4])
            if len(t.columns) > 4:
                cols += "..."
            ctx += f"- Page {t.page}: {t.col_count} cols ({cols}), {t.row_count} rows\n"

        return ctx

    def _aggregate_inventories(self, exploration_notes: list[ExplorationNotes]) -> DocumentInventory:
        """Aggregate inventories from all explorers into a master inventory.

        Aggregation strategy:
        - fields_present: UNION - if any explorer found it, it's present
        - fields_absent: Only include if NO explorer found it present
        - fields_external: UNION - any external reference is recorded
        - explicit_absences: UNION - any explicit absence is recorded
        - toc_entries: UNION with deduplication
        """
        # Track fields found by field_name for merging
        fields_by_name: dict[str, FieldPresence] = {}
        fields_external: dict[str, ExternalFieldRef] = {}
        explicit_absences: dict[str, ExplicitAbsence] = {}
        toc_entries: dict[str, any] = {}  # section_name -> TOCEntry

        for notes in exploration_notes:
            inv = notes.inventory

            # Aggregate fields_present (UNION, merge pages)
            for fp in inv.fields_present:
                if fp.field_name in fields_by_name:
                    # Merge pages
                    existing = fields_by_name[fp.field_name]
                    merged_pages = sorted(set(existing.pages) | set(fp.pages))
                    fields_by_name[fp.field_name] = FieldPresence(
                        field_name=fp.field_name,
                        pages=merged_pages,
                        table_type=fp.table_type or existing.table_type,
                        confidence=max(fp.confidence, existing.confidence),
                        notes=fp.notes or existing.notes,
                    )
                else:
                    fields_by_name[fp.field_name] = fp

            # Aggregate fields_external (UNION)
            for ef in inv.fields_external:
                if ef.field_name not in fields_external:
                    fields_external[ef.field_name] = ef

            # Aggregate explicit_absences (UNION)
            for ea in inv.explicit_absences:
                if ea.field_name not in explicit_absences:
                    explicit_absences[ea.field_name] = ea

            # Aggregate toc_entries (UNION with deduplication)
            for entry in inv.toc_entries:
                if entry.section_name not in toc_entries:
                    toc_entries[entry.section_name] = entry

        # Compute fields_absent: only if no explorer found it present
        # AND it's not in fields_external (which means it exists elsewhere)
        all_absent: set[str] = set()
        for notes in exploration_notes:
            all_absent.update(notes.inventory.fields_absent)

        # Remove any that are actually present or external
        final_absent = all_absent - set(fields_by_name.keys()) - set(fields_external.keys())

        return DocumentInventory(
            fields_present=list(fields_by_name.values()),
            fields_absent=sorted(final_absent),
            fields_external=list(fields_external.values()),
            explicit_absences=list(explicit_absences.values()),
            toc_entries=list(toc_entries.values()),
        )

    def _populate_knowledge_graph(self, exploration_notes: list[ExplorationNotes]) -> None:
        """Transfer exploration discoveries to the knowledge graph.

        This makes findings available to downstream phases via
        the shared knowledge store.
        """
        knowledge = self.context.knowledge

        # First, aggregate inventories from all explorers
        master_inventory = self._aggregate_inventories(exploration_notes)
        knowledge.document_inventory = master_inventory

        for notes in exploration_notes:
            agent_name = f"explorer_{notes.page_start}-{notes.page_end}"

            # Record TOC pages
            if notes.toc_pages:
                knowledge.set_toc_pages(notes.toc_pages)
                knowledge.add_finding(Finding(
                    finding_type=FindingType.STRUCTURAL,
                    description=f"Table of contents on pages {notes.toc_pages}",
                    source_agent=agent_name,
                    pages=notes.toc_pages,
                ))

            # Record table discoveries
            for table in notes.tables:
                # Determine field name from table type
                table_type_lower = table.table_type.lower()
                field_name = None
                if "isin" in table_type_lower:
                    field_name = "isin"
                elif "fee" in table_type_lower:
                    field_name = "fee"
                elif "share" in table_type_lower:
                    field_name = "share_class"

                knowledge.add_finding(Finding(
                    finding_type=FindingType.TABLE_LOCATION,
                    description=f"{table.table_type} table",
                    source_agent=agent_name,
                    pages=list(range(table.page_start, table.page_end + 1)),
                    field_name=field_name,
                    metadata={
                        "columns": table.columns,
                        "has_fund_name_column": table.has_fund_name_column,
                    }
                ))

            # Record fund mentions with dedicated sections
            for fund in notes.funds_mentioned:
                if fund.has_dedicated_section:
                    knowledge.add_finding(Finding(
                        finding_type=FindingType.SECTION_LOCATION,
                        description=f"Dedicated section for {fund.name}",
                        source_agent=agent_name,
                        pages=[fund.page],
                        entity_name=fund.name,
                    ))

            # Record cross-references using enhanced fields
            for xref in notes.cross_references or []:
                xref_text = xref.text if xref else ""
                if not xref_text:
                    continue

                # Use is_external flag from CrossReference (set by LLM)
                # Fall back to keyword detection for backward compatibility
                is_external = getattr(xref, 'is_external', False)
                if not is_external:
                    # Backward compatibility: check keywords
                    external_keywords = ["kiid", "key investor", "annual report", "supplement", "www.", "http"]
                    is_external = any(kw in xref_text.lower() for kw in external_keywords)

                if is_external:
                    # Use field_hint from CrossReference if available
                    field_hint = getattr(xref, 'field_hint', None)
                    external_doc = getattr(xref, 'external_doc', None)

                    # Fall back to keyword detection if field_hint not set
                    if not field_hint:
                        if "isin" in xref_text.lower():
                            field_hint = "isin"
                        elif "fee" in xref_text.lower() or "charge" in xref_text.lower():
                            field_hint = "fee"
                        elif "performance" in xref_text.lower():
                            field_hint = "performance"
                        elif "risk" in xref_text.lower():
                            field_hint = "risk_profile"

                    # Fall back to keyword detection for external_doc
                    if not external_doc:
                        if "kiid" in xref_text.lower() or "key investor" in xref_text.lower():
                            external_doc = "KIID"
                        elif "annual report" in xref_text.lower():
                            external_doc = "Annual Report"
                        elif "supplement" in xref_text.lower():
                            external_doc = "Supplement"
                        elif "www." in xref_text.lower() or "http" in xref_text.lower():
                            external_doc = "Website"
                        else:
                            external_doc = "External Document"

                    # Record external reference (now records even without field_hint)
                    knowledge.record_external_reference(
                        field_name=field_hint or "unknown",
                        external_doc=external_doc,
                        source_page=xref.source_page,
                        source_quote=xref_text[:200],
                        source_agent=agent_name,
                    )
                else:
                    # Internal cross-reference
                    target_page = getattr(xref, 'target_page', None)
                    knowledge.add_finding(Finding(
                        finding_type=FindingType.CROSS_REFERENCE,
                        description=xref_text[:100],
                        source_agent=agent_name,
                        pages=[target_page] if target_page else [xref.source_page],
                        metadata={"full_text": xref_text, "target_page": target_page}
                    ))

        # Record external references from inventory
        for ext_ref in master_inventory.fields_external:
            # Avoid duplicates (may already be recorded from cross_references)
            existing = knowledge.get_external_ref_for_field(ext_ref.field_name)
            if not existing:
                knowledge.record_external_reference(
                    field_name=ext_ref.field_name,
                    external_doc=ext_ref.external_doc,
                    source_page=ext_ref.source_page,
                    source_quote=ext_ref.source_quote,
                    source_agent="inventory_aggregator",
                )

        # Populate KG-RAG graph
        self._populate_kg_rag_graph(exploration_notes)

        # Log summary
        stats = knowledge.stats()
        graph_stats = self.context.store.stats()
        inv_stats = f", inventory: {len(master_inventory.fields_present)} present, {len(master_inventory.fields_external)} external"
        self.log(f"Knowledge graph populated: {stats['total_findings']} findings, "
                 f"{stats['external_refs']} external refs{inv_stats}")
        self.log(f"KG-RAG graph: {graph_stats['entities']} entities, {graph_stats['relations']} relations")

    def _populate_kg_rag_graph(self, exploration_notes: list[ExplorationNotes]) -> None:
        """Populate the KG-RAG knowledge graph with structured entities and relations."""
        store = self.context.store

        # First pass: record all funds
        for notes in exploration_notes:
            source_phase = f"exploration_{notes.page_start}-{notes.page_end}"

            for fund in notes.funds_mentioned:
                # Get section pages from page_index
                section_pages = [
                    entry.page for entry in notes.page_index
                    if entry.fund_name and entry.fund_name.lower() == fund.name.lower()
                ]

                store.record_fund(
                    name=fund.name,
                    section_pages=section_pages or [fund.page],
                    has_dedicated_section=fund.has_dedicated_section,
                    source_phase=source_phase,
                    source_page=fund.page,
                )

        # Second pass: record tables and link to funds
        for notes in exploration_notes:
            source_phase = f"exploration_{notes.page_start}-{notes.page_end}"

            for table in notes.tables:
                pages = list(range(table.page_start, table.page_end + 1))

                table_entity = store.record_table(
                    table_type=table.table_type,
                    pages=pages,
                    columns=table.columns,
                    lookup_column=self._infer_lookup_column(table.columns) if table.has_fund_name_column else None,
                    is_consolidated=table.has_fund_name_column,
                    source_phase=source_phase,
                    belongs_to_fund=table.belongs_to_fund,
                )

                # If consolidated table discovered after funds, link them
                if table.has_fund_name_column:
                    store.link_funds_to_table(table_entity.id, table.table_type, source_phase)

        # Third pass: record cross-references as relations
        for notes in exploration_notes:
            source_phase = f"exploration_{notes.page_start}-{notes.page_end}"

            for xref in notes.cross_references or []:
                if not xref.text:
                    continue

                # Find which fund this cross-ref belongs to
                fund_name = None
                for entry in notes.page_index:
                    if entry.page == xref.source_page and entry.fund_name:
                        fund_name = entry.fund_name
                        break

                if fund_name and not xref.is_external:
                    from extractor.pydantic_models.graph_models import RelationType
                    target_pages = [xref.target_page] if xref.target_page else []
                    store.add_relation(
                        relation_type=RelationType.REFERENCES,
                        subject_key=f"fund:{fund_name}",
                        object_key=f"section:{xref.target_description or 'unknown'}",
                        properties={
                            "text": xref.text,
                            "target_pages": target_pages,
                            "field_hint": xref.field_hint,
                            "is_external": False,
                        },
                        source_phase=source_phase,
                        source_page=xref.source_page,
                    )

    def _infer_lookup_column(self, columns: list[str]) -> str | None:
        """Infer the best lookup column from column headers."""
        if not columns:
            return None
        candidates = ["Fund Name", "Sub-Fund", "Fund", "Name", "Share Class", "Class"]
        columns_lower = {c.lower(): c for c in columns}
        for candidate in candidates:
            if candidate.lower() in columns_lower:
                return columns_lower[candidate.lower()]
        return columns[0]

    def _resolve_cross_references(
        self,
        notes: list[ExplorationNotes],
        skeleton: DocumentSkeleton,
    ) -> list[ExplorationNotes]:
        """Resolve internal cross-references to actual page numbers.

        Uses the skeleton to resolve references like "See Appendix E"
        to actual page numbers.

        Args:
            notes: Exploration notes from all explorers.
            skeleton: Document skeleton with section mappings.

        Returns:
            Updated exploration notes with resolved cross-references.
        """
        resolved_count = 0
        total_internal = 0

        for note in notes:
            for xref in note.cross_references or []:
                # Skip external references
                if getattr(xref, 'is_external', False):
                    continue

                total_internal += 1

                # Skip if already has target_page
                if xref.target_page is not None:
                    continue

                # Try to resolve using skeleton
                resolved_page = skeleton.resolve_reference(xref.text)
                if resolved_page:
                    xref.target_page = resolved_page
                    resolved_count += 1

        if total_internal > 0:
            self.log(f"Cross-references: resolved {resolved_count}/{total_internal} internal refs")

        return notes
