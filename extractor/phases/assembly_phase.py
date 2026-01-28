"""Assembly phase - final graph construction with provenance.

This is the orchestration layer for the assembly phase. The actual
logic is split into focused sub-modules:
- assembly/converters.py: Transform LLM output to typed models
- assembly/statistics.py: Compute provenance statistics
- assembly/graph_builder.py: Build the ProspectusGraph
- assembly/gap_filling.py: Fill gaps with umbrella constraints

Also performs final reconciliation to surface any pipeline issues.
"""

from dataclasses import dataclass

from extractor.phases.phase_base import PhaseRunner
from extractor.pydantic_models import ProspectusGraph
from extractor.phases.assembly import (
    fill_gaps,
    build_graph,
    count_provenance,
)
from extractor.core.value_helpers import is_not_found


@dataclass
class AssemblyResult:
    """Result from the assembly phase."""

    graph: ProspectusGraph
    provenance_stats: dict


class AssemblyPhase(PhaseRunner[AssemblyResult]):
    """Phase 6: Final assembly with gap filling and provenance.

    Responsibilities:
    - Fill gaps with umbrella-level constraints
    - Build final typed models with provenance
    - Calculate provenance statistics

    The heavy lifting is delegated to sub-modules for maintainability.
    """

    name = "Assembly"

    async def run(self) -> AssemblyResult:
        """Execute assembly phase.

        Returns:
            AssemblyResult with final graph and provenance statistics.
        """
        self.start(3)  # Gap fill, build, stats

        # Step 1: Gap filling
        await fill_gaps(self.context, self.log)
        self.progress(1, 3, "gaps filled")

        # Step 2: Build graph
        graph = build_graph(self.context)
        self.progress(2, 3, "graph built")

        # Step 3: Calculate stats
        stats = count_provenance(graph)
        self.progress(3, 3, "stats computed")

        # Update metadata with stats
        graph.metadata.total_extractions = stats["total"]
        graph.metadata.high_confidence_count = stats["high_confidence"]
        graph.metadata.low_confidence_count = stats["low_confidence"]
        graph.metadata.not_found_count = stats["not_found"]

        # Log phase result with metrics
        total_shares = sum(len(f.share_classes) for f in graph.sub_funds)
        self.logger.phase_result(
            "Assembly",
            f"{len(graph.sub_funds)} sub-funds, {total_shares} share classes",
            values_tracked=stats['total'],
            high_confidence=stats['high_confidence'],
            not_found=stats['not_found'],
            actionable=stats['actionable_not_found'],
        )

        # Final reconciliation check
        reconciliation_issues = self._reconcile(graph)
        if reconciliation_issues:
            self.log(f"Reconciliation found {len(reconciliation_issues)} issue(s)", "warning")
            for issue in reconciliation_issues[:5]:
                self.log(f"  - {issue}", "warning")
            if len(reconciliation_issues) > 5:
                self.log(f"  ... and {len(reconciliation_issues) - 5} more", "warning")

        return AssemblyResult(graph=graph, provenance_stats=stats)

    def _reconcile(self, graph: ProspectusGraph) -> list[str]:
        """Perform final reconciliation checks.

        Validates:
        1. Fund count matches between planning and extraction
        2. Every fund has at least 1 share class
        3. Share classes without ISINs (unless external)

        Returns list of issues found.
        """
        issues = []

        # Check 1: Fund count mismatch
        plan = self.context.plan
        if plan:
            planned_count = plan.total_funds
            extracted_count = len(graph.sub_funds)
            if planned_count > 0 and abs(planned_count - extracted_count) > 2:
                issues.append(
                    f"FUND_COUNT_MISMATCH: Planned {planned_count} funds, extracted {extracted_count} "
                    f"(diff: {extracted_count - planned_count})"
                )

        # Check 2: Funds without share classes
        funds_without_share_classes = []
        for fund in graph.sub_funds:
            if not fund.share_classes:
                funds_without_share_classes.append(fund.name)

        if funds_without_share_classes:
            issues.append(
                f"FUNDS_WITHOUT_SHARE_CLASSES: {len(funds_without_share_classes)} funds have 0 share classes: "
                f"{funds_without_share_classes[:3]}{'...' if len(funds_without_share_classes) > 3 else ''}"
            )

        # Check 3: Share classes without ISINs (unless external)
        # Get external fields from knowledge
        external_fields = set()
        for ref in self.context.knowledge.external_refs:
            external_fields.add(ref.field_name)

        isin_is_external = "isin" in external_fields

        if not isin_is_external:
            share_classes_without_isin = []
            total_share_classes = 0

            for fund in graph.sub_funds:
                for sc in fund.share_classes:
                    total_share_classes += 1
                    # Check if ISIN is NOT_FOUND without external reference
                    isin_val = getattr(sc, 'isin', None)
                    if isin_val:
                        raw_val = isin_val.value if hasattr(isin_val, 'value') else isin_val
                        not_found_reason = getattr(isin_val, 'not_found_reason', None) if hasattr(isin_val, 'not_found_reason') else None
                        if raw_val == "NOT_FOUND" and not_found_reason != "in_external_doc":
                            share_classes_without_isin.append(f"{fund.name}/{sc.name}")

            if share_classes_without_isin and total_share_classes > 0:
                pct = len(share_classes_without_isin) / total_share_classes * 100
                if pct > 30:  # More than 30% missing ISINs is concerning
                    issues.append(
                        f"MISSING_ISINS: {len(share_classes_without_isin)}/{total_share_classes} share classes "
                        f"({pct:.0f}%) have no ISIN: {share_classes_without_isin[:3]}..."
                    )

        # Check 4: Entity resolution stats
        entity_res_result = getattr(self.context.state, '_entity_resolution_result', None)
        if entity_res_result:
            if entity_res_result.reconciliation_issues:
                issues.extend(entity_res_result.reconciliation_issues)

        return issues
