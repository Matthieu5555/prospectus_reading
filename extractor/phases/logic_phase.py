"""Document Logic phase - aggregates exploration into explicit patterns.

SIMPLIFIED VERSION: Pure aggregation from exploration findings and knowledge graph.
No LLM call needed - exploration already discovered tables and external references.

This phase converts exploration discoveries into a DocumentLogic structure
for backward compatibility with downstream phases.
"""

from dataclasses import dataclass

from extractor.phases.phase_base import PhaseRunner
from extractor.pydantic_models import ExplorationNotes
from extractor.pydantic_models.logic_models import (
    DocumentLogic,
    ISINStrategy,
    FeeStrategy,
    CrossReferenceResolution,
)
from extractor.agents.planner_agent import compute_table_pages
from extractor.core import FindingType


@dataclass
class LogicResult:
    """Result from the document logic phase."""

    logic: DocumentLogic
    isin_in_table: bool
    fee_in_table: bool


class LogicPhase(PhaseRunner[LogicResult]):
    """Phase 2: Document Logic Aggregation.

    SIMPLIFIED: No LLM call. Aggregates exploration findings into DocumentLogic.

    This phase:
    1. Extracts table locations from exploration findings
    2. Determines ISIN/fee strategies based on discovered tables
    3. Infers template type from document structure
    4. Aggregates cross-reference resolutions from exploration
    """

    name = "Logic"

    async def run(self) -> LogicResult:
        """Aggregate exploration findings into DocumentLogic.

        No LLM call - pure aggregation from knowledge graph and exploration notes.

        Returns:
            LogicResult with document patterns.
        """
        self.start(1, model="aggregation (no LLM)")

        exploration_notes = self.context.exploration_notes
        if not exploration_notes:
            self.log("No exploration notes available, using default logic", "warning")
            return self._create_default_logic()

        self.log(f"Aggregating {len(exploration_notes)} exploration reports (no LLM call)")

        # Aggregate findings from exploration
        logic = self._aggregate_from_exploration(exploration_notes)

        # Log key findings
        self.log(f"ISIN strategy: {logic.isin_strategy.location}")
        self.log(f"Fee strategy: {logic.fee_strategy.location}")

        if logic.observations:
            for obs in logic.observations[:3]:
                self.log(f"Note: {obs}")

        # Store in context for downstream phases
        self.context.state.document_logic = logic

        self.logger.phase_result(
            "Logic",
            f"isin={logic.isin_strategy.location}, fee={logic.fee_strategy.location}",
            cross_refs=len(logic.resolved_cross_references),
        )

        return LogicResult(
            logic=logic,
            isin_in_table=logic.isin_strategy.location == "consolidated_table",
            fee_in_table=logic.fee_strategy.location == "consolidated_table",
        )

    def _aggregate_from_exploration(
        self,
        exploration_notes: list[ExplorationNotes],
    ) -> DocumentLogic:
        """Aggregate exploration findings into DocumentLogic.

        No LLM call - derives everything from exploration discoveries.
        """
        knowledge = self.context.knowledge

        # Get table pages from exploration
        table_pages = compute_table_pages(exploration_notes)
        isin_pages = table_pages.get("isin_table", [])
        fee_pages = table_pages.get("fee_table", [])

        # Determine ISIN strategy
        isin_strategy = self._determine_isin_strategy(isin_pages, knowledge)

        # Determine fee strategy
        fee_strategy = self._determine_fee_strategy(fee_pages, knowledge)

        # Aggregate cross-reference resolutions from exploration
        cross_refs = self._aggregate_cross_refs(exploration_notes)

        # Build appendix map from skeleton if available
        appendix_map = {}
        skeleton = getattr(self.context, 'skeleton', None)
        if skeleton and hasattr(skeleton, 'appendix_map'):
            for name, pages in skeleton.appendix_map.items():
                if isinstance(pages, (list, tuple)) and len(pages) >= 2:
                    appendix_map[name] = (pages[0], pages[-1])

        # Build observations
        observations = self._build_observations(
            isin_strategy, fee_strategy, isin_pages, fee_pages, knowledge
        )

        return DocumentLogic(
            isin_strategy=isin_strategy,
            fee_strategy=fee_strategy,
            resolved_cross_references=cross_refs,
            appendix_map=appendix_map,
            section_boundaries={},
            observations=observations,
        )

    def _determine_isin_strategy(self, isin_pages: list[int], knowledge) -> ISINStrategy:
        """Determine ISIN extraction strategy from knowledge graph."""
        # Check if ISINs are external
        ext_ref = knowledge.get_external_ref_for_field("isin")
        if ext_ref:
            return ISINStrategy(
                location="external",
                external_doc=ext_ref.external_doc,
                table_pages=None,
                lookup_column=None,
                value_column="ISIN",
            )

        # Check for consolidated table
        if isin_pages:
            # Try to get column info from findings
            lookup_col = self._find_lookup_column("isin", knowledge)
            return ISINStrategy(
                location="consolidated_table",
                table_pages=isin_pages,
                lookup_column=lookup_col,
                value_column="ISIN",
            )

        # Default to per-fund inline
        return ISINStrategy(
            location="per_fund_inline",
            table_pages=None,
            lookup_column=None,
            value_column="ISIN",
        )

    def _determine_fee_strategy(self, fee_pages: list[int], knowledge) -> FeeStrategy:
        """Determine fee extraction strategy from knowledge graph."""
        # Check if fees are external
        ext_ref = knowledge.get_external_ref_for_field("fee")
        if ext_ref:
            return FeeStrategy(
                location="external",
                external_doc=ext_ref.external_doc,
                table_pages=None,
                lookup_column=None,
                fee_columns={},
            )

        # Check for consolidated table
        if fee_pages:
            lookup_col = self._find_lookup_column("fee", knowledge)
            return FeeStrategy(
                location="consolidated_table",
                table_pages=fee_pages,
                lookup_column=lookup_col,
                fee_columns={},
            )

        # Default to per-fund inline
        return FeeStrategy(
            location="per_fund_inline",
            table_pages=None,
            lookup_column=None,
            fee_columns={},
        )

    def _find_lookup_column(self, field_name: str, knowledge) -> str | None:
        """Find lookup column from table findings."""
        table_findings = [
            f for f in knowledge.get_findings_by_type(FindingType.TABLE_LOCATION)
            if f.field_name == field_name
        ]
        for finding in table_findings:
            cols = finding.metadata.get("columns", [])
            for candidate in ["Fund Name", "Sub-Fund", "Share Class", "Class", "Name"]:
                if candidate in cols:
                    return candidate
        return None

    def _aggregate_cross_refs(
        self, exploration_notes: list[ExplorationNotes]
    ) -> list[CrossReferenceResolution]:
        """Aggregate cross-reference resolutions from exploration."""
        cross_refs = []

        for notes in exploration_notes:
            for xref in notes.cross_references or []:
                if xref.target_page:
                    cross_refs.append(CrossReferenceResolution(
                        reference_text=xref.text,
                        target_pages=[xref.target_page],
                        field_hint=xref.field_hint,
                        source_page=xref.source_page,
                    ))

        return cross_refs

    def _build_observations(
        self,
        isin_strategy: ISINStrategy,
        fee_strategy: FeeStrategy,
        isin_pages: list[int],
        fee_pages: list[int],
        knowledge,
    ) -> list[str]:
        """Build human-readable observations about the document logic."""
        obs = []

        # ISIN observations
        if isin_strategy.location == "external":
            ext_ref = knowledge.get_external_ref_for_field("isin")
            doc = ext_ref.external_doc if ext_ref else "external document"
            obs.append(f"ISINs are documented in {doc}, not in this prospectus")
        elif isin_strategy.location == "consolidated_table":
            obs.append(f"ISIN table found on pages {isin_pages}")
        else:
            obs.append("ISINs appear inline within fund sections")

        # Fee observations
        if fee_strategy.location == "external":
            ext_ref = knowledge.get_external_ref_for_field("fee")
            doc = ext_ref.external_doc if ext_ref else "external document"
            obs.append(f"Fees are documented in {doc}")
        elif fee_strategy.location == "consolidated_table":
            obs.append(f"Fee table found on pages {fee_pages}")
        else:
            obs.append("Fees appear inline within fund sections")

        # External refs summary
        external_count = len(knowledge.external_refs)
        if external_count > 0:
            external_fields = [ref.field_name for ref in knowledge.external_refs]
            obs.append(f"External references: {', '.join(set(external_fields))}")

        obs.append("Logic derived from exploration (no LLM call)")

        return obs

    def _create_default_logic(self) -> LogicResult:
        """Create default logic when no exploration notes available."""
        logic = DocumentLogic(
            isin_strategy=ISINStrategy(
                location="per_fund_inline",
                table_pages=None,
                lookup_column=None,
                value_column="ISIN",
            ),
            fee_strategy=FeeStrategy(
                location="per_fund_inline",
                table_pages=None,
                lookup_column=None,
                fee_columns={},
            ),
            observations=["Default logic used - no exploration notes available"],
        )

        self.context.state.document_logic = logic

        self.logger.phase_result(
            "Logic",
            "default (no exploration)",
        )

        return LogicResult(
            logic=logic,
            isin_in_table=False,
            fee_in_table=False,
        )
