"""Pipeline orchestrator — runs every phase of the extraction pipeline in order.

Phases are separate classes so each one can be tested, retried, and reasoned
about in isolation.  Between phases, data flows through a shared PipelineState
object (see phase_base.py) which acts as a typed mailbox: one phase writes its
output, the next phase reads it.  This keeps memory usage bounded because each
phase can drop intermediate data after handing off results.

High-level flow:
  Skeleton → TableScan → ExternalRefScan → Exploration → EntityResolution
  → Logic → Planning → Extraction → FailureRecovery → Assembly
"""

import asyncio
import gc
from pathlib import Path

from extractor.core import PDFReader, PipelineErrors, get_logger, CostTracker
from extractor.core.config import DEFAULT_MODELS, SMART_MODEL, FAST_MODEL, ChunkingConfig
from extractor.phases import (
    PhaseContext,
    ExtractionResources,
    ExtractionConfig,
    PipelineState,
    SkeletonPhase,
    TableScanPhase,
    ExternalRefScanPhase,
    ExplorationPhase,
    EntityResolutionPhase,
    LogicPhase,
    PlanningPhase,
    ExtractionPhase,
    FailureRecoveryPhase,
    AssemblyPhase,
)
from extractor.pydantic_models import ProspectusGraph


class Orchestrator:
    """Pipeline orchestrator coordinating phase runners."""

    def __init__(
        self,
        pdf_path: str | Path,
        smart_model: str | None = None,
        exploration_model: str | None = None,
        planner_model: str | None = None,
        reader_model: str = DEFAULT_MODELS["reader"],
        critic_model: str = DEFAULT_MODELS["critic"],
        max_concurrent: int = 5,
        chunk_size: int = ChunkingConfig.CHUNK_SIZE,
        use_critic: bool = True,
        verbose: bool = False,
        log_dir: str | Path | None = None,
        max_funds: int | None = None,
        gleaning_passes: int = 1,
        discover_bonus: bool = False,
    ):
        """Initialize the orchestrator.

        Args:
            pdf_path: Path to the PDF file.
            smart_model: Model for high-impact phases (exploration, planning).
                         Defaults to SMART_MODEL. If None, uses fast model for all.
            exploration_model: Model for explorers. Defaults to smart_model.
            planner_model: Model for planner. Defaults to smart_model.
            reader_model: Model for the reader agent (data extraction).
            critic_model: Model for critics.
            max_concurrent: Max concurrent API calls.
            chunk_size: Pages per exploration chunk.
            use_critic: Whether to verify with critics.
            verbose: If True, print detailed logs.
            log_dir: Directory for log files.
            max_funds: Limit number of funds to extract (for testing).
            gleaning_passes: Number of extraction passes (1 = no gleaning, 2+ = gleaning).
            discover_bonus: Whether to discover PMS-relevant fields beyond schema.
        """
        self.pdf = PDFReader(pdf_path)
        self.logger = get_logger(verbose=verbose, log_dir=log_dir)

        # Resolve smart model (defaults to SMART_MODEL constant)
        resolved_smart = smart_model or SMART_MODEL
        # Exploration and planner default to smart model if not explicitly set
        resolved_exploration = exploration_model or resolved_smart
        resolved_planner = planner_model or resolved_smart

        # Create the three component contexts
        resources = ExtractionResources(
            pdf=self.pdf,
            semaphore=asyncio.Semaphore(max_concurrent),
            logger=self.logger,
            cost_tracker=CostTracker(),
        )

        config = ExtractionConfig(
            smart_model=resolved_smart,
            exploration_model=resolved_exploration,
            planner_model=resolved_planner,
            reader_model=reader_model,
            critic_model=critic_model,
            chunk_size=chunk_size,
            use_critic=use_critic,
            verbose=verbose,
            max_funds=max_funds,
            gleaning_passes=gleaning_passes,
            discover_bonus=discover_bonus,
        )

        state = PipelineState()

        # Compose into PhaseContext
        self.context = PhaseContext(
            resources=resources,
            config=config,
            state=state,
        )

        # Track results
        self._skeleton_result = None
        self._table_scan_result = None
        self._external_ref_result = None
        self._exploration_result = None
        self._entity_resolution_result = None
        self._logic_result = None
        self._planning_result = None
        self._extraction_result = None
        self._resolver_result = None
        self._assembly_result = None

    async def run(self) -> ProspectusGraph:
        """Run the complete extraction pipeline.

        Phase DAG (data dependencies shown as arrows)::

            Skeleton ─────────────┬──────────────────────────────────┐
            TableScan ────────────┤                                  │
            ExternalRefScan ──────┤                                  │
                                  ▼                                  │
            Exploration ──► EntityResolution ──► Logic ──► Planning  │
                                                              │      │
                                                              ▼      ▼
                                                          Extraction
                                                              │
                                                              ▼
                                                      FailureRecovery
                                                              │
                                                              ▼
                                                          Assembly

        Returns:
            Extracted ProspectusGraph.
        """
        self.logger.start_pipeline(self.pdf.filename)

        try:
            # Phase 0: Skeleton (document structure detection)
            skeleton = SkeletonPhase(self.context)
            self._skeleton_result = await skeleton.run()

            # Phase 0.5: Table Scan (upfront table detection)
            table_scan = TableScanPhase(self.context)
            self._table_scan_result = await table_scan.run()

            # Phase 0.6: External Reference Pre-Scan (detect "See KIID" patterns)
            external_ref_scan = ExternalRefScanPhase(self.context)
            self._external_ref_result = await external_ref_scan.run()

            # Phase 1: Exploration
            exploration = ExplorationPhase(self.context)
            self._exploration_result = await exploration.run()

            # Phase 1.5: Entity Resolution (deduplicate funds EARLY)
            entity_resolution = EntityResolutionPhase(self.context)
            self._entity_resolution_result = await entity_resolution.run()

            # Phase 2: Document Logic (aggregate patterns - no LLM call)
            logic = LogicPhase(self.context)
            self._logic_result = await logic.run()

            # Phase 3: Planning
            planning = PlanningPhase(self.context)
            self._planning_result = await planning.run()

            # Phase 4-5: Extraction (uses recipes if DocumentLogic available)
            extraction = ExtractionPhase(self.context)
            self._extraction_result = await extraction.run()

            # Phase 6: Resolver (resolve EXTRACTION_FAILED fields)
            resolver = FailureRecoveryPhase(self.context)
            self._resolver_result = await resolver.run()

            # Phase 7: Assembly
            assembly = AssemblyPhase(self.context)
            self._assembly_result = await assembly.run()

            self.logger.end_pipeline(success=True, stats=self.get_stats())
            return self._assembly_result.graph

        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}")
            self.logger.end_pipeline(success=False, stats=self.get_stats())
            raise

        finally:
            await self._cleanup()

    async def _cleanup(self):
        """Clean up resources."""
        self.pdf.close()
        await asyncio.sleep(0.1)
        gc.collect()

    def get_stats(self) -> dict:
        """Get extraction statistics."""
        resolver_stats = {}
        if self._resolver_result:
            resolver_stats = {
                "fields_needing_resolution": self._resolver_result.total_actionable,
                "resolved": self._resolver_result.resolved_count,
                "upgraded_to_typed": self._resolver_result.upgraded_to_typed,
                "still_failed": self._resolver_result.still_failed,
            }

        # Calculate average confidence from assembly stats or critic results
        avg_confidence = 0.0
        if self.context.critic_results:
            avg_confidence = sum(r.overall_confidence for r in self.context.critic_results) / len(self.context.critic_results)
        elif self._assembly_result and self._assembly_result.provenance_stats:
            stats = self._assembly_result.provenance_stats
            found_values = stats.get("total", 0) - stats.get("not_found", 0)
            if found_values > 0:
                high_conf = stats.get("high_confidence", 0)
                avg_confidence = high_conf / found_values

        # Table scan stats
        table_scan_stats = {}
        if self._table_scan_result:
            table_scan_stats = {
                "tables_found": len(self._table_scan_result.tables),
                "pages_with_tables": len(self._table_scan_result.pages_with_tables),
            }

        # External ref scan stats
        external_ref_stats = {}
        if self._external_ref_result:
            external_ref_stats = {
                "refs_found": self._external_ref_result.total_refs_found,
                "fields_external": self._external_ref_result.fields_external,
            }

        # Entity resolution stats
        entity_resolution_stats = {}
        if self._entity_resolution_result:
            entity_resolution_stats = {
                "mentions_found": self._entity_resolution_result.total_mentions,
                "canonical_funds": len(self._entity_resolution_result.canonical_funds),
                "duplicates_merged": self._entity_resolution_result.duplicates_merged,
                "low_confidence_filtered": self._entity_resolution_result.low_confidence_filtered,
            }

        return {
            "explorers": len(self.context.exploration_notes) if self.context.exploration_notes else 0,
            "funds_planned": self.context.plan.total_funds if self.context.plan else 0,
            "funds_extracted": len(self.context.funds_data) if self.context.funds_data else 0,
            "broadcast_tables": self.context.parsed_broadcast_tables,
            "critic_verifications": len(self.context.critic_results) if self.context.critic_results else 0,
            "average_confidence": round(avg_confidence, 4),
            "table_scan": table_scan_stats,
            "external_refs": external_ref_stats,
            "entity_resolution": entity_resolution_stats,
            "resolver": resolver_stats,
            "errors": self.context.errors.summary(),
        }

    def get_errors(self) -> PipelineErrors:
        """Get pipeline errors."""
        return self.context.errors
