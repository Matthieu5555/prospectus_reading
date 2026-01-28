"""Phase runners for the extraction pipeline.

Each phase is encapsulated in its own runner class with:
- Clear inputs and outputs
- Error handling
- Logging
- Progress tracking
"""

from extractor.phases.phase_base import (
    PhaseRunner,
    PhaseContext,
    ExtractionResources,
    ExtractionConfig,
    PipelineState,
)
from extractor.phases.skeleton_phase import SkeletonPhase, structure_aware_chunks
from extractor.phases.table_scan_phase import (
    TableScanPhase,
    TableScanResult,
    ScannedTable,
)
from extractor.phases.external_ref_scan_phase import (
    ExternalRefScanPhase,
    ExternalRefResult,
)
from extractor.phases.exploration_phase import ExplorationPhase
from extractor.phases.entity_resolution_phase import (
    EntityResolutionPhase,
    EntityResolutionResult,
)
from extractor.phases.logic_phase import LogicPhase
from extractor.phases.planning_phase import PlanningPhase
from extractor.phases.extraction_phase import ExtractionPhase
from extractor.phases.resolver_phase import FailureRecoveryPhase
from extractor.phases.assembly_phase import AssemblyPhase

__all__ = [
    "PhaseRunner",
    "PhaseContext",
    "ExtractionResources",
    "ExtractionConfig",
    "PipelineState",
    "SkeletonPhase",
    "structure_aware_chunks",
    "TableScanPhase",
    "TableScanResult",
    "ScannedTable",
    "ExternalRefScanPhase",
    "ExternalRefResult",
    "ExplorationPhase",
    "EntityResolutionPhase",
    "EntityResolutionResult",
    "LogicPhase",
    "PlanningPhase",
    "ExtractionPhase",
    "FailureRecoveryPhase",
    "AssemblyPhase",
]
