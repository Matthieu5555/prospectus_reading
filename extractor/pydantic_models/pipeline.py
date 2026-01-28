"""Pydantic schemas for pipeline internal communication.

This module re-exports models from focused sub-modules for backwards compatibility.
New code should import directly from the specific modules:
- exploration_models: ExplorationNotes, FundMention, TableDiscovery, etc.
- planning_models: PlannerOutput, FundExtractionTask, BroadcastTable, etc.
- critic_models: CriticResult, FieldVerification
- extraction_models: RawSubFundData, RawShareClassData, RawUmbrellaData
"""

# Re-export from exploration_models
from extractor.pydantic_models.exploration_models import (
    FundMention,
    TableDiscovery,
    CrossReference,
    FieldPresence,
    ExternalFieldRef,
    TOCEntry,
    ExplicitAbsence,
    DocumentInventory,
    SectionInfo,
    DocumentSkeleton,
    PageContent,
    PageIndexEntry,
    ExplorationNotes,
)

# Re-export from planning_models
from extractor.pydantic_models.planning_models import (
    PageLookup,
    FundExtractionTask,
    BroadcastTable,
    PlannerOutput,
)

# Re-export from critic_models
from extractor.pydantic_models.critic_models import (
    FieldVerification,
    CriticResult,
)

# Re-export from extraction_models
from extractor.pydantic_models.extraction_models import (
    RawFieldValue,
    RawShareClassData,
    RawSubFundData,
    RawUmbrellaData,
    ExtractionPhaseResult,
)

# Backwards compatibility alias
LegacyOrProvenanceValue = RawFieldValue

# Convenience alias for TableDiscovery (used as TableInfo in some places)
TableInfo = TableDiscovery

__all__ = [
    # Exploration models
    "FundMention",
    "TableDiscovery",
    "TableInfo",  # Alias for TableDiscovery
    "CrossReference",
    "FieldPresence",
    "ExternalFieldRef",
    "TOCEntry",
    "ExplicitAbsence",
    "DocumentInventory",
    "SectionInfo",
    "DocumentSkeleton",
    "PageContent",
    "PageIndexEntry",
    "ExplorationNotes",
    # Planning models
    "PageLookup",
    "FundExtractionTask",
    "BroadcastTable",
    "PlannerOutput",
    # Critic models
    "FieldVerification",
    "CriticResult",
    # Extraction models
    "RawFieldValue",
    "LegacyOrProvenanceValue",  # Backwards compatibility
    "RawShareClassData",
    "RawSubFundData",
    "RawUmbrellaData",
    "ExtractionPhaseResult",
]
