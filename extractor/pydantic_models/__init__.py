"""Pydantic models for the extraction pipeline.

Modules:
- provenance: ExtractedValue wrapper for auditable extraction
- exploration_models: ExplorationNotes, FundMention, DocumentSkeleton, etc.
- planning_models: PlannerOutput, FundExtractionTask, BroadcastTable
- critic_models: CriticResult, FieldVerification
- extraction_models: RawSubFundData, RawShareClassData, RawUmbrellaData
- output: Final extraction result schemas (Umbrella, SubFund, ShareClass, ProspectusGraph)
- constraints: Typed constraint models for binding compliance rules

For backwards compatibility, pipeline.py re-exports from the focused modules.
"""

# Provenance tracking
from extractor.pydantic_models.provenance import (
    ExtractedValue,
    MaybeExtracted,
    NOT_FOUND,
    NotFoundReason,
)

# Pipeline internal models (re-exported from focused modules)
from extractor.pydantic_models.pipeline import (
    # Type aliases
    RawFieldValue,
    LegacyOrProvenanceValue,  # Backwards compatibility alias
    # Exploration models
    FundMention,
    TableDiscovery,
    TableInfo,  # Alias for TableDiscovery
    CrossReference,
    PageContent,
    PageIndexEntry,
    ExplorationNotes,
    # Document inventory models
    FieldPresence,
    ExternalFieldRef,
    TOCEntry,
    ExplicitAbsence,
    DocumentInventory,
    # Document skeleton models
    SectionInfo,
    DocumentSkeleton,
    # Planning models
    PageLookup,
    FundExtractionTask,
    BroadcastTable,
    PlannerOutput,
    # Critic models
    FieldVerification,
    CriticResult,
    # Intermediate extraction models
    RawShareClassData,
    RawSubFundData,
    RawUmbrellaData,
    ExtractionPhaseResult,
)

# Logic models (document structure patterns)
from extractor.pydantic_models.logic_models import (
    ISINStrategy,
    FeeStrategy,
    CrossReferenceResolution,
    DocumentLogic,
)

# Recipe models (per-field extraction strategies)
from extractor.pydantic_models.recipe_models import (
    SourceType,
    TableLookupSource,
    TextExtractionSource,
    CrossReferenceSource,
    InheritedSource,
    FieldStrategy,
    FundExtractionRecipe,
    ExtractionRecipeSet,
)

# Constraint models
from extractor.pydantic_models.constraints import (
    BindingStatus,
    Constraint,
    ConstraintSet,
    FundConstraintType,
    ShareClassConstraintType,
    PREDEFINED_CONSTRAINT_TYPES,
    CONSTRAINT_FIELDS,
    CONSTRAINT_FIELD_DESCRIPTIONS,
    # Helper functions
    asset_limit,
    concentration_limit,
    leverage_limit,
    hedging_policy,
    investor_eligibility,
    minimum_investment_constraint,
    holding_period,
)

# Output models
from extractor.pydantic_models.output import (
    Extracted,
    ShareClass,
    SubFund,
    Umbrella,
    DocumentStructure,
    TableLocation,
    ExplorationSummary,
    ExtractionMetadata,
    ExternalDocReference,
    DiscoveredField,
    SchemaSuggestion,
    ProspectusGraph,
    # Audit trail models
    AgentCostBreakdown,
    CostSummary,
    UnresolvedQuestion,
    FundPageAssignment,
    ExtractionPlanSummary,
)

__all__ = [
    # Provenance
    "ExtractedValue",
    "MaybeExtracted",
    "NOT_FOUND",
    "NotFoundReason",
    "Extracted",
    # Logic models
    "ISINStrategy",
    "FeeStrategy",
    "CrossReferenceResolution",
    "DocumentLogic",
    # Recipe models
    "SourceType",
    "TableLookupSource",
    "TextExtractionSource",
    "CrossReferenceSource",
    "InheritedSource",
    "FieldStrategy",
    "FundExtractionRecipe",
    "ExtractionRecipeSet",
    # Type aliases
    "RawFieldValue",
    "LegacyOrProvenanceValue",
    # Exploration models
    "FundMention",
    "TableDiscovery",
    "TableInfo",
    "CrossReference",
    "PageContent",
    "PageIndexEntry",
    "ExplorationNotes",
    # Document inventory models
    "FieldPresence",
    "ExternalFieldRef",
    "TOCEntry",
    "ExplicitAbsence",
    "DocumentInventory",
    # Document skeleton models
    "SectionInfo",
    "DocumentSkeleton",
    # Planning models
    "PageLookup",
    "FundExtractionTask",
    "BroadcastTable",
    "PlannerOutput",
    # Critic models
    "FieldVerification",
    "CriticResult",
    # Intermediate extraction models
    "RawShareClassData",
    "RawSubFundData",
    "RawUmbrellaData",
    "ExtractionPhaseResult",
    # Constraint models
    "BindingStatus",
    "Constraint",
    "ConstraintSet",
    "FundConstraintType",
    "ShareClassConstraintType",
    "PREDEFINED_CONSTRAINT_TYPES",
    "CONSTRAINT_FIELDS",
    "CONSTRAINT_FIELD_DESCRIPTIONS",
    "asset_limit",
    "concentration_limit",
    "leverage_limit",
    "hedging_policy",
    "investor_eligibility",
    "minimum_investment_constraint",
    "holding_period",
    # Output models
    "ShareClass",
    "SubFund",
    "Umbrella",
    "DocumentStructure",
    "TableLocation",
    "ExplorationSummary",
    "ExtractionMetadata",
    "ExternalDocReference",
    "DiscoveredField",
    "SchemaSuggestion",
    "ProspectusGraph",
    # Audit trail models
    "AgentCostBreakdown",
    "CostSummary",
    "UnresolvedQuestion",
    "FundPageAssignment",
    "ExtractionPlanSummary",
]
