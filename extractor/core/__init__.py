"""Core utilities for the extraction pipeline."""

from extractor.core.pdf_reader import PDFReader
from extractor.core.config import (
    LLM_PROVIDER,
    API_KEY_ENV_VAR,
    API_KEY_ENV_VARS,
    DEFAULT_MODELS,
    SMART_MODEL,
    FAST_MODEL,
    ConfidenceThresholds,
    SearchLimits,
    PageLimits,
    ChunkingConfig,
    ExtractionConfig as ExtractionConfigConstants,
    UmbrellaConfig,
    SearchPatterns,
    LLMConfig,
    RetryConfig,
    RegexPatterns,
)
from extractor.core.umbrella_page_selector import (
    UmbrellaPagePlan,
    select_umbrella_pages,
    validate_umbrella_pages,
    estimate_tokens,
    check_token_limit,
)
from extractor.core.llm_client import (
    LLMClient,
    LLMResponse,
    llm_complete,
)
from extractor.core.pipeline_logger import PipelineLogger, get_logger, reset_logger
from extractor.core.errors import (
    ErrorSeverity,
    ErrorCategory,
    ExtractionError,
    PipelineErrors,
    llm_api_error,
    llm_parse_error,
    validation_error,
    timeout_error,
    resource_error,
)
from extractor.core.document_knowledge import (
    DocumentKnowledge,
    Finding,
    FindingType,
    Question,
    QuestionPriority,
    ExternalReference,
)
from extractor.core.field_strategy import KnowledgeContext, FieldStrategy
from extractor.core.graph_store import GraphStore
from extractor.core.knowledge_consolidator import (
    KnowledgeConsolidator,
    ConflictResolution,
    EntityMerge,
    ReconciliationReport,
    SourceType,
    ResolutionStrategy,
)
from extractor.core.field_searchers import (
    BaseResolver,
    ResolverResult,
    get_resolver,
    FIELD_RESOLVERS,
)
from extractor.core.constraint_parser import (
    parse_fund_constraints,
    parse_share_class_constraints,
)
from extractor.core.fund_names import (
    FundNameMapping,
    deduplicate_fund_names,
    deduplicate_from_exploration_notes,
)
from extractor.core.cost_tracker import (
    CostTracker,
    CallUsage,
)
from extractor.core.value_helpers import (
    is_not_found,
    is_actionable_not_found,
    get_raw_value,
)
from extractor.core.fund_names import (
    normalize_fund_name,
    fund_names_match,
    strip_umbrella_prefix,
)
from extractor.pydantic_models.constraints import (
    CONSTRAINT_FIELDS,
    CONSTRAINT_FIELD_DESCRIPTIONS,
)
from extractor.core.smart_chunker import (
    create_smart_chunks,
    create_smart_chunks_detailed,
    skeleton_from_native_toc,
    ChunkInfo,
)
from extractor.core.table_extraction import (
    TableExtractor,
    ParsedTable,
)

__all__ = [
    # Configuration
    "LLM_PROVIDER",
    "API_KEY_ENV_VAR",
    "API_KEY_ENV_VARS",
    "DEFAULT_MODELS",
    "SMART_MODEL",
    "FAST_MODEL",
    "ConfidenceThresholds",
    "SearchLimits",
    "PageLimits",
    "ChunkingConfig",
    "ExtractionConfigConstants",
    "UmbrellaConfig",
    "SearchPatterns",
    "LLMConfig",
    "RetryConfig",
    "RegexPatterns",
    # Umbrella page selection
    "UmbrellaPagePlan",
    "select_umbrella_pages",
    "validate_umbrella_pages",
    "estimate_tokens",
    "check_token_limit",
    # LLM Client
    "LLMClient",
    "LLMResponse",
    "llm_complete",
    # Core classes
    "PDFReader",
    "PipelineLogger",
    "get_logger",
    "reset_logger",
    "ErrorSeverity",
    "ErrorCategory",
    "ExtractionError",
    "PipelineErrors",
    "llm_api_error",
    "llm_parse_error",
    "validation_error",
    "timeout_error",
    "resource_error",
    # Knowledge graph
    "DocumentKnowledge",
    "Finding",
    "FindingType",
    "Question",
    "QuestionPriority",
    "ExternalReference",
    "KnowledgeContext",
    "FieldStrategy",
    # Graph store
    "GraphStore",
    # Knowledge consolidation
    "KnowledgeConsolidator",
    "ConflictResolution",
    "EntityMerge",
    "ReconciliationReport",
    "SourceType",
    "ResolutionStrategy",
    # Field searchers (find missing values)
    "BaseResolver",
    "ResolverResult",
    "get_resolver",
    "FIELD_RESOLVERS",
    # Constraint parser
    "parse_fund_constraints",
    "parse_share_class_constraints",
    # Fund name deduplication
    "FundNameMapping",
    "deduplicate_fund_names",
    "deduplicate_from_exploration_notes",
    # Cost tracking
    "CostTracker",
    "CallUsage",
    # Shared utilities
    "is_not_found",
    "get_raw_value",
    "normalize_fund_name",
    "fund_names_match",
    "strip_umbrella_prefix",
    # Field constants
    "CONSTRAINT_FIELDS",
    "CONSTRAINT_FIELD_DESCRIPTIONS",
    # Smart chunking
    "create_smart_chunks",
    "create_smart_chunks_detailed",
    "skeleton_from_native_toc",
    "ChunkInfo",
    # Table extraction
    "TableExtractor",
    "ParsedTable",
]
