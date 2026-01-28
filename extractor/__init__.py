"""Fund Prospectus Extraction Pipeline.

A parallel, context-fresh approach to extracting hierarchical entity graphs
from UCITS fund prospectuses.

Architecture:
    core/       - PDF reader, search utilities, logging, errors
    prompts/    - LLM prompt templates
    agents/     - Agent implementations (explorer, planner, extractor, critic)
    models/     - Pydantic models for internal and output data
    phases/     - Phase runner classes for modular pipeline

Usage:
    from extractor import Orchestrator

    orchestrator = Orchestrator("path/to/prospectus.pdf")
    graph = await orchestrator.run()

CLI:
    uv run extract prospectuses/jpm_umbrella.pdf
"""

from extractor.orchestrator import Orchestrator
from extractor.pydantic_models import (
    # Provenance
    ExtractedValue,
    NOT_FOUND,
    NotFoundReason,
    # Pipeline models
    ExplorationNotes,
    FundMention,
    TableDiscovery,
    PlannerOutput,
    FundExtractionTask,
    BroadcastTable,
    CriticResult,
    FieldVerification,
    # Output models
    Umbrella,
    SubFund,
    ShareClass,
    ProspectusGraph,
    ExtractionMetadata,
)

__all__ = [
    # Main entry point
    "Orchestrator",
    # Provenance
    "ExtractedValue",
    "NOT_FOUND",
    "NotFoundReason",
    # Pipeline models
    "ExplorationNotes",
    "FundMention",
    "TableDiscovery",
    "PlannerOutput",
    "FundExtractionTask",
    "BroadcastTable",
    "CriticResult",
    "FieldVerification",
    # Output models
    "Umbrella",
    "SubFund",
    "ShareClass",
    "ProspectusGraph",
    "ExtractionMetadata",
]
