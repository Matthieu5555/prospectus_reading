"""Base classes for pipeline phases.

The context is split into three parts so responsibilities are clear:
- **ExtractionResources** (frozen): external dependencies created once —
  PDF reader, concurrency semaphore, logger, cost tracker.
- **ExtractionConfig** (frozen): user-chosen settings that never change
  mid-run — model names, chunk size, feature flags.
- **PipelineState** (mutable): the data that accumulates as each phase runs —
  exploration notes, plans, extracted fund data, critic results, etc.

PhaseContext wraps all three and exposes convenience properties so phases can
write ``ctx.pdf`` instead of ``ctx.resources.pdf``.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, TypeVar, Generic, TYPE_CHECKING

from extractor.core import PDFReader, PipelineErrors, PipelineLogger, DocumentKnowledge, CostTracker
from extractor.core.config import DEFAULT_MODELS, SMART_MODEL, ChunkingConfig
from extractor.core.graph_store import GraphStore

if TYPE_CHECKING:
    from extractor.pydantic_models.exploration_models import DocumentSkeleton, ExplorationNotes
    from extractor.pydantic_models.logic_models import DocumentLogic
    from extractor.pydantic_models.planning_models import PlannerOutput
    from extractor.pydantic_models.critic_models import CriticResult
    from extractor.phases.table_scan_phase import ScannedTable


# Split Context Classes

@dataclass(frozen=True)
class ExtractionResources:
    """Shared resources - created once, never modified.

    These are the external dependencies that all phases need access to.
    """

    pdf: PDFReader
    semaphore: asyncio.Semaphore
    logger: PipelineLogger
    cost_tracker: CostTracker


@dataclass(frozen=True)
class ExtractionConfig:
    """Configuration - set at init, never modified.

    These control how extraction behaves but don't change during the pipeline.
    Default models come from extractor.core.config.DEFAULT_MODELS.

    Model tiers:
    - smart_model: Used for high-impact phases (exploration, planning) where
      reasoning quality significantly affects downstream extraction.
    - reader_model: Used for routine per-fund extraction (fast/cheap model).
    - critic_model: Used for verification passes.
    """

    smart_model: str = SMART_MODEL  # High-impact phases (exploration, planning)
    exploration_model: str = DEFAULT_MODELS["exploration"]  # Will be set to smart_model if not specified
    planner_model: str = DEFAULT_MODELS["planning"]  # Will be set to smart_model if not specified
    reader_model: str = DEFAULT_MODELS["reader"]  # Fast model for per-fund extraction
    critic_model: str = DEFAULT_MODELS["critic"]
    chunk_size: int = ChunkingConfig.CHUNK_SIZE
    use_critic: bool = True
    verbose: bool = False
    max_funds: int | None = None  # Limit number of funds to extract (for testing)
    gleaning_passes: int = 1  # Number of extraction passes (1 = no gleaning, 2+ = gleaning)
    discover_bonus: bool = False  # Whether to run schema discovery for bonus fields


@dataclass
class PipelineState:
    """Mutable state that accumulates during the pipeline.

    Each field is written by exactly one phase and read by downstream phases:
    - exploration_notes: Written by Exploration, read by Logic/Planning/Assembly
    - document_logic: Written by Logic, read by Planning/Extraction
    - plan: Written by Planning, read by Extraction/Assembly
    - broadcast_data: Written by Extraction, read by Assembly
    - umbrella_data: Written by Extraction, read by Assembly
    - funds_data: Written by Extraction, read by Resolver/Assembly
    - critic_results: Written by Extraction (if critic enabled), read by Assembly
    - discovered_fields/schema_suggestions: Written by Extraction, read by Assembly
    - errors: Accumulated by all phases
    - knowledge: Written by Exploration/Extraction, read by Resolver/Assembly

    Knowledge systems:
    - ``knowledge`` (DocumentKnowledge): Exploration findings — page-level notes,
      fund mentions, table descriptions, and external-reference annotations
      discovered during the exploration phase.
    - ``store`` (GraphStore): Entity graph — canonical fund entities, their
      relationships (share-class → fund, fund → umbrella), and alias mappings
      built during entity resolution.
    - ``errors`` (PipelineErrors): Structured error log — every warning / error
      raised by any phase, categorised by ErrorCategory and ErrorSeverity for
      downstream reporting and debugging.
    """

    # Phase outputs (each written by one phase)
    skeleton: DocumentSkeleton | None = None  # Written by Skeleton phase, read by Planning/Extraction
    exploration_notes: list[ExplorationNotes] = field(default_factory=list)
    document_logic: DocumentLogic | None = None  # Written by Logic phase, read by Planning/Extraction
    plan: PlannerOutput | None = None  # Written by Planning phase, read by Extraction/Assembly
    broadcast_data: dict[str, list[dict]] = field(default_factory=dict)
    parsed_broadcast_tables: int = 0  # Count of successfully parsed broadcast tables (recipe system)
    umbrella_data: dict = field(default_factory=dict)
    funds_data: list = field(default_factory=list)
    critic_results: list[CriticResult] = field(default_factory=list)
    discovered_fields: list = field(default_factory=list)  # Bonus fields beyond schema
    schema_suggestions: list = field(default_factory=list)  # Suggested schema additions

    # Table scan results (written by TableScanPhase, read by Exploration/Extraction)
    scanned_tables: list[ScannedTable] = field(default_factory=list)
    pages_with_tables: set[int] = field(default_factory=set)

    # Shared mutable state (accumulated across phases)
    errors: PipelineErrors = field(default_factory=PipelineErrors)
    knowledge: DocumentKnowledge = field(default_factory=DocumentKnowledge)
    store: GraphStore = field(default_factory=GraphStore)

    # Entity resolution outputs (used by planning)
    fund_name_variants: dict[str, str] = field(default_factory=dict)  # name -> canonical_name mapping

    # Audit trail from extraction
    extraction_audit: str = ""  # Conflict resolution audit summary


class PhaseContext:
    """Slim context holding references to the three component contexts.

    This is what phases receive. It provides:
    - resources: Immutable shared resources (pdf, logger, etc.)
    - config: Immutable configuration (models, flags)
    - state: Mutable pipeline state (results from each phase)

    All sub-context fields are accessible directly (e.g. ``ctx.pdf``
    instead of ``ctx.resources.pdf``) via explicit properties below.
    """

    def __init__(self, resources: ExtractionResources, config: ExtractionConfig, state: PipelineState):
        self.resources = resources
        self.config = config
        self.state = state

    # -- Resource properties (read-only) --

    @property
    def pdf(self) -> PDFReader:
        return self.resources.pdf

    @property
    def semaphore(self) -> asyncio.Semaphore:
        return self.resources.semaphore

    @property
    def logger(self) -> PipelineLogger:
        return self.resources.logger

    @property
    def cost_tracker(self) -> CostTracker:
        return self.resources.cost_tracker

    # -- Config properties (read-only) --

    @property
    def smart_model(self) -> str:
        return self.config.smart_model

    @property
    def exploration_model(self) -> str:
        return self.config.exploration_model

    @property
    def planner_model(self) -> str:
        return self.config.planner_model

    @property
    def reader_model(self) -> str:
        return self.config.reader_model

    @property
    def critic_model(self) -> str:
        return self.config.critic_model

    @property
    def chunk_size(self) -> int:
        return self.config.chunk_size

    @property
    def use_critic(self) -> bool:
        return self.config.use_critic

    @property
    def verbose(self) -> bool:
        return self.config.verbose

    @property
    def max_funds(self) -> int | None:
        return self.config.max_funds

    @property
    def gleaning_passes(self) -> int:
        return self.config.gleaning_passes

    @property
    def discover_bonus(self) -> bool:
        return self.config.discover_bonus

    # -- State properties (read-write) --

    @property
    def skeleton(self) -> DocumentSkeleton | None:
        return self.state.skeleton

    @skeleton.setter
    def skeleton(self, value: DocumentSkeleton | None) -> None:
        self.state.skeleton = value

    @property
    def exploration_notes(self) -> list[ExplorationNotes]:
        return self.state.exploration_notes

    @exploration_notes.setter
    def exploration_notes(self, value: list[ExplorationNotes]) -> None:
        self.state.exploration_notes = value

    @property
    def document_logic(self) -> DocumentLogic | None:
        return self.state.document_logic

    @document_logic.setter
    def document_logic(self, value: DocumentLogic | None) -> None:
        self.state.document_logic = value

    @property
    def plan(self) -> PlannerOutput | None:
        return self.state.plan

    @plan.setter
    def plan(self, value: PlannerOutput | None) -> None:
        self.state.plan = value

    @property
    def broadcast_data(self) -> dict[str, list[dict]]:
        return self.state.broadcast_data

    @broadcast_data.setter
    def broadcast_data(self, value: dict[str, list[dict]]) -> None:
        self.state.broadcast_data = value

    @property
    def parsed_broadcast_tables(self) -> int:
        return self.state.parsed_broadcast_tables

    @parsed_broadcast_tables.setter
    def parsed_broadcast_tables(self, value: int) -> None:
        self.state.parsed_broadcast_tables = value

    @property
    def umbrella_data(self) -> dict:
        return self.state.umbrella_data

    @umbrella_data.setter
    def umbrella_data(self, value: dict) -> None:
        self.state.umbrella_data = value

    @property
    def funds_data(self) -> list:
        return self.state.funds_data

    @funds_data.setter
    def funds_data(self, value: list) -> None:
        self.state.funds_data = value

    @property
    def critic_results(self) -> list[CriticResult]:
        return self.state.critic_results

    @critic_results.setter
    def critic_results(self, value: list[CriticResult]) -> None:
        self.state.critic_results = value

    @property
    def discovered_fields(self) -> list:
        return self.state.discovered_fields

    @discovered_fields.setter
    def discovered_fields(self, value: list) -> None:
        self.state.discovered_fields = value

    @property
    def schema_suggestions(self) -> list:
        return self.state.schema_suggestions

    @schema_suggestions.setter
    def schema_suggestions(self, value: list) -> None:
        self.state.schema_suggestions = value

    @property
    def scanned_tables(self) -> list[ScannedTable]:
        return self.state.scanned_tables

    @scanned_tables.setter
    def scanned_tables(self, value: list[ScannedTable]) -> None:
        self.state.scanned_tables = value

    @property
    def pages_with_tables(self) -> set[int]:
        return self.state.pages_with_tables

    @pages_with_tables.setter
    def pages_with_tables(self, value: set[int]) -> None:
        self.state.pages_with_tables = value

    @property
    def errors(self) -> PipelineErrors:
        return self.state.errors

    @errors.setter
    def errors(self, value: PipelineErrors) -> None:
        self.state.errors = value

    @property
    def knowledge(self) -> DocumentKnowledge:
        return self.state.knowledge

    @knowledge.setter
    def knowledge(self, value: DocumentKnowledge) -> None:
        self.state.knowledge = value

    @property
    def store(self) -> GraphStore:
        return self.state.store

    @store.setter
    def store(self, value: GraphStore) -> None:
        self.state.store = value

    @property
    def fund_name_variants(self) -> dict[str, str]:
        return self.state.fund_name_variants

    @fund_name_variants.setter
    def fund_name_variants(self, value: dict[str, str]) -> None:
        self.state.fund_name_variants = value

    @property
    def extraction_audit(self) -> str:
        return self.state.extraction_audit

    @extraction_audit.setter
    def extraction_audit(self, value: str) -> None:
        self.state.extraction_audit = value

    def create_search_context(self) -> PDFReader:
        """Return the PDF reader (which now includes search context capabilities)."""
        return self.resources.pdf


T = TypeVar("T")


class PhaseRunner(ABC, Generic[T]):
    """Base class for pipeline phase runners.

    Each phase:
    - Has a name for logging
    - Takes a PhaseContext with shared state
    - Produces a typed result
    - Handles errors gracefully
    """

    name: str = "unnamed"

    def __init__(self, context: PhaseContext):
        """Initialize the phase runner.

        Args:
            context: Shared pipeline context.
        """
        self.context = context
        self.logger = context.logger

    @abstractmethod
    async def run(self) -> T:
        """Execute the phase.

        Returns:
            Phase-specific result type.
        """
        pass

    def log(self, message: str, level: str = "info", **data):
        """Log a message with phase context."""
        if level == "debug":
            self.logger.debug(f"[{self.name}] {message}", **data)
        elif level == "warning":
            self.logger.warning(f"[{self.name}] {message}", **data)
        elif level == "error":
            self.logger.error(f"[{self.name}] {message}", **data)
        else:
            self.logger.info(f"[{self.name}] {message}", **data)

    def start(self, total: int = 0, model: str = ""):
        """Signal phase start."""
        self.logger.start_phase(self.name, total, model)

    def end(self, message: str = ""):
        """Signal phase end."""
        self.logger.end_phase(message)

    def progress(self, current: int, total: int, item: str = ""):
        """Report progress."""
        self.logger.progress(current, total, item)
