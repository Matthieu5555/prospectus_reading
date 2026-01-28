"""Knowledge graph models for KG-RAG extraction."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Literal


class EntityType(str, Enum):
    """Types of entities in the knowledge graph."""
    FUND = "fund"
    SHARE_CLASS = "share_class"
    TABLE = "table"
    SECTION = "section"
    UMBRELLA = "umbrella"
    CONSTRAINT = "constraint"
    EXTERNAL_DOC = "external_doc"


class RelationType(str, Enum):
    """Types of relations between entities."""
    HAS_SHARE_CLASS = "has_share_class"
    MENTIONED_ON = "mentioned_on"
    HAS_SECTION = "has_section"
    HAS_ISIN_IN = "has_isin_in"
    HAS_FEE_IN = "has_fee_in"
    HAS_CONSTRAINT_IN = "has_constraint_in"
    CONTAINS_DATA_FOR = "contains_data_for"
    LOCATED_ON = "located_on"
    REFERENCES = "references"
    REFERENCES_EXTERNAL = "references_external"
    INHERITS_FROM = "inherits_from"
    APPLIES_TO = "applies_to"


@dataclass
class Entity:
    """A node in the knowledge graph."""
    entity_type: EntityType
    id: str
    name: str
    properties: dict[str, Any]
    source_phase: str
    source_pages: list[int]
    confidence: float = 1.0

    def key(self) -> str:
        return f"{self.entity_type.value}:{self.id}"


@dataclass
class Relation:
    """A directed edge in the knowledge graph."""
    relation_type: RelationType
    subject_key: str
    object_key: str
    properties: dict[str, Any] = field(default_factory=dict)
    source_phase: str = ""
    source_page: int | None = None
    confidence: float = 1.0


@dataclass
class Fact:
    """An extracted value with provenance."""
    entity_key: str
    field_name: str
    value: Any
    source_page: int | None
    source_quote: str | None
    source_type: Literal["table", "text", "search", "inferred"]
    extraction_phase: str
    confidence: float
    created_at: datetime = field(default_factory=datetime.now)
    is_verified: bool = False
    superseded_by: str | None = None


@dataclass
class TableHint:
    """Hint for extracting from a table."""
    table_id: str
    pages: list[int]
    columns: list[str]
    lookup_column: str
    lookup_value: str
    is_consolidated: bool


@dataclass
class CrossRefHint:
    """Hint from a cross-reference."""
    text: str
    target_pages: list[int]
    is_external: bool
    external_doc: str | None = None


@dataclass
class ExtractionHint:
    """Complete hint for extracting a field."""
    pages: list[int]
    primary_source: Literal["table", "text", "cross_ref", "inherited"]
    table_hint: TableHint | None = None
    existing_facts: list[Fact] = field(default_factory=list)
    cross_refs: list[CrossRefHint] = field(default_factory=list)
    confidence: float = 0.5
    notes: list[str] = field(default_factory=list)


@dataclass
class FundContext:
    """Complete context for extracting a fund."""
    entity: Entity
    section_pages: list[int]
    share_classes: list[Entity]
    isin_table: TableHint | None = None
    fee_table: TableHint | None = None
    existing_facts: list[Fact] = field(default_factory=list)
    cross_refs: list[CrossRefHint] = field(default_factory=list)
