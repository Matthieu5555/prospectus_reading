"""Document Knowledge Graph - shared mutable store for agent communication.

This module provides a central knowledge store that any agent can read/write.
It enables:
- Explorers to record document structure discoveries
- Extractors to find information based on exploration findings
- Agents to post questions for later resolution
- Tracking of external document references

The KnowledgeContext class provides a unified query interface for extraction,
implementing the "standing on shoulders of giants" principle where each phase
builds on accumulated knowledge from prior phases.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from extractor.pydantic_models.pipeline import DocumentInventory

# Import at runtime for discovered_fields storage
from extractor.pydantic_models.output import DiscoveredField


class FindingType(str, Enum):
    """Types of findings agents can record."""

    TABLE_LOCATION = "table_location"       # "ISIN table on page 250"
    SECTION_LOCATION = "section_location"   # "Investment Restrictions at page 120"
    CROSS_REFERENCE = "cross_reference"     # "Page 45 says 'See Appendix E'"
    PATTERN_MATCH = "pattern_match"         # "Found LU0 ISINs on pages 200-210"
    FIELD_VALUE = "field_value"             # "Management fee for Fund X is 1.5%"
    STRUCTURAL = "structural"               # "TOC spans pages 3-5"


class QuestionPriority(str, Enum):
    """Priority levels for unresolved questions."""

    HIGH = "high"       # Critical field missing
    MEDIUM = "medium"   # Important but not blocking
    LOW = "low"         # Nice to have


@dataclass
class Finding:
    """A discovery made by an agent during processing.

    Findings allow agents to share discoveries with downstream phases.
    For example, an explorer might record "ISIN table found on page 250"
    which extractors can later use.
    """

    finding_type: FindingType
    description: str
    source_agent: str           # "explorer_1-50", "extractor_fund_x"
    pages: list[int]            # Related page numbers
    entity_name: str | None = None  # Fund/share class name if applicable
    field_name: str | None = None   # Field this relates to (isin, fee, etc.)
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    def __str__(self) -> str:
        pages_str = f"pages {self.pages}" if self.pages else "no pages"
        return f"[{self.finding_type.value}] {self.description} ({pages_str})"


@dataclass
class Question:
    """An unresolved question posted by an agent.

    Questions allow agents to flag information they couldn't find
    for potential resolution by specialized resolvers or manual review.
    """

    question: str               # "Where are dividend dates for Fund X?"
    field_name: str             # Field this relates to
    entity_name: str            # Fund/share class this is about
    source_agent: str           # Agent that posted the question
    priority: QuestionPriority = QuestionPriority.MEDIUM
    pages_searched: list[int] = field(default_factory=list)
    resolved: bool = False
    resolution: str | None = None
    timestamp: datetime = field(default_factory=datetime.now)

    def resolve(self, resolution: str) -> None:
        """Mark question as resolved with an answer."""
        self.resolved = True
        self.resolution = resolution


@dataclass
class ExternalReference:
    """A reference to information in an external document.

    Tracks when the prospectus points to external documents
    like KIID, Annual Report, etc. for specific information.
    """

    external_doc: str           # "KIID", "Annual Report", "Supplement"
    field_name: str             # What field is referenced externally
    source_page: int            # Page where reference was found
    source_quote: str           # Verbatim text of the reference
    entity_name: str | None = None  # Specific fund if applicable, None = all
    timestamp: datetime = field(default_factory=datetime.now)

    def __str__(self) -> str:
        entity = f" for {self.entity_name}" if self.entity_name else " (all funds)"
        return f"{self.field_name}{entity} -> {self.external_doc} (page {self.source_page})"


@dataclass
class DocumentKnowledge:
    """Central knowledge store for the extraction pipeline.

    Any agent can read from or write to this store. It accumulates
    knowledge across phases to enable better extraction.

    Usage:
        knowledge = DocumentKnowledge()

        # Explorer records a finding
        knowledge.add_finding(Finding(
            finding_type=FindingType.TABLE_LOCATION,
            description="ISIN table with all share classes",
            source_agent="explorer_200-250",
            pages=[245, 246, 247],
            field_name="isin"
        ))

        # Extractor queries for ISIN locations
        isin_pages = knowledge.get_pages_for_field("isin")
        # Returns [245, 246, 247]

        # Extractor posts a question
        knowledge.add_question(Question(
            question="Cannot find dividend dates for Fund X",
            field_name="dividend_dates",
            entity_name="JPM Global Bond Fund",
            source_agent="extractor_fund_x",
            priority=QuestionPriority.HIGH
        ))
    """

    # Document structure
    toc_pages: list[int] = field(default_factory=list)
    appendix_map: dict[str, list[int]] = field(default_factory=dict)  # "Appendix E" -> [200-210]

    # Agent discoveries
    findings: list[Finding] = field(default_factory=list)

    # Unresolved questions
    questions: list[Question] = field(default_factory=list)

    # External document references
    external_refs: list[ExternalReference] = field(default_factory=list)

    # Document inventory (aggregated from all explorers)
    document_inventory: DocumentInventory | None = None

    # Discovered fields - LLM-suggested fields not in our schema
    # Any agent can add to this when they notice interesting data
    discovered_fields: list[DiscoveredField] = field(default_factory=list)

    # Quick lookup caches (updated automatically)
    _field_pages: dict[str, set[int]] = field(default_factory=dict)
    _entity_findings: dict[str, list[Finding]] = field(default_factory=dict)

    def add_finding(self, finding: Finding) -> None:
        """Record a new finding and update caches."""
        self.findings.append(finding)

        # Update field->pages cache
        if finding.field_name:
            if finding.field_name not in self._field_pages:
                self._field_pages[finding.field_name] = set()
            self._field_pages[finding.field_name].update(finding.pages)

        # Update entity->findings cache
        if finding.entity_name:
            if finding.entity_name not in self._entity_findings:
                self._entity_findings[finding.entity_name] = []
            self._entity_findings[finding.entity_name].append(finding)

    def add_question(self, question: Question) -> None:
        """Record an unresolved question."""
        self.questions.append(question)

    def add_external_ref(self, ref: ExternalReference) -> None:
        """Record an external document reference."""
        self.external_refs.append(ref)

    def set_toc_pages(self, pages: list[int]) -> None:
        """Record table of contents location."""
        self.toc_pages = pages

    # Query methods

    def get_pages_for_field(self, field_name: str) -> list[int]:
        """Get all pages known to contain information for a field."""
        return sorted(self._field_pages.get(field_name, set()))

    def get_findings_by_type(self, finding_type: FindingType) -> list[Finding]:
        """Get all findings of a specific type."""
        return [f for f in self.findings if f.finding_type == finding_type]

    def get_external_ref_for_field(self, field_name: str) -> ExternalReference | None:
        """Get external reference for a field if one exists."""
        for ref in self.external_refs:
            if ref.field_name == field_name:
                return ref
        return None

    def get_pages_already_searched(
        self, field_name: str, entity_name: str | None = None,
    ) -> set[int]:
        """Return pages already searched for a field, based on posted questions."""
        pages: set[int] = set()
        for q in self.questions:
            if q.field_name != field_name:
                continue
            if entity_name and q.entity_name not in (entity_name, "multiple_funds"):
                continue
            pages.update(q.pages_searched)
        return pages

    def get_unresolved_questions(self, priority: QuestionPriority | None = None) -> list[Question]:
        """Get unresolved questions, optionally filtered by priority."""
        questions = [q for q in self.questions if not q.resolved]
        if priority:
            questions = [q for q in questions if q.priority == priority]
        return questions

    # Inventory query methods

    def is_field_absent(self, field_name: str) -> bool:
        """Check if a field is confirmed absent from the document (via inventory)."""
        if self.document_inventory:
            return self.document_inventory.is_absent(field_name)
        return False

    def is_field_explicitly_absent(self, field_name: str) -> bool:
        """Check if a field is explicitly stated as not applicable (via inventory)."""
        if self.document_inventory:
            return self.document_inventory.is_explicitly_absent(field_name)
        return False

    def suggest_not_found_reason(self, field_name: str) -> str:
        """Suggest appropriate NOT_FOUND reason based on inventory and external refs.

        Returns string reason: 'not_applicable', 'in_external_doc', 'not_in_document',
        or 'extraction_failed' if unknown.
        """
        # First check inventory
        if self.document_inventory:
            reason = self.document_inventory.get_not_found_reason(field_name)
            if reason:
                return reason

        # Fall back to external_refs
        if self.get_external_ref_for_field(field_name):
            return "in_external_doc"

        # Default to extraction_failed (needs investigation)
        return "extraction_failed"

    def get_inventory_pages_for_field(self, field_name: str) -> list[int]:
        """Get pages from inventory for a field."""
        if self.document_inventory:
            return self.document_inventory.get_pages_for_field(field_name)
        return []

    # Convenience methods for common patterns

    def record_external_reference(
        self,
        field_name: str,
        external_doc: str,
        source_page: int,
        source_quote: str,
        source_agent: str,
        entity_name: str | None = None
    ) -> None:
        """Convenience method to record an external reference."""
        self.add_external_ref(ExternalReference(
            external_doc=external_doc,
            field_name=field_name,
            source_page=source_page,
            source_quote=source_quote,
            entity_name=entity_name
        ))
        # Also add as a finding
        self.add_finding(Finding(
            finding_type=FindingType.CROSS_REFERENCE,
            description=f"{field_name} referenced in {external_doc}",
            source_agent=source_agent,
            pages=[source_page],
            field_name=field_name,
            entity_name=entity_name,
            metadata={"external_doc": external_doc, "quote": source_quote}
        ))

    def ask_question(
        self,
        question: str,
        field_name: str,
        entity_name: str,
        source_agent: str,
        priority: QuestionPriority = QuestionPriority.MEDIUM,
        pages_searched: list[int] | None = None
    ) -> None:
        """Convenience method to post a question."""
        self.add_question(Question(
            question=question,
            field_name=field_name,
            entity_name=entity_name,
            source_agent=source_agent,
            priority=priority,
            pages_searched=pages_searched or []
        ))

    def add_discovered_field(self, discovered: DiscoveredField) -> None:
        """Record a discovered field that's not in our schema.

        Any agent can call this when they find interesting data
        that doesn't fit existing schema fields.
        """
        # Avoid duplicates
        existing_names = {d.field_name for d in self.discovered_fields}
        if discovered.field_name not in existing_names:
            self.discovered_fields.append(discovered)

    # Statistics

    def stats(self) -> dict[str, Any]:
        """Get statistics about the knowledge graph."""
        return {
            "total_findings": len(self.findings),
            "findings_by_type": {
                t.value: len(self.get_findings_by_type(t))
                for t in FindingType
            },
            "fields_with_pages": list(self._field_pages.keys()),
            "entities_with_findings": list(self._entity_findings.keys()),
            "external_refs": len(self.external_refs),
            "external_ref_fields": [ref.field_name for ref in self.external_refs],
            "unresolved_questions": len(self.get_unresolved_questions()),
            "high_priority_questions": len(self.get_unresolved_questions(QuestionPriority.HIGH)),
            "toc_pages": self.toc_pages,
            "appendices": list(self.appendix_map.keys()),
            "discovered_fields": len(self.discovered_fields),
            "discovered_field_names": [d.field_name for d in self.discovered_fields],
        }


