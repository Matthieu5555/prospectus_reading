"""Field extraction strategy and knowledge context.

Extracted from document_knowledge.py to break the circular dependency
between DocumentKnowledge and GraphStore.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from extractor.core.document_knowledge import DocumentKnowledge
    from extractor.core.graph_store import GraphStore
    from extractor.pydantic_models.graph_models import RelationType

from extractor.core.document_knowledge import FindingType


@dataclass
class FieldStrategy:
    """Extraction strategy for a specific field based on accumulated knowledge."""

    strategy: str  # "external" | "not_applicable" | "table_lookup" | "text_extraction" | "search"
    pages: list[int] | None = None
    external_doc: str | None = None
    external_quote: str | None = None
    table_pages: list[int] | None = None
    lookup_column: str | None = None
    value_column: str | None = None
    confidence: float = 0.5

    def should_skip_extraction(self) -> bool:
        """Return True if we should skip extraction entirely for this field."""
        return self.strategy in ("external", "not_applicable")


@dataclass
class KnowledgeContext:
    """Unified query interface for extraction phases.

    Provides the "standing on shoulders of giants" interface.  Before any
    extraction work, query this context to decide whether to skip, do a
    targeted table lookup, or fall back to search.
    """

    knowledge: DocumentKnowledge
    store: GraphStore

    def get_field_strategy(self, field_name: str, fund_name: str | None = None) -> FieldStrategy:
        """Determine extraction strategy for a field based on all accumulated knowledge."""
        return (
            self._check_external(field_name, fund_name)
            or self._check_absent(field_name)
            or self._check_table(field_name)
            or self._check_pages(field_name)
            or self._check_cross_refs(field_name, fund_name)
            or FieldStrategy(strategy="search", confidence=0.5)
        )

    def _check_external(self, field_name: str, fund_name: str | None) -> FieldStrategy | None:
        """Return external strategy if field is in an external document."""
        ext_ref = self.knowledge.get_external_ref_for_field(field_name)
        if ext_ref and (ext_ref.entity_name is None or ext_ref.entity_name == fund_name):
            return FieldStrategy(
                strategy="external",
                external_doc=ext_ref.external_doc,
                external_quote=ext_ref.source_quote,
                confidence=0.95,
            )
        return None

    def _check_absent(self, field_name: str) -> FieldStrategy | None:
        """Return not_applicable strategy if field is known absent."""
        if self.knowledge.is_field_explicitly_absent(field_name):
            return FieldStrategy(strategy="not_applicable", confidence=0.95)
        if self.knowledge.is_field_absent(field_name):
            return FieldStrategy(strategy="not_applicable", confidence=0.8)
        return None

    def _check_table(self, field_name: str) -> FieldStrategy | None:
        """Return table_lookup strategy if field is in a discovered table."""
        table_findings = [
            f for f in self.knowledge.get_findings_by_type(FindingType.TABLE_LOCATION)
            if f.field_name == field_name
        ]
        if not table_findings:
            return None

        pages = sorted(set(p for f in table_findings for p in f.pages))
        lookup_col, value_col = _infer_table_columns(table_findings, field_name)

        return FieldStrategy(
            strategy="table_lookup",
            table_pages=pages,
            pages=pages,
            lookup_column=lookup_col,
            value_column=value_col,
            confidence=0.9,
        )

    def _check_pages(self, field_name: str) -> FieldStrategy | None:
        """Return text_extraction strategy if known page locations exist."""
        pages = self.knowledge.get_pages_for_field(field_name)
        if pages:
            return FieldStrategy(strategy="text_extraction", pages=pages, confidence=0.8)

        inv_pages = self.knowledge.get_inventory_pages_for_field(field_name)
        if inv_pages:
            return FieldStrategy(strategy="text_extraction", pages=inv_pages, confidence=0.7)

        return None

    def _check_cross_refs(self, field_name: str, fund_name: str | None) -> FieldStrategy | None:
        """Return text_extraction strategy if a cross-reference points to pages for this field."""
        if fund_name is None:
            return None

        from extractor.pydantic_models.graph_models import RelationType

        relations = self.store.query_relations(
            subject=f"fund:{fund_name}",
            relation_type=RelationType.REFERENCES,
        )
        pages: set[int] = set()
        for rel in relations:
            props = rel.properties
            if props.get("field_hint") != field_name:
                continue
            if props.get("is_external"):
                continue
            target = props.get("target_pages")
            if target:
                pages.update(target)

        if pages:
            return FieldStrategy(
                strategy="text_extraction",
                pages=sorted(pages),
                confidence=0.85,
            )
        return None

    def get_not_found_reason(self, field_name: str) -> str:
        """Get appropriate NOT_FOUND reason for a field."""
        return self.knowledge.suggest_not_found_reason(field_name)


def _infer_table_columns(
    table_findings: list,
    field_name: str,
) -> tuple[str | None, str | None]:
    """Infer lookup and value columns from table finding metadata."""
    lookup_col = None
    value_col = None
    for finding in table_findings:
        cols = finding.metadata.get("columns")
        if not cols:
            continue
        for candidate in ["Fund Name", "Sub-Fund", "Share Class", "Class"]:
            if candidate in cols:
                lookup_col = candidate
                break
        if field_name == "isin":
            for candidate in ["ISIN", "Isin", "ISIN Code"]:
                if candidate in cols:
                    value_col = candidate
                    break
        break
    return lookup_col, value_col
