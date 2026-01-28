"""Knowledge graph storage and query engine for KG-RAG extraction.

Single deep module that owns the full lifecycle of graph data: storage,
indexing, domain-specific writes (fund/table recording), queries
(extraction context, fund context), and mutations (name normalization).

Callers interact through a small set of high-level public methods without
needing to know about internal indexing, key prefixing, or relation wiring.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from extractor.core.fund_names import fund_names_match
from extractor.pydantic_models.graph_models import (
    CrossRefHint,
    Entity,
    EntityType,
    ExtractionHint,
    Fact,
    FundContext,
    Relation,
    RelationType,
    TableHint,
)

logger = logging.getLogger(__name__)

# Fee-like fields that should look for table hints
_TABLE_FIELD_TYPES = frozenset(
    ["isin", "management_fee", "performance_fee", "subscription_fee", "redemption_fee"],
)


@dataclass
class GraphStore:
    """Knowledge graph engine for the extraction pipeline.

    Provides a deep interface: callers record funds/tables, query extraction
    context, and normalize names through simple public methods.  All internal
    complexity (indexing, key prefixing, relation wiring, deduplication) is
    hidden behind this interface.
    """

    entities: dict[str, Entity] = field(default_factory=dict)
    relations: list[Relation] = field(default_factory=list)
    facts: list[Fact] = field(default_factory=list)

    _relations_by_subject: dict[str, list[Relation]] = field(
        default_factory=lambda: defaultdict(list),
    )
    _relations_by_object: dict[str, list[Relation]] = field(
        default_factory=lambda: defaultdict(list),
    )
    _relations_by_type: dict[RelationType, list[Relation]] = field(
        default_factory=lambda: defaultdict(list),
    )
    _facts_by_entity: dict[str, list[Fact]] = field(
        default_factory=lambda: defaultdict(list),
    )
    _facts_by_field: dict[str, list[Fact]] = field(
        default_factory=lambda: defaultdict(list),
    )
    _entities_by_type: dict[EntityType, list[Entity]] = field(
        default_factory=lambda: defaultdict(list),
    )

    # Domain writes — exploration phase

    def record_fund(
        self,
        name: str,
        section_pages: list[int] | None,
        has_dedicated_section: bool,
        source_phase: str,
        source_page: int | None = None,
    ) -> Entity:
        """Record a discovered fund entity."""
        return self._add_entity(
            entity_type=EntityType.FUND,
            entity_id=name,
            name=name,
            properties={
                "section_pages": section_pages or [],
                "has_dedicated_section": has_dedicated_section,
            },
            source_phase=source_phase,
            source_pages=[source_page] if source_page else (section_pages or []),
        )

    def record_table(
        self,
        table_type: str,
        pages: list[int],
        columns: list[str],
        lookup_column: str | None,
        is_consolidated: bool,
        source_phase: str,
        belongs_to_fund: str | None = None,
    ) -> Entity:
        """Record a discovered table and wire relations to funds.

        For consolidated tables (has a fund-name column), automatically links
        all existing fund entities.  For fund-specific tables, links only to
        the specified fund.
        """
        table_id = f"{table_type}_{pages[0]}"

        table_entity = self._add_entity(
            entity_type=EntityType.TABLE,
            entity_id=table_id,
            name=f"{table_type.title()} Table",
            properties={
                "table_type": table_type,
                "pages": pages,
                "columns": columns,
                "lookup_column": lookup_column or self._infer_lookup_column(columns),
                "is_consolidated": is_consolidated,
            },
            source_phase=source_phase,
            source_pages=pages,
        )

        if is_consolidated:
            rel_type = (
                RelationType.HAS_ISIN_IN if table_type == "isin"
                else RelationType.HAS_FEE_IN
            )
            for fund_entity in self.get_entities_by_type(EntityType.FUND):
                self._add_relation(
                    rel_type, fund_entity.key(), table_entity.key(),
                    source_phase=source_phase,
                )
        elif belongs_to_fund:
            rel_type = (
                RelationType.HAS_ISIN_IN if table_type == "isin"
                else RelationType.HAS_FEE_IN
            )
            self._add_relation(
                rel_type, f"fund:{belongs_to_fund}", table_entity.key(),
                source_phase=source_phase,
            )

        return table_entity

    def link_funds_to_table(
        self, table_id: str, table_type: str, source_phase: str,
    ) -> None:
        """Link all known funds to a consolidated table."""
        relation_type = (
            RelationType.HAS_ISIN_IN if table_type == "isin"
            else RelationType.HAS_FEE_IN
        )
        table_key = f"table:{table_id}"
        for fund_entity in self.get_entities_by_type(EntityType.FUND):
            existing = self.query_relations(
                subject=fund_entity.key(),
                relation_type=relation_type,
                object_key=table_key,
            )
            if not existing:
                self._add_relation(
                    relation_type, fund_entity.key(), table_key,
                    source_phase=source_phase,
                )

    def add_relation(
        self,
        relation_type: RelationType,
        subject_key: str,
        object_key: str,
        properties: dict[str, Any] | None = None,
        source_phase: str = "",
        source_page: int | None = None,
        confidence: float = 1.0,
    ) -> Relation:
        """Add a relation between entities (public convenience wrapper)."""
        return self._add_relation(
            relation_type=relation_type,
            subject_key=subject_key,
            object_key=object_key,
            properties=properties,
            source_phase=source_phase,
            source_page=source_page,
            confidence=confidence,
        )

    # Domain queries — extraction phase

    def get_extraction_context(
        self, entity_name: str, field_type: str,
    ) -> ExtractionHint:
        """Get extraction hints for a specific entity and field.

        Assembles table hints, existing facts, cross-references, and page
        locations into a single hint object the extractor can act on.
        """
        entity_key = f"fund:{entity_name}"
        entity = self.get_entity(entity_key)
        existing_facts = self.get_facts_for_entity(entity_key, field_type)

        table_hint, pages, primary_source, notes = self._resolve_table_hint(
            entity_key, entity_name, field_type,
        )

        if not pages and entity:
            pages = entity.properties.get("section_pages", entity.source_pages)
            notes.append(f"Extract from section pages {pages}")

        cross_refs = self._collect_cross_refs(entity_key, field_type)
        if table_hint:
            confidence = 0.9
        elif existing_facts:
            confidence = 0.8
        elif pages:
            confidence = 0.6
        else:
            confidence = 0.5

        return ExtractionHint(
            pages=pages,
            primary_source=primary_source,
            table_hint=table_hint,
            existing_facts=existing_facts,
            cross_refs=cross_refs,
            confidence=confidence,
            notes=notes,
        )

    def get_fund_context(self, fund_name: str) -> FundContext | None:
        """Get complete context for extracting a fund.

        Returns None if the fund is not in the graph.
        """
        entity_key = f"fund:{fund_name}"
        entity = self.get_entity(entity_key)
        if not entity:
            return None

        section_pages = entity.properties.get("section_pages", entity.source_pages)

        sc_relations = self.query_relations(
            subject=entity_key,
            relation_type=RelationType.HAS_SHARE_CLASS,
        )
        share_classes = [
            self.get_entity(rel.object_key)
            for rel in sc_relations
            if self.get_entity(rel.object_key)
        ]

        isin_hint = self.get_extraction_context(fund_name, "isin")
        fee_hint = self.get_extraction_context(fund_name, "management_fee")

        existing_facts = self._facts_by_entity.get(entity_key, [])

        xref_relations = self.query_relations(
            subject=entity_key,
            relation_type=RelationType.REFERENCES,
        )
        cross_refs = [
            CrossRefHint(
                text=rel.properties.get("text", ""),
                target_pages=rel.properties.get("target_pages", []),
                is_external=rel.properties.get("is_external", False),
                external_doc=rel.properties.get("external_doc"),
            )
            for rel in xref_relations
        ]

        return FundContext(
            entity=entity,
            section_pages=section_pages,
            share_classes=share_classes,
            isin_table=isin_hint.table_hint,
            fee_table=fee_hint.table_hint,
            existing_facts=existing_facts,
            cross_refs=cross_refs,
        )

    # Domain mutations — planning phase

    def normalize_fund_names(self, name_to_canonical: dict[str, str]) -> int:
        """Update fund entity keys to canonical names after planning deduplication.

        Fixes the mismatch between exploration (raw names) and extraction
        (canonical names).  Returns the number of entities renamed.
        """
        to_rename: list[tuple[str, str, Entity]] = []
        for old_key, entity in list(self.entities.items()):
            if not old_key.startswith("fund:"):
                continue
            old_name = old_key[5:]
            canonical = name_to_canonical.get(old_name)
            if canonical and canonical != old_name:
                to_rename.append((old_key, f"fund:{canonical}", entity))

        for old_key, new_key, entity in to_rename:
            self._apply_entity_rename(old_key, new_key, entity, name_to_canonical)
            self._rekey_relations(old_key, new_key)
            self._rekey_facts(old_key, new_key)

        self._entities_by_type[EntityType.FUND] = [
            e for e in self.entities.values() if e.entity_type == EntityType.FUND
        ]
        self._deduplicate_relations()

        return len(to_rename)

    # Generic read methods

    def get_entity(self, key: str, fuzzy_fallback: bool = True) -> Entity | None:
        """Get entity by key, with optional fuzzy matching for fund names."""
        if ":" not in key:
            for prefix in ["fund", "table", "share_class"]:
                full_key = f"{prefix}:{key}"
                if full_key in self.entities:
                    return self.entities[full_key]
            if fuzzy_fallback:
                return self._fuzzy_find_fund(key)
            return None

        result = self.entities.get(key)
        if result:
            return result

        if fuzzy_fallback and key.startswith("fund:"):
            return self._fuzzy_find_fund(key[5:])

        return None

    def get_entities_by_type(self, entity_type: EntityType) -> list[Entity]:
        """Get all entities of a type."""
        return self._entities_by_type.get(entity_type, [])

    def query_relations(
        self,
        subject: str | None = None,
        relation_type: RelationType | None = None,
        object_key: str | None = None,
    ) -> list[Relation]:
        """Query relations by pattern."""
        if subject:
            candidates = self._relations_by_subject.get(subject, [])
        elif object_key:
            candidates = self._relations_by_object.get(object_key, [])
        elif relation_type:
            candidates = self._relations_by_type.get(relation_type, [])
        else:
            candidates = self.relations

        results = []
        for rel in candidates:
            if subject and rel.subject_key != subject:
                continue
            if relation_type and rel.relation_type != relation_type:
                continue
            if object_key and rel.object_key != object_key:
                continue
            results.append(rel)
        return results

    def get_facts_for_entity(
        self, entity_key: str, field_name: str | None = None,
    ) -> list[Fact]:
        """Get all facts about an entity."""
        if ":" not in entity_key:
            entity_key = f"fund:{entity_key}"
        facts = self._facts_by_entity.get(entity_key, [])
        if field_name:
            facts = [f for f in facts if f.field_name == field_name]
        return facts

    def stats(self) -> dict[str, Any]:
        """Get graph statistics."""
        return {
            "entities": len(self.entities),
            "entities_by_type": {
                t.value: len(self._entities_by_type.get(t, []))
                for t in EntityType
            },
            "relations": len(self.relations),
            "relations_by_type": {
                t.value: len(self._relations_by_type.get(t, []))
                for t in RelationType
            },
            "facts": len(self.facts),
            "facts_by_field": {
                fld: len(fcts) for fld, fcts in self._facts_by_field.items()
            },
        }

    def summary(self) -> str:
        """Human-readable summary."""
        s = self.stats()
        lines = [
            "Knowledge Graph:",
            f"  Entities: {s['entities']}"
            f" (funds: {s['entities_by_type'].get('fund', 0)},"
            f" tables: {s['entities_by_type'].get('table', 0)})",
            f"  Relations: {s['relations']}",
            f"  Facts: {s['facts']}",
        ]
        return "\n".join(lines)

    # Internal write primitives (used by domain methods + knowledge_consolidator)

    def _add_entity(
        self,
        entity_type: EntityType,
        entity_id: str,
        name: str,
        properties: dict[str, Any],
        source_phase: str,
        source_pages: list[int] | None = None,
        confidence: float = 1.0,
    ) -> Entity:
        """Add or update an entity."""
        entity = Entity(
            entity_type=entity_type,
            id=entity_id,
            name=name,
            properties=properties,
            source_phase=source_phase,
            source_pages=source_pages or [],
            confidence=confidence,
        )
        key = entity.key()

        if key in self.entities:
            existing = self.entities[key]
            for prop_key, prop_val in properties.items():
                if prop_key not in existing.properties:
                    existing.properties[prop_key] = prop_val
                elif isinstance(prop_val, list) and isinstance(
                    existing.properties[prop_key], list,
                ):
                    merged = existing.properties[prop_key] + prop_val
                    existing.properties[prop_key] = list(set(merged))
                else:
                    existing.properties[prop_key] = prop_val
            existing.confidence = max(existing.confidence, confidence)
            if source_pages:
                merged_pages = existing.source_pages + source_pages
                existing.source_pages = list(set(merged_pages))
            return existing

        self.entities[key] = entity
        self._entities_by_type[entity_type].append(entity)
        return entity

    def _add_relation(
        self,
        relation_type: RelationType,
        subject_key: str,
        object_key: str,
        properties: dict[str, Any] | None = None,
        source_phase: str = "",
        source_page: int | None = None,
        confidence: float = 1.0,
    ) -> Relation:
        """Add a relation between entities. Skips duplicates."""
        if ":" not in subject_key:
            subject_key = f"fund:{subject_key}"
        if ":" not in object_key:
            object_key = f"table:{object_key}"

        for existing in self._relations_by_subject.get(subject_key, []):
            if (existing.relation_type == relation_type
                    and existing.object_key == object_key):
                existing.confidence = max(existing.confidence, confidence)
                if properties:
                    existing.properties.update(properties)
                return existing

        relation = Relation(
            relation_type=relation_type,
            subject_key=subject_key,
            object_key=object_key,
            properties=properties or {},
            source_phase=source_phase,
            source_page=source_page,
            confidence=confidence,
        )

        self.relations.append(relation)
        self._relations_by_subject[subject_key].append(relation)
        self._relations_by_object[object_key].append(relation)
        self._relations_by_type[relation_type].append(relation)
        return relation

    def _add_fact(
        self,
        entity_key: str,
        field_name: str,
        value: Any,
        source_page: int | None = None,
        source_quote: str | None = None,
        source_type: str = "text",
        extraction_phase: str = "",
        confidence: float = 0.8,
    ) -> Fact:
        """Record an extracted fact."""
        if ":" not in entity_key:
            entity_key = f"fund:{entity_key}"

        fact = Fact(
            entity_key=entity_key,
            field_name=field_name,
            value=value,
            source_page=source_page,
            source_quote=source_quote,
            source_type=source_type,
            extraction_phase=extraction_phase,
            confidence=confidence,
        )

        self.facts.append(fact)
        self._facts_by_entity[entity_key].append(fact)
        self._facts_by_field[field_name].append(fact)
        return fact

    # Internal helpers

    def _resolve_table_hint(
        self,
        entity_key: str,
        entity_name: str,
        field_type: str,
    ) -> tuple[TableHint | None, list[int], str, list[str]]:
        """Resolve table hint for a field type."""
        if field_type not in _TABLE_FIELD_TYPES:
            return None, [], "text", []

        rel_type = (
            RelationType.HAS_ISIN_IN if field_type == "isin"
            else RelationType.HAS_FEE_IN
        )
        table_relations = self.query_relations(
            subject=entity_key, relation_type=rel_type,
        )
        if not table_relations:
            return None, [], "text", []

        table_entity = self.get_entity(table_relations[0].object_key)
        if not table_entity:
            return None, [], "text", []

        props = table_entity.properties
        hint = TableHint(
            table_id=table_entity.id,
            pages=props.get("pages", []),
            columns=props.get("columns", []),
            lookup_column=props.get("lookup_column", "Fund Name"),
            lookup_value=entity_name,
            is_consolidated=props.get("is_consolidated", True),
        )
        note = f"{field_type.upper()} in table on pages {hint.pages}"
        return hint, hint.pages, "table", [note]

    def _collect_cross_refs(
        self, entity_key: str, field_type: str,
    ) -> list[CrossRefHint]:
        """Collect cross-reference hints matching the given field type."""
        xref_relations = self.query_relations(
            subject=entity_key, relation_type=RelationType.REFERENCES,
        )
        return [
            CrossRefHint(
                text=rel.properties.get("text", ""),
                target_pages=rel.properties.get("target_pages", []),
                is_external=rel.properties.get("is_external", False),
                external_doc=rel.properties.get("external_doc"),
            )
            for rel in xref_relations
            if rel.properties.get("field_hint") == field_type
        ]

    def _apply_entity_rename(
        self,
        old_key: str,
        new_key: str,
        entity: Entity,
        name_to_canonical: dict[str, str],
    ) -> None:
        """Rename or merge an entity."""
        if new_key in self.entities:
            existing = self.entities[new_key]
            for prop_key, prop_val in entity.properties.items():
                if prop_key not in existing.properties:
                    existing.properties[prop_key] = prop_val
                elif isinstance(prop_val, list) and isinstance(
                    existing.properties[prop_key], list,
                ):
                    merged = existing.properties[prop_key] + prop_val
                    existing.properties[prop_key] = list(set(merged))
            merged_pages = existing.source_pages + entity.source_pages
            existing.source_pages = list(set(merged_pages))
            existing.confidence = max(existing.confidence, entity.confidence)
        else:
            entity.id = name_to_canonical.get(entity.id, entity.id)
            entity.name = name_to_canonical.get(entity.name, entity.name)
            self.entities[new_key] = entity

        del self.entities[old_key]

    def _rekey_relations(self, old_key: str, new_key: str) -> None:
        """Update all relations referencing old_key to use new_key."""
        for relation in self.relations:
            if relation.subject_key == old_key:
                relation.subject_key = new_key
            if relation.object_key == old_key:
                relation.object_key = new_key

        if old_key in self._relations_by_subject:
            self._relations_by_subject[new_key].extend(self._relations_by_subject.pop(old_key))
        if old_key in self._relations_by_object:
            self._relations_by_object[new_key].extend(self._relations_by_object.pop(old_key))

    def _rekey_facts(self, old_key: str, new_key: str) -> None:
        """Update all facts referencing old_key to use new_key."""
        for fact in self.facts:
            if fact.entity_key == old_key:
                fact.entity_key = new_key

        if old_key in self._facts_by_entity:
            self._facts_by_entity[new_key].extend(self._facts_by_entity.pop(old_key))

    def _deduplicate_relations(self) -> int:
        """Remove duplicate relations. Returns count of duplicates removed."""
        seen: set[tuple] = set()
        unique_relations = []
        duplicates = 0

        for rel in self.relations:
            key = (rel.relation_type, rel.subject_key, rel.object_key)
            if key not in seen:
                seen.add(key)
                unique_relations.append(rel)
            else:
                duplicates += 1

        if duplicates > 0:
            self.relations = unique_relations
            self._relations_by_subject.clear()
            self._relations_by_object.clear()
            self._relations_by_type.clear()
            for rel in self.relations:
                self._relations_by_subject[rel.subject_key].append(rel)
                self._relations_by_object[rel.object_key].append(rel)
                self._relations_by_type[rel.relation_type].append(rel)

        return duplicates

    def _fuzzy_find_fund(self, fund_name: str) -> Entity | None:
        """Find a fund entity by fuzzy name matching."""
        fund_entities = self.get_entities_by_type(EntityType.FUND)
        for entity in fund_entities:
            if fund_names_match(fund_name, entity.name):
                logger.debug(
                    "Fuzzy matched fund '%s' to entity '%s'",
                    fund_name, entity.name,
                )
                return entity
        return None

    def _infer_lookup_column(self, columns: list[str]) -> str:
        """Infer the best lookup column from column headers."""
        candidates = [
            "Fund Name", "Sub-Fund", "Fund", "Name",
            "Share Class", "Class", "Class Name",
        ]
        columns_lower = {c.lower(): c for c in columns}
        for candidate in candidates:
            if candidate.lower() in columns_lower:
                return columns_lower[candidate.lower()]
        return columns[0] if columns else "Fund Name"
