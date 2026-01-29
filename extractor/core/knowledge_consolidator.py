"""Knowledge Consolidator - resolves conflicts and maintains consistency.

This module implements the "consolidate on write" pattern. Every mutation
to the knowledge graph goes through the consolidator, which:

1. Detects conflicts with existing knowledge
2. Resolves conflicts using escalating strategies
3. Maintains an audit trail of resolutions

Conflict Resolution Strategies (in order):
1. Confidence comparison - higher confidence wins
2. Source hierarchy - TOC > Table > Section > Body text
3. Recency - later in document often supersedes earlier
4. LLM verification - call smart model to verify on source pages

Entity Resolution:
- When adding a fund entity, check for similar existing entities
- Merge if similarity > threshold, otherwise keep separate
- Track aliases for fuzzy matching in extraction
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from extractor.core.graph_store import GraphStore
    from extractor.core.cost_tracker import CostTracker

from extractor.core.value_helpers import get_raw_value

logger = logging.getLogger(__name__)


class SourceType(Enum):
    """Source types in order of reliability."""
    TOC = "toc"                      # Table of contents - highest reliability
    TABLE = "table"                  # Structured table data
    DEDICATED_SECTION = "section"   # Fund's dedicated section
    BODY_TEXT = "body_text"         # General body text
    INFERRED = "inferred"           # Derived/computed value
    UNKNOWN = "unknown"             # Source not specified


# Source reliability ranking (higher = more reliable)
SOURCE_RELIABILITY = {
    SourceType.TOC: 1.0,
    SourceType.TABLE: 0.9,
    SourceType.DEDICATED_SECTION: 0.8,
    SourceType.BODY_TEXT: 0.6,
    SourceType.INFERRED: 0.4,
    SourceType.UNKNOWN: 0.3,
}


class ResolutionStrategy(Enum):
    """How a conflict was resolved."""
    CONFIDENCE = "confidence"           # Higher confidence won
    SOURCE_HIERARCHY = "source_hierarchy"  # More reliable source won
    RECENCY = "recency"                 # Later page won (supersedes)
    LLM_VERIFIED = "llm_verified"       # LLM verified correct value
    MANUAL = "manual"                   # Manually resolved
    KEPT_BOTH = "kept_both"             # Both values retained (different contexts)


@dataclass
class ConflictResolution:
    """Record of how a conflict was resolved."""
    entity_key: str
    field_name: str
    old_value: Any
    old_source_page: int | None
    old_confidence: float
    new_value: Any
    new_source_page: int | None
    new_confidence: float
    resolution: ResolutionStrategy
    kept_value: Any
    rationale: str
    timestamp: datetime = field(default_factory=datetime.now)
    llm_response: str | None = None  # If LLM was used


@dataclass
class EntityMerge:
    """Record of entity merge/deduplication."""
    canonical_key: str
    merged_key: str
    similarity_score: float
    rationale: str
    timestamp: datetime = field(default_factory=datetime.now)


class KnowledgeConsolidator:
    """Consolidates knowledge mutations, resolving conflicts on write.

    Usage:
        consolidator = KnowledgeConsolidator(knowledge, cost_tracker)

        # Instead of: knowledge.graph.add_fact(...)
        # Use: consolidator.add_fact(...)

        # The consolidator will:
        # 1. Check for conflicts
        # 2. Resolve if needed
        # 3. Write the consolidated value
        # 4. Record the resolution for audit
    """

    def __init__(
        self,
        store: GraphStore,
        cost_tracker: CostTracker | None = None,
        verification_model: str | None = None,  # Defaults to SMART_MODEL
        auto_verify_threshold: float = 0.3,  # If confidence diff < this, use LLM
    ):
        from extractor.core.config import SMART_MODEL
        self.store = store
        self.cost_tracker = cost_tracker
        self.verification_model = verification_model or SMART_MODEL
        self.auto_verify_threshold = auto_verify_threshold

        # Audit trail
        self.conflict_resolutions: list[ConflictResolution] = []
        self.entity_merges: list[EntityMerge] = []

        # Entity aliases for fuzzy matching
        self._entity_aliases: dict[str, str] = {}  # alias -> canonical_key

    @property
    def graph(self) -> GraphStore:
        return self.store

    # Fact Management (with conflict resolution)

    async def add_fact(
        self,
        entity_key: str,
        field_name: str,
        value: Any,
        source_page: int | None = None,
        source_quote: str | None = None,
        source_type: str | SourceType = SourceType.UNKNOWN,
        confidence: float = 0.5,
        extraction_phase: str = "",
        pdf_reader: Any = None,  # For LLM verification
    ) -> tuple[Any, ConflictResolution | None]:
        """Add a fact with automatic conflict detection and resolution.

        Returns:
            Tuple of (final_value, resolution_record or None if no conflict)
        """
        if isinstance(source_type, str):
            source_type = SourceType(source_type) if source_type in [e.value for e in SourceType] else SourceType.UNKNOWN

        # Normalize entity key
        if ":" not in entity_key:
            entity_key = f"fund:{entity_key}"

        # Check for existing facts on this entity+field
        existing_facts = self.graph.get_facts_for_entity(entity_key, field_name)

        if not existing_facts:
            # No conflict - just add
            self.graph._add_fact(
                entity_key, field_name, value,
                source_page=source_page,
                source_quote=source_quote,
                source_type=source_type.value,
                extraction_phase=extraction_phase,
                confidence=confidence,
            )
            return value, None

        # Conflict detected - resolve
        # Take the most recent existing fact for comparison
        existing = existing_facts[-1]

        if get_raw_value(existing.value) == get_raw_value(value):
            # Same value (possibly different provenance), just update confidence if higher
            if confidence > existing.confidence:
                existing.confidence = confidence
                existing.source_page = source_page or existing.source_page
            return value, None

        # Different values - need to resolve
        resolution = await self._resolve_conflict(
            entity_key=entity_key,
            field_name=field_name,
            old_value=existing.value,
            old_source_page=existing.source_page,
            old_confidence=existing.confidence,
            old_source_type=SourceType(existing.source_type) if existing.source_type else SourceType.UNKNOWN,
            new_value=value,
            new_source_page=source_page,
            new_confidence=confidence,
            new_source_type=source_type,
            pdf_reader=pdf_reader,
        )

        self.conflict_resolutions.append(resolution)

        if resolution.kept_value != existing.value:
            existing.value = resolution.kept_value
            existing.source_page = resolution.new_source_page if resolution.kept_value == value else existing.source_page
            existing.confidence = max(confidence, existing.confidence)

        logger.info(
            f"Conflict resolved for {entity_key}.{field_name}: "
            f"'{get_raw_value(existing.value)}' vs '{get_raw_value(value)}' → kept '{get_raw_value(resolution.kept_value)}' "
            f"({resolution.resolution.value})"
        )

        return resolution.kept_value, resolution

    async def _resolve_conflict(
        self,
        entity_key: str,
        field_name: str,
        old_value: Any,
        old_source_page: int | None,
        old_confidence: float,
        old_source_type: SourceType,
        new_value: Any,
        new_source_page: int | None,
        new_confidence: float,
        new_source_type: SourceType,
        pdf_reader: Any = None,
    ) -> ConflictResolution:
        """Resolve a conflict between two values using escalating strategies."""

        # Strategy 1: Confidence comparison
        confidence_diff = abs(new_confidence - old_confidence)
        if confidence_diff > self.auto_verify_threshold:
            if new_confidence > old_confidence:
                return ConflictResolution(
                    entity_key=entity_key,
                    field_name=field_name,
                    old_value=old_value,
                    old_source_page=old_source_page,
                    old_confidence=old_confidence,
                    new_value=new_value,
                    new_source_page=new_source_page,
                    new_confidence=new_confidence,
                    resolution=ResolutionStrategy.CONFIDENCE,
                    kept_value=new_value,
                    rationale=f"New value has higher confidence ({new_confidence:.2f} vs {old_confidence:.2f})",
                )
            else:
                return ConflictResolution(
                    entity_key=entity_key,
                    field_name=field_name,
                    old_value=old_value,
                    old_source_page=old_source_page,
                    old_confidence=old_confidence,
                    new_value=new_value,
                    new_source_page=new_source_page,
                    new_confidence=new_confidence,
                    resolution=ResolutionStrategy.CONFIDENCE,
                    kept_value=old_value,
                    rationale=f"Existing value has higher confidence ({old_confidence:.2f} vs {new_confidence:.2f})",
                )

        # Strategy 2: Source hierarchy
        old_reliability = SOURCE_RELIABILITY.get(old_source_type, 0.3)
        new_reliability = SOURCE_RELIABILITY.get(new_source_type, 0.3)

        if abs(old_reliability - new_reliability) > 0.15:
            if new_reliability > old_reliability:
                return ConflictResolution(
                    entity_key=entity_key,
                    field_name=field_name,
                    old_value=old_value,
                    old_source_page=old_source_page,
                    old_confidence=old_confidence,
                    new_value=new_value,
                    new_source_page=new_source_page,
                    new_confidence=new_confidence,
                    resolution=ResolutionStrategy.SOURCE_HIERARCHY,
                    kept_value=new_value,
                    rationale=f"New value from more reliable source ({new_source_type.value} > {old_source_type.value})",
                )
            else:
                return ConflictResolution(
                    entity_key=entity_key,
                    field_name=field_name,
                    old_value=old_value,
                    old_source_page=old_source_page,
                    old_confidence=old_confidence,
                    new_value=new_value,
                    new_source_page=new_source_page,
                    new_confidence=new_confidence,
                    resolution=ResolutionStrategy.SOURCE_HIERARCHY,
                    kept_value=old_value,
                    rationale=f"Existing value from more reliable source ({old_source_type.value} > {new_source_type.value})",
                )

        # Strategy 3: Recency (later page often supersedes)
        # This assumes later pages have more specific/updated information
        if old_source_page and new_source_page:
            if new_source_page > old_source_page + 10:  # Significant page gap
                return ConflictResolution(
                    entity_key=entity_key,
                    field_name=field_name,
                    old_value=old_value,
                    old_source_page=old_source_page,
                    old_confidence=old_confidence,
                    new_value=new_value,
                    new_source_page=new_source_page,
                    new_confidence=new_confidence,
                    resolution=ResolutionStrategy.RECENCY,
                    kept_value=new_value,
                    rationale=f"New value from later page ({new_source_page} vs {old_source_page}) - likely more specific",
                )

        # Strategy 4: LLM verification (if pdf_reader available)
        if pdf_reader and old_source_page and new_source_page:
            llm_result = await self._verify_with_llm(
                entity_key, field_name,
                old_value, old_source_page,
                new_value, new_source_page,
                pdf_reader,
            )
            if llm_result:
                return llm_result

        # Fallback: Keep newer value but with lower confidence
        return ConflictResolution(
            entity_key=entity_key,
            field_name=field_name,
            old_value=old_value,
            old_source_page=old_source_page,
            old_confidence=old_confidence,
            new_value=new_value,
            new_source_page=new_source_page,
            new_confidence=new_confidence,
            resolution=ResolutionStrategy.RECENCY,
            kept_value=new_value,
            rationale="Fallback: kept newer value (ambiguous conflict)",
        )

    async def _verify_with_llm(
        self,
        entity_key: str,
        field_name: str,
        old_value: Any,
        old_source_page: int,
        new_value: Any,
        new_source_page: int,
        pdf_reader: Any,
    ) -> ConflictResolution | None:
        """Use LLM to verify which value is correct."""
        try:
            from extractor.core.llm_client import LLMClient

            # Read both pages
            old_text = pdf_reader.read_pages(old_source_page, old_source_page)
            new_text = pdf_reader.read_pages(new_source_page, new_source_page)

            prompt = f"""I found conflicting values in a prospectus document. Help me determine which is correct.

FIELD: {field_name}
ENTITY: {entity_key}

VALUE A: "{old_value}" (found on page {old_source_page})
Page {old_source_page} text:
{old_text[:3000]}

VALUE B: "{new_value}" (found on page {new_source_page})
Page {new_source_page} text:
{new_text[:3000]}

QUESTION: Which value is the CURRENT, CORRECT value for this field?
Consider:
- Is one value historical/superseded and the other current?
- Is one value a general statement and the other specific to this entity?
- Is one value more complete/precise?

Return JSON:
{{
  "correct_value": "A" or "B",
  "rationale": "Brief explanation of why this value is correct",
  "context": "Any relevant context about why there are two values"
}}"""

            client = LLMClient(cost_tracker=self.cost_tracker)

            response = await client.complete(
                system_prompt="You are verifying data from financial documents. Be precise and cite evidence from the text.",
                user_prompt=prompt,
                model=self.verification_model,
                agent="conflict_resolver",
            )
            result = response.content

            correct = result.get("correct_value", "").upper()
            rationale = result.get("rationale", "LLM verification")

            if correct == "A":
                return ConflictResolution(
                    entity_key=entity_key,
                    field_name=field_name,
                    old_value=old_value,
                    old_source_page=old_source_page,
                    old_confidence=0.9,  # LLM verified
                    new_value=new_value,
                    new_source_page=new_source_page,
                    new_confidence=0.5,
                    resolution=ResolutionStrategy.LLM_VERIFIED,
                    kept_value=old_value,
                    rationale=rationale,
                    llm_response=str(result),
                )
            elif correct == "B":
                return ConflictResolution(
                    entity_key=entity_key,
                    field_name=field_name,
                    old_value=old_value,
                    old_source_page=old_source_page,
                    old_confidence=0.5,
                    new_value=new_value,
                    new_source_page=new_source_page,
                    new_confidence=0.9,  # LLM verified
                    resolution=ResolutionStrategy.LLM_VERIFIED,
                    kept_value=new_value,
                    rationale=rationale,
                    llm_response=str(result),
                )

        except Exception as e:
            logger.warning(f"LLM verification failed: {e}")

        return None

    # Entity Resolution (fund name deduplication)

    def resolve_entity(
        self,
        entity_type: str,
        name: str,
        source_page: int | None = None,
        confidence: float = 0.5,
        properties: dict | None = None,
    ) -> tuple[str, EntityMerge | None]:
        """Resolve an entity, merging with existing if similar enough.

        Returns:
            Tuple of (canonical_entity_key, merge_record or None)
        """
        from extractor.core.fund_names import fund_names_match, normalize_fund_name
        from extractor.pydantic_models.graph_models import EntityType

        normalized = normalize_fund_name(name)
        if normalized in self._entity_aliases:
            return self._entity_aliases[normalized], None

        # Search for similar existing entities
        entity_type_enum = EntityType(entity_type) if isinstance(entity_type, str) else entity_type
        existing_entities = self.graph.get_entities_by_type(entity_type_enum)

        best_match = None
        best_score = 0.0

        for entity in existing_entities:
            # Check direct match
            if fund_names_match(name, entity.name, threshold=0.85):
                from difflib import SequenceMatcher
                score = SequenceMatcher(None, normalize_fund_name(name), normalize_fund_name(entity.name)).ratio()
                if score > best_score:
                    best_match = entity
                    best_score = score

        if best_match and best_score > 0.85:
            # Merge into existing entity
            canonical_key = best_match.key()

            # Update properties (merge, don't overwrite)
            if properties:
                for k, v in properties.items():
                    if k not in best_match.properties:
                        best_match.properties[k] = v
                    elif isinstance(v, list) and isinstance(best_match.properties[k], list):
                        best_match.properties[k] = list(set(best_match.properties[k] + v))

            if source_page and source_page not in best_match.source_pages:
                best_match.source_pages.append(source_page)

            best_match.confidence = max(best_match.confidence, confidence)

            self._entity_aliases[normalized] = canonical_key

            merge = EntityMerge(
                canonical_key=canonical_key,
                merged_key=f"{entity_type}:{name}",
                similarity_score=best_score,
                rationale=f"Merged '{name}' into '{best_match.name}' (similarity: {best_score:.2f})",
            )
            self.entity_merges.append(merge)

            logger.debug(f"Entity merged: '{name}' → '{best_match.name}'")

            return canonical_key, merge

        # No match - create new entity
        entity = self.graph._add_entity(
            entity_type=entity_type_enum,
            entity_id=name,
            name=name,
            properties=properties or {},
            source_phase="consolidator",
            source_pages=[source_page] if source_page else [],
            confidence=confidence,
        )

        canonical_key = entity.key()
        self._entity_aliases[normalized] = canonical_key

        return canonical_key, None

    # Reconciliation

    def reconcile(self, external_refs: list | None = None) -> ReconciliationReport:
        """Generate a reconciliation report of current knowledge state.

        Args:
            external_refs: Optional list of ExternalReference objects for coverage check.

        This should be called at key checkpoints to surface issues.
        """
        from extractor.pydantic_models.graph_models import EntityType

        issues = []
        warnings = []
        stats = {}

        fund_entities = self.graph.get_entities_by_type(EntityType.FUND)
        table_entities = self.graph.get_entities_by_type(EntityType.TABLE)

        stats["fund_count"] = len(fund_entities)
        stats["table_count"] = len(table_entities)
        stats["fact_count"] = len(self.graph.facts)
        stats["conflict_resolutions"] = len(self.conflict_resolutions)
        stats["entity_merges"] = len(self.entity_merges)

        funds_without_facts = []
        for fund in fund_entities:
            facts = self.graph.get_facts_for_entity(fund.key())
            if not facts:
                funds_without_facts.append(fund.name)

        if funds_without_facts:
            warnings.append(f"{len(funds_without_facts)} funds have no extracted facts: {funds_without_facts[:5]}")

        # Check for orphaned facts (facts for non-existent entities)
        entity_keys = set(self.graph.entities.keys())
        orphaned_facts = []
        for fact in self.graph.facts:
            if fact.entity_key not in entity_keys:
                orphaned_facts.append(fact)

        if orphaned_facts:
            issues.append(f"{len(orphaned_facts)} orphaned facts (entity not found)")

        # Check for unresolved conflicts (should be 0 if consolidator is working)
        unresolved = self._find_unresolved_conflicts()
        if unresolved:
            issues.append(f"{len(unresolved)} unresolved conflicts remain")

        if external_refs:
            external_fields = set(ref.field_name for ref in external_refs)
            stats["external_fields"] = list(external_fields)

        return ReconciliationReport(
            timestamp=datetime.now(),
            stats=stats,
            issues=issues,
            warnings=warnings,
            conflict_resolutions=self.conflict_resolutions[-10:],  # Last 10
            entity_merges=self.entity_merges[-10:],  # Last 10
        )

    def _find_unresolved_conflicts(self) -> list[tuple[str, str, list]]:
        """Find any facts where we have multiple different values."""
        from collections import defaultdict

        facts_by_key: dict[tuple[str, str], list] = defaultdict(list)
        for fact in self.graph.facts:
            key = (fact.entity_key, fact.field_name)
            facts_by_key[key].append(fact)

        unresolved = []
        for (entity_key, field_name), facts in facts_by_key.items():
            unique_values = set(str(f.value) for f in facts)
            if len(unique_values) > 1:
                unresolved.append((entity_key, field_name, facts))

        return unresolved

    def get_audit_summary(self) -> str:
        """Get a human-readable audit summary."""
        lines = ["Knowledge Consolidation Audit:"]
        lines.append(f"  Total conflict resolutions: {len(self.conflict_resolutions)}")
        lines.append(f"  Total entity merges: {len(self.entity_merges)}")

        if self.conflict_resolutions:
            lines.append("\n  Recent conflict resolutions:")
            for res in self.conflict_resolutions[-5:]:
                lines.append(f"    - {res.entity_key}.{res.field_name}: '{res.old_value}' vs '{res.new_value}'")
                lines.append(f"      → kept '{res.kept_value}' ({res.resolution.value})")

        if self.entity_merges:
            lines.append("\n  Recent entity merges:")
            for merge in self.entity_merges[-5:]:
                lines.append(f"    - '{merge.merged_key}' → '{merge.canonical_key}' (score: {merge.similarity_score:.2f})")

        return "\n".join(lines)


@dataclass
class ReconciliationReport:
    """Report from reconciliation check."""
    timestamp: datetime
    stats: dict[str, Any]
    issues: list[str]  # Critical issues that need attention
    warnings: list[str]  # Non-critical warnings
    conflict_resolutions: list[ConflictResolution]
    entity_merges: list[EntityMerge]

    def is_healthy(self) -> bool:
        """Return True if no critical issues."""
        return len(self.issues) == 0

