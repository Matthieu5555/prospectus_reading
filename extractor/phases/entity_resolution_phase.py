"""Entity Resolution Phase - consolidates and validates entities after exploration.

This phase runs AFTER exploration, BEFORE logic/planning. It:

1. Gets canonical fund names from TOC (skeleton) - AUTHORITATIVE SOURCE
2. Maps exploration mentions to TOC entries using fuzzy matching
3. Falls back to LLM deduplication only if TOC has no fund names
4. Scores confidence for each cluster
5. Updates the knowledge graph with clean entities

TOC-first architecture: The document's own TOC is ground truth.
Exploration discovers page locations; TOC provides canonical names.
"""

from dataclasses import dataclass, field

from extractor.phases.phase_base import PhaseRunner
from extractor.pydantic_models import ExplorationNotes, DocumentSkeleton
from extractor.pydantic_models.graph_models import EntityType
from extractor.core.fund_names import (
    normalize_fund_name,
    fund_names_match,
    deduplicate_with_llm,
    deduplicate_heuristic,
    FundNameMapping,
)
from extractor.core.knowledge_consolidator import KnowledgeConsolidator, EntityMerge


@dataclass
class ResolvedFundMention:
    """A single fund mention with resolution metadata.

    Unlike ``pydantic_models.exploration_models.FundMention`` (which captures
    raw explorer output), this dataclass adds resolution-specific fields:
    ``from_toc``, ``from_table``, and ``explorer_chunk``.
    """
    name: str
    page: int
    has_dedicated_section: bool
    from_toc: bool = False
    from_table: bool = False
    explorer_chunk: tuple[int, int] | None = None


@dataclass
class FundCluster:
    """A cluster of fund name mentions that refer to the same fund."""
    canonical_name: str
    variants: list[str]
    mentions: list[ResolvedFundMention]
    confidence: float = 0.5
    confidence_factors: list[str] = field(default_factory=list)

    @property
    def total_mentions(self) -> int:
        return len(self.mentions)

    @property
    def pages(self) -> list[int]:
        return sorted(set(m.page for m in self.mentions))


@dataclass
class EntityResolutionResult:
    """Result from entity resolution phase."""
    canonical_funds: list[str]
    total_mentions: int
    clusters_formed: int
    duplicates_merged: int
    low_confidence_filtered: int
    reconciliation_issues: list[str]


class EntityResolutionPhase(PhaseRunner[EntityResolutionResult]):
    """Phase 1.5: Entity Resolution.

    Runs after exploration, before logic/planning.
    Consolidates fund mentions into canonical entities using LLM.
    """

    name = "EntityResolution"

    async def run(self) -> EntityResolutionResult:
        """Execute entity resolution.

        Returns:
            EntityResolutionResult with consolidated fund list.
        """
        self.start(1, model=self.context.reader_model)

        exploration_notes = self.context.exploration_notes
        if not exploration_notes:
            self.log("No exploration notes, skipping entity resolution", "warning")
            return EntityResolutionResult(
                canonical_funds=[],
                total_mentions=0,
                clusters_formed=0,
                duplicates_merged=0,
                low_confidence_filtered=0,
                reconciliation_issues=[],
            )

        # Step 1: Extract all fund mentions
        mentions = self._extract_all_mentions(exploration_notes)
        self.log(f"Extracted {len(mentions)} fund mentions from exploration")

        # Step 2: Get canonical fund names from TOC (authoritative source)
        skeleton = self.context.skeleton
        toc_fund_names = skeleton.get_toc_fund_names() if skeleton else []

        if toc_fund_names:
            # TOC-first: use TOC as ground truth
            self.log(f"Using {len(toc_fund_names)} canonical funds from TOC (authoritative)")
            name_mapping = self._map_mentions_to_toc(mentions, toc_fund_names, skeleton)
        else:
            # Fallback: use LLM deduplication
            self.log("No funds in TOC, falling back to LLM deduplication")
            unique_names = list(set(m.name for m in mentions))
            self.log(f"Deduplicating {len(unique_names)} unique fund names with LLM...")

            try:
                name_mapping = await deduplicate_with_llm(
                    unique_names,
                    model=self.context.reader_model,
                    cost_tracker=self.context.cost_tracker,
                )
                self.log(f"LLM identified {len(name_mapping.canonical_names)} canonical funds, merged {name_mapping.duplicates_found} duplicates")
            except Exception as e:
                self.log(f"LLM deduplication failed ({e}), falling back to heuristic", "warning")
                name_mapping = deduplicate_heuristic([(m.name, m.page) for m in mentions])

        # Step 3: Build clusters from LLM groupings
        clusters = self._build_clusters_from_mapping(mentions, name_mapping)
        self.log(f"Formed {len(clusters)} clusters from {len(mentions)} mentions")

        # Step 4: Score confidence for each cluster
        for cluster in clusters:
            self._score_cluster_confidence(cluster)

        # Step 5: Filter low-confidence clusters (noise)
        min_confidence = 0.4
        valid_clusters = [c for c in clusters if c.confidence >= min_confidence]
        filtered_count = len(clusters) - len(valid_clusters)

        if filtered_count > 0:
            filtered_names = [c.canonical_name for c in clusters if c.confidence < min_confidence]
            self.log(f"Filtered {filtered_count} low-confidence clusters: {filtered_names[:5]}")

        # Step 6: Update knowledge graph with canonical entities
        consolidator = KnowledgeConsolidator(
            self.context.store,
            cost_tracker=self.context.cost_tracker,
        )

        canonical_funds = []
        for cluster in valid_clusters:
            # Add/merge entity
            entity_key, merge = consolidator.resolve_entity(
                entity_type="fund",
                name=cluster.canonical_name,
                source_page=cluster.pages[0] if cluster.pages else None,
                confidence=cluster.confidence,
                properties={
                    "variants": cluster.variants,
                    "mention_count": cluster.total_mentions,
                    "has_dedicated_section": any(m.has_dedicated_section for m in cluster.mentions),
                    "confidence_factors": cluster.confidence_factors,
                },
            )
            canonical_funds.append(cluster.canonical_name)

            # Register all variants as aliases
            for variant in cluster.variants:
                if variant != cluster.canonical_name:
                    consolidator._entity_aliases[normalize_fund_name(variant)] = entity_key

        # Step 7: Build name mapping for downstream phases
        name_to_canonical = {}
        for cluster in valid_clusters:
            for variant in cluster.variants:
                name_to_canonical[variant] = cluster.canonical_name

        # Store for planning phase
        self.context.state.fund_name_variants = name_to_canonical

        # Step 8: Reconciliation check
        report = consolidator.reconcile()
        issues = report.issues + report.warnings

        # Log results
        duplicates_merged = sum(len(c.variants) - 1 for c in valid_clusters)

        self.logger.phase_result(
            self.name,
            f"{len(canonical_funds)} canonical funds",
            mentions=len(mentions),
            clusters=len(clusters),
            merged=duplicates_merged,
            filtered=filtered_count,
        )

        for i, cluster in enumerate(valid_clusters[:10]):
            variants_str = f" (variants: {cluster.variants[:3]})" if len(cluster.variants) > 1 else ""
            self.log(f"  Fund {i+1}: {cluster.canonical_name} (conf: {cluster.confidence:.2f}){variants_str}")

        if len(valid_clusters) > 10:
            self.log(f"  ... and {len(valid_clusters) - 10} more funds")

        return EntityResolutionResult(
            canonical_funds=canonical_funds,
            total_mentions=len(mentions),
            clusters_formed=len(clusters),
            duplicates_merged=duplicates_merged,
            low_confidence_filtered=filtered_count,
            reconciliation_issues=issues,
        )

    def _extract_all_mentions(
        self, exploration_notes: list[ExplorationNotes]
    ) -> list[ResolvedFundMention]:
        """Extract all fund mentions from exploration notes."""
        mentions = []

        for notes in exploration_notes:
            chunk = (notes.page_start, notes.page_end)

            # Mentions from funds_mentioned
            for fund in notes.funds_mentioned:
                mentions.append(ResolvedFundMention(
                    name=fund.name,
                    page=fund.page,
                    has_dedicated_section=fund.has_dedicated_section,
                    from_toc=False,
                    from_table=False,
                    explorer_chunk=chunk,
                ))

            # Mentions from TOC if available
            for toc_entry in getattr(notes, 'toc_entries', []) or []:
                if toc_entry.get('is_fund', False):
                    mentions.append(ResolvedFundMention(
                        name=toc_entry.get('title', ''),
                        page=toc_entry.get('page', notes.page_start),
                        has_dedicated_section=True,  # TOC implies dedicated section
                        from_toc=True,
                        from_table=False,
                        explorer_chunk=chunk,
                    ))

            # Mentions from tables (fund name column)
            for table in notes.tables:
                if table.has_fund_name_column:
                    # We don't have individual fund names from tables here,
                    # but we note that table-based extraction is available
                    pass

        return mentions

    def _build_clusters_from_mapping(
        self,
        mentions: list[ResolvedFundMention],
        name_mapping,
    ) -> list[FundCluster]:
        """Build clusters from name mapping.

        Args:
            mentions: All fund mentions with metadata
            name_mapping: FundNameMapping from TOC or LLM deduplication

        Returns:
            List of FundCluster objects
        """
        # Build clusters indexed by canonical name
        clusters_by_canonical: dict[str, FundCluster] = {}

        # First, ensure all canonical names from the mapping have clusters
        # This is important for TOC-first: we want all TOC funds even if
        # exploration didn't find them
        for canonical in name_mapping.canonical_names:
            clusters_by_canonical[canonical] = FundCluster(
                canonical_name=canonical,
                variants=[canonical],  # Include canonical as a variant
                mentions=[],
            )

        # Then, add mentions to their clusters
        for mention in mentions:
            # Get canonical name for this mention
            canonical = name_mapping.name_to_canonical.get(mention.name)

            if canonical and canonical in clusters_by_canonical:
                cluster = clusters_by_canonical[canonical]
                cluster.mentions.append(mention)
                if mention.name not in cluster.variants:
                    cluster.variants.append(mention.name)
            # If mention doesn't map to any canonical, it's an unmatched mention
            # We ignore it - exploration found something not in our authoritative source

        # For TOC-first, don't override canonical name based on mentions
        # The TOC name IS the canonical name
        # Only refine if not using TOC (fallback path)
        skeleton = self.context.skeleton
        toc_fund_names = skeleton.get_toc_fund_names() if skeleton else []

        if not toc_fund_names:
            # Fallback path: refine canonical names based on mention metadata
            for cluster in clusters_by_canonical.values():
                cluster.canonical_name = self._select_canonical_name(cluster)

        return list(clusters_by_canonical.values())

    def _select_canonical_name(self, cluster: FundCluster) -> str:
        """Select the best canonical name for a cluster."""
        # Prefer name from TOC
        for mention in cluster.mentions:
            if mention.from_toc:
                return mention.name

        # Prefer name with dedicated section
        for mention in cluster.mentions:
            if mention.has_dedicated_section:
                return mention.name

        # Fall back to shortest name (usually the "clean" version)
        return min(cluster.variants, key=len)

    def _map_mentions_to_toc(
        self,
        mentions: list[ResolvedFundMention],
        toc_fund_names: list[str],
        skeleton: DocumentSkeleton,
    ) -> FundNameMapping:
        """Map exploration mentions to TOC fund names.

        TOC is the authoritative source. Each mention is matched to the
        best-matching TOC fund name using fuzzy matching.

        Args:
            mentions: All fund mentions from exploration
            toc_fund_names: Canonical fund names from TOC
            skeleton: Document skeleton with page ranges

        Returns:
            FundNameMapping with TOC names as canonical
        """
        name_to_canonical: dict[str, str] = {}
        merge_groups: dict[str, list[str]] = {}
        unmatched_mentions: list[str] = []

        # For each unique mention name, find the best TOC match
        unique_mention_names = list(set(m.name for m in mentions))

        for mention_name in unique_mention_names:
            best_match = self._find_best_toc_match(mention_name, toc_fund_names)

            if best_match:
                name_to_canonical[mention_name] = best_match
                if best_match not in merge_groups:
                    merge_groups[best_match] = [best_match]
                if mention_name != best_match and mention_name not in merge_groups[best_match]:
                    merge_groups[best_match].append(mention_name)
            else:
                unmatched_mentions.append(mention_name)

        # Log unmatched mentions (exploration found something not in TOC)
        if unmatched_mentions:
            self.log(f"Unmatched exploration mentions (not in TOC): {unmatched_mentions[:5]}", "debug")

        # Calculate duplicates merged
        duplicates_found = sum(
            len(variants) - 1 for variants in merge_groups.values() if len(variants) > 1
        )

        return FundNameMapping(
            canonical_names=list(toc_fund_names),  # TOC order preserved
            name_to_canonical=name_to_canonical,
            duplicates_found=duplicates_found,
            merge_groups={k: v for k, v in merge_groups.items() if len(v) > 1},
        )

    def _find_best_toc_match(
        self,
        mention_name: str,
        toc_fund_names: list[str],
    ) -> str | None:
        """Find the best matching TOC fund name for a mention.

        Uses suffix matching and fuzzy matching to handle:
        - "JPMorgan Investment Funds - Global Macro Fund" → "Global Macro Fund"
        - "GLOBAL MACRO FUND" → "Global Macro Fund"
        """
        norm_mention = normalize_fund_name(mention_name)

        # First pass: exact or suffix match (most reliable)
        for toc_name in toc_fund_names:
            norm_toc = normalize_fund_name(toc_name)

            # Exact match after normalization
            if norm_mention == norm_toc:
                return toc_name

            # Suffix match (handles umbrella prefix)
            if norm_mention.endswith(norm_toc) or norm_toc.endswith(norm_mention):
                return toc_name

        # Second pass: fuzzy match with high threshold
        from difflib import SequenceMatcher

        best_match = None
        best_score = 0.0

        for toc_name in toc_fund_names:
            norm_toc = normalize_fund_name(toc_name)
            score = SequenceMatcher(None, norm_mention, norm_toc).ratio()

            if score > best_score and score >= 0.85:
                best_score = score
                best_match = toc_name

        return best_match

    def _score_cluster_confidence(self, cluster: FundCluster) -> None:
        """Score confidence for a cluster based on evidence.

        TOC-first: Funds from TOC start with high confidence (0.7) since
        they come from the document's own authoritative source.
        """
        # Check if we're using TOC-first approach
        skeleton = self.context.skeleton
        toc_fund_names = skeleton.get_toc_fund_names() if skeleton else []
        is_from_toc = cluster.canonical_name in toc_fund_names

        if is_from_toc:
            # TOC-sourced funds start with high base confidence
            confidence = 0.7
            factors = ["from_toc"]
        else:
            # Non-TOC funds start with lower base confidence
            confidence = 0.3
            factors = []

        # From exploration TOC mention: +0.1 (confirms TOC presence)
        if any(m.from_toc for m in cluster.mentions):
            confidence += 0.1
            factors.append("explorer_found_in_toc")

        # Has dedicated section: +0.15
        if any(m.has_dedicated_section for m in cluster.mentions):
            confidence += 0.15
            factors.append("has_dedicated_section")

        # Multiple mentions: +0.05 per additional mention (max +0.1)
        extra_mentions = min(cluster.total_mentions - 1, 2)
        if extra_mentions > 0:
            confidence += extra_mentions * 0.05
            factors.append(f"mentioned_{cluster.total_mentions}_times")

        # Multiple explorers found it: +0.05
        unique_chunks = set(m.explorer_chunk for m in cluster.mentions if m.explorer_chunk)
        if len(unique_chunks) > 1:
            confidence += 0.05
            factors.append(f"found_by_{len(unique_chunks)}_explorers")

        # Penalize non-TOC funds that look like umbrella names
        if not is_from_toc:
            name_lower = cluster.canonical_name.lower()
            if len(cluster.canonical_name) < 20 and "fund" not in name_lower:
                confidence -= 0.2
                factors.append("possibly_umbrella_name")

            if cluster.canonical_name.replace(" ", "").isalnum() and len(cluster.canonical_name) < 10:
                confidence -= 0.3
                factors.append("looks_like_code")

        cluster.confidence = max(0.0, min(1.0, confidence))
        cluster.confidence_factors = factors
