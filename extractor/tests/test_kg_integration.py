"""Integration tests for GraphStore in the extraction pipeline.

Tests end-to-end flows:
- Exploration populates graph -> Extraction queries graph
- Table relations guide extraction to correct pages
- Cross-references are included in hints
- Empty graph returns sensible defaults
"""

import sys
from pathlib import Path
from types import ModuleType

# Set up v4_extractor_KGRAG as the extractor package before other imports
V4_ROOT = Path(__file__).parent.parent
if str(V4_ROOT) not in sys.path:
    sys.path.insert(0, str(V4_ROOT))

# Create extractor package alias pointing to v4 modules
if "extractor" not in sys.modules or sys.modules["extractor"].__path__ != [str(V4_ROOT)]:
    _extractor_pkg = ModuleType("extractor")
    _extractor_pkg.__path__ = [str(V4_ROOT)]
    sys.modules["extractor"] = _extractor_pkg

    # Import pydantic_models and its submodules, register under extractor
    import pydantic_models
    import pydantic_models.graph_models
    sys.modules["extractor.pydantic_models"] = pydantic_models
    sys.modules["extractor.pydantic_models.graph_models"] = pydantic_models.graph_models

import pytest

from pydantic_models.graph_models import (
    EntityType, RelationType, ExtractionHint, FundContext,
)

# Import graph_store directly from file to avoid triggering core/__init__.py
import importlib.util

def _load_module(name, path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

# Load fund_names first (dependency)
_load_module("extractor.core.fund_names", V4_ROOT / "core" / "fund_names.py")

# Load graph_store
_graph_store_mod = _load_module("extractor.core.graph_store", V4_ROOT / "core" / "graph_store.py")
GraphStore = _graph_store_mod.GraphStore


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def exploration_populated_graph():
    """Simulate a graph after exploration phase has run."""
    store = GraphStore()

    # Exploration discovers multiple funds
    store.record_fund("Global Equity Fund", [10, 11, 12, 13], True, "exploration", 10)
    store.record_fund("Global Bond Fund", [20, 21, 22], True, "exploration", 20)
    store.record_fund("Money Market Fund", [30, 31], True, "exploration", 30)

    # Exploration finds consolidated ISIN table
    store.record_table(
        table_type="isin",
        pages=[100, 101, 102],
        columns=["Fund Name", "Share Class", "ISIN", "Currency"],
        lookup_column="Fund Name",
        is_consolidated=True,
        source_phase="exploration",
    )

    # Exploration finds consolidated fee table
    store.record_table(
        table_type="fee",
        pages=[150, 151],
        columns=["Sub-Fund", "Management Fee", "Performance Fee", "Entry Fee"],
        lookup_column="Sub-Fund",
        is_consolidated=True,
        source_phase="exploration",
    )

    # Exploration records a cross-reference to external doc
    store.add_relation(
        relation_type=RelationType.REFERENCES,
        subject_key="fund:Global Equity Fund",
        object_key="external:kiid",
        properties={
            "field_hint": "risk_profile",
            "text": "For risk profile details, see the KIID",
            "target_pages": [],
            "is_external": True,
            "external_doc": "KIID",
        },
        source_phase="exploration",
    )

    return store


@pytest.fixture
def partially_extracted_graph(exploration_populated_graph):
    """Graph after some extraction has occurred."""
    store = exploration_populated_graph

    store._add_fact(
        entity_key="fund:Global Equity Fund",
        field_name="investment_objective",
        value="Long-term capital growth through equity investments",
        source_page=10,
        source_quote="The fund aims to achieve long-term capital growth...",
        source_type="text",
        extraction_phase="extraction",
        confidence=0.9,
    )

    store._add_fact(
        entity_key="fund:Global Equity Fund",
        field_name="management_fee",
        value="1.50%",
        source_page=150,
        source_quote="Global Equity Fund | 1.50%",
        source_type="table",
        extraction_phase="extraction",
        confidence=0.95,
    )

    return store


# =============================================================================
# Exploration -> Extraction Flow Tests
# =============================================================================


class TestExplorationPopulatesGraph:
    """Test that exploration data is properly stored and queryable."""

    def test_funds_are_discoverable(self, exploration_populated_graph):
        """All funds recorded during exploration can be retrieved."""
        funds = exploration_populated_graph.get_entities_by_type(EntityType.FUND)

        assert len(funds) == 3
        fund_names = {f.name for f in funds}
        assert fund_names == {"Global Equity Fund", "Global Bond Fund", "Money Market Fund"}

    def test_tables_are_discoverable(self, exploration_populated_graph):
        """All tables recorded during exploration can be retrieved."""
        tables = exploration_populated_graph.get_entities_by_type(EntityType.TABLE)

        assert len(tables) == 2
        table_types = {t.properties.get("table_type") for t in tables}
        assert table_types == {"isin", "fee"}

    def test_fund_table_relations_exist(self, exploration_populated_graph):
        """Consolidated tables are linked to all funds."""
        isin_relations = exploration_populated_graph.query_relations(
            relation_type=RelationType.HAS_ISIN_IN
        )
        fee_relations = exploration_populated_graph.query_relations(
            relation_type=RelationType.HAS_FEE_IN
        )

        assert len(isin_relations) == 3
        assert len(fee_relations) == 3


class TestTableRelationsGuideExtraction:
    """Test that table relations provide correct extraction hints."""

    def test_isin_extraction_points_to_isin_table(self, exploration_populated_graph):
        """ISIN field extraction gets hint pointing to ISIN table."""
        hint = exploration_populated_graph.get_extraction_context("Global Equity Fund", "isin")

        assert hint.table_hint is not None
        assert hint.table_hint.pages == [100, 101, 102]
        assert hint.table_hint.lookup_column == "Fund Name"
        assert hint.table_hint.lookup_value == "Global Equity Fund"

    def test_fee_extraction_points_to_fee_table(self, exploration_populated_graph):
        """Fee field extraction gets hint pointing to fee table."""
        hint = exploration_populated_graph.get_extraction_context("Global Bond Fund", "management_fee")

        assert hint.table_hint is not None
        assert hint.table_hint.pages == [150, 151]
        assert hint.table_hint.lookup_column == "Sub-Fund"

    def test_text_field_points_to_section_pages(self, exploration_populated_graph):
        """Non-table fields point to fund's section pages."""
        hint = exploration_populated_graph.get_extraction_context("Money Market Fund", "investment_objective")

        assert hint.table_hint is None
        assert hint.pages == [30, 31]
        assert hint.primary_source == "text"

    def test_different_funds_get_correct_lookup_values(self, exploration_populated_graph):
        """Each fund gets its own name as lookup value."""
        hint1 = exploration_populated_graph.get_extraction_context("Global Equity Fund", "isin")
        hint2 = exploration_populated_graph.get_extraction_context("Global Bond Fund", "isin")
        hint3 = exploration_populated_graph.get_extraction_context("Money Market Fund", "isin")

        assert hint1.table_hint.lookup_value == "Global Equity Fund"
        assert hint2.table_hint.lookup_value == "Global Bond Fund"
        assert hint3.table_hint.lookup_value == "Money Market Fund"


class TestCrossReferencesInHints:
    """Test that cross-references are properly included in extraction hints."""

    def test_cross_ref_included_for_matching_field(self, exploration_populated_graph):
        """Cross-reference is included when field_hint matches."""
        hint = exploration_populated_graph.get_extraction_context("Global Equity Fund", "risk_profile")

        assert len(hint.cross_refs) == 1
        assert hint.cross_refs[0].is_external is True
        assert hint.cross_refs[0].external_doc == "KIID"

    def test_cross_ref_not_included_for_other_fields(self, exploration_populated_graph):
        """Cross-reference is not included for unrelated fields."""
        hint = exploration_populated_graph.get_extraction_context("Global Equity Fund", "investment_objective")

        risk_cross_refs = [cr for cr in hint.cross_refs if cr.external_doc == "KIID"]
        assert len(risk_cross_refs) == 0

    def test_cross_refs_in_fund_context(self, exploration_populated_graph):
        """Fund context includes all cross-references for the fund."""
        context = exploration_populated_graph.get_fund_context("Global Equity Fund")

        assert len(context.cross_refs) == 1


class TestExistingFactsInContext:
    """Test that previously extracted facts are included in context."""

    def test_existing_facts_appear_in_extraction_context(self, partially_extracted_graph):
        """Previously extracted facts are included in hints."""
        hint = partially_extracted_graph.get_extraction_context("Global Equity Fund", "management_fee")

        assert len(hint.existing_facts) == 1
        assert hint.existing_facts[0].value == "1.50%"

    def test_existing_facts_appear_in_fund_context(self, partially_extracted_graph):
        """Fund context includes all existing facts."""
        context = partially_extracted_graph.get_fund_context("Global Equity Fund")

        assert len(context.existing_facts) == 2
        field_names = {f.field_name for f in context.existing_facts}
        assert field_names == {"investment_objective", "management_fee"}


# =============================================================================
# Empty Graph Behavior Tests
# =============================================================================


class TestEmptyGraphDefaults:
    """Test that empty graph returns sensible defaults."""

    def test_extraction_context_for_unknown_fund(self):
        """Extraction context for unknown fund returns empty but valid hint."""
        store = GraphStore()

        hint = store.get_extraction_context("Unknown Fund", "isin")

        assert type(hint).__name__ == "ExtractionHint"
        assert hint.pages == []
        assert hint.table_hint is None
        assert hint.confidence == 0.5

    def test_fund_context_for_unknown_fund(self):
        """Fund context for unknown fund returns None."""
        store = GraphStore()

        context = store.get_fund_context("Unknown Fund")

        assert context is None

    def test_query_relations_empty_graph(self):
        """Querying relations on empty graph returns empty list."""
        store = GraphStore()

        relations = store.query_relations(relation_type=RelationType.HAS_ISIN_IN)

        assert relations == []

    def test_get_facts_empty_graph(self):
        """Getting facts for unknown entity returns empty list."""
        store = GraphStore()

        facts = store.get_facts_for_entity("unknown_entity")

        assert facts == []


# =============================================================================
# Multi-Phase Extraction Flow Tests
# =============================================================================


class TestMultiPhaseExtraction:
    """Test realistic multi-phase extraction scenarios."""

    def test_exploration_then_extraction_flow(self):
        """Full flow: exploration populates -> extraction queries."""
        store = GraphStore()

        # Phase 1: Exploration
        store.record_fund("Test Fund", [5, 6, 7], True, "exploration")
        store.record_table(
            table_type="isin",
            pages=[50],
            columns=["Fund", "ISIN"],
            lookup_column="Fund",
            is_consolidated=True,
            source_phase="exploration",
        )

        # Phase 2: Extraction queries the graph
        context = store.get_fund_context("Test Fund")
        assert context is not None
        assert context.isin_table is not None
        assert context.isin_table.pages == [50]

        # Extraction records facts
        store._add_fact(
            entity_key="fund:Test Fund",
            field_name="isin",
            value="LU0123456789",
            source_page=50,
            source_quote="Test Fund | LU0123456789",
            source_type="table",
            extraction_phase="extraction",
        )

        # Later queries include the fact
        hint = store.get_extraction_context("Test Fund", "isin")
        assert len(hint.existing_facts) == 1

    def test_incremental_table_discovery(self):
        """Tables discovered after funds are properly linked."""
        store = GraphStore()

        # First, funds are discovered
        store.record_fund("Fund A", [10], True, "exploration")
        store.record_fund("Fund B", [20], True, "exploration")

        # Initially no tables
        context = store.get_fund_context("Fund A")
        assert context.isin_table is None

        # Later, table is discovered
        store.record_table(
            table_type="isin",
            pages=[100],
            columns=["Fund Name", "ISIN"],
            lookup_column="Fund Name",
            is_consolidated=True,
            source_phase="exploration",
        )

        # Now both funds have table access
        context_a = store.get_fund_context("Fund A")
        context_b = store.get_fund_context("Fund B")

        assert context_a.isin_table is not None
        assert context_b.isin_table is not None

    def test_fund_discovered_after_table(self):
        """Funds discovered after consolidated tables need manual linking."""
        store = GraphStore()

        # Table discovered first
        store.record_table(
            table_type="isin",
            pages=[100],
            columns=["Fund Name", "ISIN"],
            lookup_column="Fund Name",
            is_consolidated=True,
            source_phase="exploration",
        )

        # Fund discovered later
        store.record_fund("Late Fund", [50], True, "exploration")

        # Fund was not linked to existing table by record_fund
        context = store.get_fund_context("Late Fund")
        assert context.isin_table is None

        # Manual linking resolves this
        store.link_funds_to_table("isin_100", "isin", "exploration")

        context = store.get_fund_context("Late Fund")
        assert context.isin_table is not None


# =============================================================================
# Confidence and Source Tracking Tests
# =============================================================================


class TestConfidenceTracking:
    """Test confidence score handling through the pipeline."""

    def test_table_extraction_has_higher_confidence(self, exploration_populated_graph):
        """Table-based extraction hints have higher confidence."""
        table_hint = exploration_populated_graph.get_extraction_context("Global Equity Fund", "isin")
        text_hint = exploration_populated_graph.get_extraction_context("Global Equity Fund", "investment_objective")

        assert table_hint.confidence > text_hint.confidence

    def test_existing_facts_boost_confidence(self, partially_extracted_graph):
        """Having existing facts maintains good confidence."""
        hint = partially_extracted_graph.get_extraction_context("Global Equity Fund", "management_fee")

        # Has table hint = 0.9 confidence
        assert hint.confidence == 0.9

    def test_unknown_entity_has_low_confidence(self):
        """Unknown entities have low extraction confidence."""
        store = GraphStore()

        hint = store.get_extraction_context("Unknown", "isin")

        assert hint.confidence == 0.5


class TestSourceTracking:
    """Test source information preservation."""

    def test_fact_source_preserved(self, partially_extracted_graph):
        """Fact source information is preserved through queries."""
        facts = partially_extracted_graph.get_facts_for_entity(
            "Global Equity Fund", "management_fee"
        )

        assert len(facts) == 1
        fact = facts[0]
        assert fact.source_page == 150
        assert fact.source_type == "table"
        assert fact.extraction_phase == "extraction"

    def test_primary_source_indicates_table_vs_text(self, exploration_populated_graph):
        """Extraction hint correctly indicates primary source type."""
        isin_hint = exploration_populated_graph.get_extraction_context("Global Equity Fund", "isin")
        text_hint = exploration_populated_graph.get_extraction_context("Global Equity Fund", "investment_objective")

        assert isin_hint.primary_source == "table"
        assert text_hint.primary_source == "text"


# =============================================================================
# Edge Cases and Error Handling Tests
# =============================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_fund_with_no_section_pages(self):
        """Fund without section pages still works."""
        store = GraphStore()
        store.record_fund("No Pages Fund", None, False, "exploration")

        context = store.get_fund_context("No Pages Fund")

        assert context is not None
        assert context.section_pages == []

    def test_table_with_no_lookup_column_infers_one(self):
        """Table without explicit lookup column infers from headers."""
        store = GraphStore()
        store.record_fund("Test Fund", [10], True, "exploration")

        table = store.record_table(
            table_type="isin",
            pages=[50],
            columns=["Sub-Fund", "ISIN", "Currency"],
            lookup_column=None,
            is_consolidated=True,
            source_phase="exploration",
        )

        assert table.properties["lookup_column"] == "Sub-Fund"

    def test_multiple_tables_of_same_type(self):
        """Multiple tables of same type create unique entities."""
        store = GraphStore()
        store.record_fund("Test Fund", [10], True, "exploration")

        store.record_table(
            table_type="isin", pages=[50], columns=["Fund", "ISIN"],
            lookup_column="Fund", is_consolidated=True, source_phase="exploration",
        )
        store.record_table(
            table_type="isin", pages=[60], columns=["Fund", "ISIN"],
            lookup_column="Fund", is_consolidated=True, source_phase="exploration",
        )

        tables = store.get_entities_by_type(EntityType.TABLE)
        assert len(tables) == 2

    def test_fund_specific_and_consolidated_tables(self):
        """Fund can have both consolidated and fund-specific table relations."""
        store = GraphStore()
        store.record_fund("Test Fund", [10, 11], True, "exploration")

        # Consolidated table
        store.record_table(
            table_type="isin", pages=[100], columns=["Fund", "ISIN"],
            lookup_column="Fund", is_consolidated=True, source_phase="exploration",
        )

        # Fund-specific table
        store.record_table(
            table_type="fee", pages=[11], columns=["Share Class", "Fee"],
            lookup_column="Share Class", is_consolidated=False,
            source_phase="exploration", belongs_to_fund="Test Fund",
        )

        isin_rels = store.query_relations(
            subject="fund:Test Fund",
            relation_type=RelationType.HAS_ISIN_IN
        )
        fee_rels = store.query_relations(
            subject="fund:Test Fund",
            relation_type=RelationType.HAS_FEE_IN
        )

        assert len(isin_rels) == 1
        assert len(fee_rels) == 1
