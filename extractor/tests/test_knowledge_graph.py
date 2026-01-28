"""Tests for the GraphStore class.

Tests cover:
- Entity operations: _add_entity, get_entity, merge duplicates, update confidence
- Relation operations: _add_relation, auto-prefix keys, query_relations
- Fact operations: _add_fact, get_facts_for_entity
- get_extraction_context: returns correct pages, table_hint, lookup_column
- get_fund_context: assembles complete context
- record_table: links consolidated tables to all funds
- record_fund: convenience method
"""

import sys
from pathlib import Path
from types import ModuleType

# Set up v4_extractor_KGRAG as the extractor package before other imports
V4_ROOT = Path(__file__).parent.parent
if str(V4_ROOT) not in sys.path:
    sys.path.insert(0, str(V4_ROOT))

# Create extractor package alias pointing to v4 modules
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
    Entity, EntityType, Relation, RelationType, Fact,
    ExtractionHint, FundContext, TableHint,
)

# Import new modules directly from file to avoid triggering core/__init__.py
import importlib.util

def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

# Load fund_names first (dependency)
_fund_names = _load_module("extractor.core.fund_names", V4_ROOT / "core" / "fund_names.py")
sys.modules["core.fund_names"] = _fund_names

# Load graph_store
_graph_store_mod = _load_module("extractor.core.graph_store", V4_ROOT / "core" / "graph_store.py")
GraphStore = _graph_store_mod.GraphStore


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def empty_graph():
    """Create an empty GraphStore."""
    return GraphStore()


@pytest.fixture
def graph_with_fund():
    """Create a GraphStore with a single fund."""
    store = GraphStore()
    store._add_entity(
        entity_type=EntityType.FUND,
        entity_id="Global Equity Fund",
        name="Global Equity Fund",
        properties={"section_pages": [10, 11, 12], "has_dedicated_section": True},
        source_phase="exploration",
        source_pages=[10],
    )
    return store


@pytest.fixture
def graph_with_fund_and_table():
    """Create a GraphStore with a fund linked to a table."""
    store = GraphStore()
    store._add_entity(
        entity_type=EntityType.FUND,
        entity_id="Global Equity Fund",
        name="Global Equity Fund",
        properties={"section_pages": [10, 11, 12], "has_dedicated_section": True},
        source_phase="exploration",
        source_pages=[10],
    )
    store._add_entity(
        entity_type=EntityType.TABLE,
        entity_id="isin_50",
        name="ISIN Table",
        properties={
            "table_type": "isin",
            "pages": [50, 51],
            "columns": ["Fund Name", "Share Class", "ISIN"],
            "lookup_column": "Fund Name",
            "is_consolidated": True,
        },
        source_phase="exploration",
        source_pages=[50, 51],
    )
    store._add_relation(
        relation_type=RelationType.HAS_ISIN_IN,
        subject_key="fund:Global Equity Fund",
        object_key="table:isin_50",
        source_phase="exploration",
    )
    return store


# =============================================================================
# Entity Operations Tests
# =============================================================================


class TestAddEntity:
    """Tests for _add_entity method."""

    def test_add_new_entity(self, empty_graph):
        """Adding a new entity stores it in the graph."""
        entity = empty_graph._add_entity(
            entity_type=EntityType.FUND,
            entity_id="Test Fund",
            name="Test Fund",
            properties={"section_pages": [1, 2, 3]},
            source_phase="exploration",
            source_pages=[1],
        )

        assert entity.id == "Test Fund"
        assert entity.entity_type == EntityType.FUND
        assert "fund:Test Fund" in empty_graph.entities

    def test_add_entity_with_confidence(self, empty_graph):
        """Entity confidence is stored correctly."""
        entity = empty_graph._add_entity(
            entity_type=EntityType.FUND,
            entity_id="High Confidence Fund",
            name="High Confidence Fund",
            properties={},
            source_phase="exploration",
            confidence=0.95,
        )

        assert entity.confidence == 0.95

    def test_add_entity_updates_type_index(self, empty_graph):
        """Adding entity updates the by-type index."""
        empty_graph._add_entity(
            entity_type=EntityType.FUND,
            entity_id="Fund A",
            name="Fund A",
            properties={},
            source_phase="exploration",
        )
        empty_graph._add_entity(
            entity_type=EntityType.TABLE,
            entity_id="Table A",
            name="Table A",
            properties={},
            source_phase="exploration",
        )

        assert len(empty_graph.get_entities_by_type(EntityType.FUND)) == 1
        assert len(empty_graph.get_entities_by_type(EntityType.TABLE)) == 1


class TestMergeEntities:
    """Tests for entity merging behavior."""

    def test_duplicate_entity_merges_properties(self, empty_graph):
        """Adding an entity with same key merges properties."""
        empty_graph._add_entity(
            entity_type=EntityType.FUND,
            entity_id="Test Fund",
            name="Test Fund",
            properties={"section_pages": [1, 2]},
            source_phase="exploration",
        )

        empty_graph._add_entity(
            entity_type=EntityType.FUND,
            entity_id="Test Fund",
            name="Test Fund",
            properties={"has_dedicated_section": True},
            source_phase="extraction",
        )

        entity = empty_graph.get_entity("fund:Test Fund")
        assert entity.properties["section_pages"] == [1, 2]
        assert entity.properties["has_dedicated_section"] is True

    def test_duplicate_entity_takes_higher_confidence(self, empty_graph):
        """When merging, higher confidence is retained."""
        empty_graph._add_entity(
            entity_type=EntityType.FUND,
            entity_id="Test Fund",
            name="Test Fund",
            properties={},
            source_phase="exploration",
            confidence=0.7,
        )

        empty_graph._add_entity(
            entity_type=EntityType.FUND,
            entity_id="Test Fund",
            name="Test Fund",
            properties={},
            source_phase="extraction",
            confidence=0.9,
        )

        entity = empty_graph.get_entity("fund:Test Fund")
        assert entity.confidence == 0.9

    def test_duplicate_entity_merges_source_pages(self, empty_graph):
        """When merging, source pages are combined and deduplicated."""
        empty_graph._add_entity(
            entity_type=EntityType.FUND,
            entity_id="Test Fund",
            name="Test Fund",
            properties={},
            source_phase="exploration",
            source_pages=[1, 2],
        )

        empty_graph._add_entity(
            entity_type=EntityType.FUND,
            entity_id="Test Fund",
            name="Test Fund",
            properties={},
            source_phase="extraction",
            source_pages=[2, 3],
        )

        entity = empty_graph.get_entity("fund:Test Fund")
        assert set(entity.source_pages) == {1, 2, 3}


class TestGetEntity:
    """Tests for get_entity method."""

    def test_get_entity_by_full_key(self, graph_with_fund):
        """Get entity using full prefixed key."""
        entity = graph_with_fund.get_entity("fund:Global Equity Fund")
        assert entity is not None
        assert entity.name == "Global Equity Fund"

    def test_get_entity_by_partial_key(self, graph_with_fund):
        """Get entity using partial key (auto-prefix search)."""
        entity = graph_with_fund.get_entity("Global Equity Fund")
        assert entity is not None
        assert entity.name == "Global Equity Fund"

    def test_get_nonexistent_entity(self, empty_graph):
        """Getting nonexistent entity returns None."""
        entity = empty_graph.get_entity("fund:Does Not Exist")
        assert entity is None

    def test_get_entity_partial_key_tries_prefixes(self, graph_with_fund_and_table):
        """Partial key search tries fund, table, share_class prefixes."""
        fund = graph_with_fund_and_table.get_entity("Global Equity Fund")
        assert fund is not None
        assert fund.entity_type == EntityType.FUND


# =============================================================================
# Relation Operations Tests
# =============================================================================


class TestAddRelation:
    """Tests for _add_relation method."""

    def test_add_relation(self, graph_with_fund):
        """Adding a relation stores it in the graph."""
        graph_with_fund._add_entity(
            entity_type=EntityType.TABLE,
            entity_id="fee_table",
            name="Fee Table",
            properties={},
            source_phase="exploration",
        )

        relation = graph_with_fund._add_relation(
            relation_type=RelationType.HAS_FEE_IN,
            subject_key="fund:Global Equity Fund",
            object_key="table:fee_table",
            source_phase="exploration",
        )

        assert relation.relation_type == RelationType.HAS_FEE_IN
        assert relation in graph_with_fund.relations

    def test_add_relation_auto_prefix_subject(self, graph_with_fund):
        """Subject key without prefix gets 'fund:' added."""
        relation = graph_with_fund._add_relation(
            relation_type=RelationType.HAS_FEE_IN,
            subject_key="Global Equity Fund",
            object_key="table:some_table",
            source_phase="exploration",
        )

        assert relation.subject_key == "fund:Global Equity Fund"

    def test_add_relation_auto_prefix_object(self, graph_with_fund):
        """Object key without prefix gets 'table:' added."""
        relation = graph_with_fund._add_relation(
            relation_type=RelationType.HAS_ISIN_IN,
            subject_key="fund:Global Equity Fund",
            object_key="isin_table",
            source_phase="exploration",
        )

        assert relation.object_key == "table:isin_table"

    def test_add_relation_updates_indexes(self, graph_with_fund):
        """Adding relation updates subject/object/type indexes."""
        relation = graph_with_fund._add_relation(
            relation_type=RelationType.HAS_ISIN_IN,
            subject_key="fund:Global Equity Fund",
            object_key="table:isin_table",
            source_phase="exploration",
        )

        assert relation in graph_with_fund._relations_by_subject["fund:Global Equity Fund"]
        assert relation in graph_with_fund._relations_by_object["table:isin_table"]
        assert relation in graph_with_fund._relations_by_type[RelationType.HAS_ISIN_IN]


class TestQueryRelations:
    """Tests for query_relations method."""

    def test_query_by_subject(self, graph_with_fund_and_table):
        """Query relations by subject key."""
        relations = graph_with_fund_and_table.query_relations(
            subject="fund:Global Equity Fund"
        )

        assert len(relations) == 1
        assert relations[0].relation_type == RelationType.HAS_ISIN_IN

    def test_query_by_relation_type(self, graph_with_fund_and_table):
        """Query relations by relation type."""
        relations = graph_with_fund_and_table.query_relations(
            relation_type=RelationType.HAS_ISIN_IN
        )

        assert len(relations) == 1

    def test_query_by_object(self, graph_with_fund_and_table):
        """Query relations by object key."""
        relations = graph_with_fund_and_table.query_relations(
            object_key="table:isin_50"
        )

        assert len(relations) == 1

    def test_query_with_multiple_filters(self, graph_with_fund_and_table):
        """Query with multiple filters applies all of them."""
        relations = graph_with_fund_and_table.query_relations(
            subject="fund:Global Equity Fund",
            relation_type=RelationType.HAS_ISIN_IN,
        )

        assert len(relations) == 1

    def test_query_no_matches(self, graph_with_fund_and_table):
        """Query with no matches returns empty list."""
        relations = graph_with_fund_and_table.query_relations(
            relation_type=RelationType.HAS_FEE_IN
        )

        assert relations == []

    def test_query_all_relations(self, graph_with_fund_and_table):
        """Query with no filters returns all relations."""
        relations = graph_with_fund_and_table.query_relations()

        assert len(relations) == 1


# =============================================================================
# Fact Operations Tests
# =============================================================================


class TestAddFact:
    """Tests for _add_fact method."""

    def test_add_fact(self, graph_with_fund):
        """Adding a fact stores it in the graph."""
        fact = graph_with_fund._add_fact(
            entity_key="fund:Global Equity Fund",
            field_name="management_fee",
            value="1.50%",
            source_page=50,
            source_quote="Management Fee: 1.50%",
            extraction_phase="extraction",
        )

        assert fact.field_name == "management_fee"
        assert fact.value == "1.50%"
        assert fact in graph_with_fund.facts

    def test_add_fact_auto_prefix(self, graph_with_fund):
        """Entity key without prefix gets 'fund:' added."""
        fact = graph_with_fund._add_fact(
            entity_key="Global Equity Fund",
            field_name="isin",
            value="LU0123456789",
            source_page=11,
            extraction_phase="extraction",
        )

        assert fact.entity_key == "fund:Global Equity Fund"

    def test_add_fact_updates_indexes(self, graph_with_fund):
        """Adding fact updates entity and field indexes."""
        fact = graph_with_fund._add_fact(
            entity_key="fund:Global Equity Fund",
            field_name="management_fee",
            value="1.50%",
            source_page=50,
            extraction_phase="extraction",
        )

        assert fact in graph_with_fund._facts_by_entity["fund:Global Equity Fund"]
        assert fact in graph_with_fund._facts_by_field["management_fee"]


class TestGetFactsForEntity:
    """Tests for get_facts_for_entity method."""

    def test_get_all_facts(self, graph_with_fund):
        """Get all facts for an entity."""
        graph_with_fund._add_fact(
            entity_key="fund:Global Equity Fund",
            field_name="management_fee",
            value="1.50%",
            source_page=50,
            extraction_phase="extraction",
        )
        graph_with_fund._add_fact(
            entity_key="fund:Global Equity Fund",
            field_name="isin",
            value="LU0123456789",
            source_page=11,
            extraction_phase="extraction",
        )

        facts = graph_with_fund.get_facts_for_entity("fund:Global Equity Fund")
        assert len(facts) == 2

    def test_get_facts_by_field_name(self, graph_with_fund):
        """Filter facts by field name."""
        graph_with_fund._add_fact(
            entity_key="fund:Global Equity Fund",
            field_name="management_fee",
            value="1.50%",
            source_page=50,
            extraction_phase="extraction",
        )
        graph_with_fund._add_fact(
            entity_key="fund:Global Equity Fund",
            field_name="isin",
            value="LU0123456789",
            source_page=11,
            extraction_phase="extraction",
        )

        facts = graph_with_fund.get_facts_for_entity("fund:Global Equity Fund", "isin")
        assert len(facts) == 1
        assert facts[0].value == "LU0123456789"

    def test_get_facts_auto_prefix(self, graph_with_fund):
        """Entity key without prefix is auto-prefixed."""
        graph_with_fund._add_fact(
            entity_key="fund:Global Equity Fund",
            field_name="isin",
            value="LU0123456789",
            source_page=11,
            extraction_phase="extraction",
        )

        facts = graph_with_fund.get_facts_for_entity("Global Equity Fund")
        assert len(facts) == 1


# =============================================================================
# get_extraction_context Tests
# =============================================================================


class TestGetExtractionContext:
    """Tests for get_extraction_context method."""

    def test_returns_table_hint_for_isin(self, graph_with_fund_and_table):
        """ISIN field returns table hint when table relation exists."""
        hint = graph_with_fund_and_table.get_extraction_context("Global Equity Fund", "isin")

        assert hint.table_hint is not None
        assert hint.table_hint.pages == [50, 51]
        assert hint.table_hint.lookup_column == "Fund Name"
        assert hint.table_hint.lookup_value == "Global Equity Fund"
        assert hint.primary_source == "table"

    def test_returns_table_hint_for_fees(self):
        """Fee field returns table hint when fee table relation exists."""
        store = GraphStore()
        store._add_entity(
            entity_type=EntityType.FUND,
            entity_id="Test Fund",
            name="Test Fund",
            properties={"section_pages": [10]},
            source_phase="exploration",
        )
        store._add_entity(
            entity_type=EntityType.TABLE,
            entity_id="fee_60",
            name="Fee Table",
            properties={
                "table_type": "fee",
                "pages": [60, 61],
                "columns": ["Fund", "Management Fee", "Performance Fee"],
                "lookup_column": "Fund",
                "is_consolidated": True,
            },
            source_phase="exploration",
        )
        store._add_relation(
            relation_type=RelationType.HAS_FEE_IN,
            subject_key="fund:Test Fund",
            object_key="table:fee_60",
            source_phase="exploration",
        )

        hint = store.get_extraction_context("Test Fund", "management_fee")

        assert hint.table_hint is not None
        assert hint.table_hint.pages == [60, 61]

    def test_returns_section_pages_when_no_table(self, graph_with_fund):
        """When no table, returns section pages from entity."""
        hint = graph_with_fund.get_extraction_context("Global Equity Fund", "investment_objective")

        assert hint.pages == [10, 11, 12]
        assert hint.table_hint is None
        assert hint.primary_source == "text"

    def test_includes_existing_facts(self, graph_with_fund):
        """Context includes previously extracted facts."""
        graph_with_fund._add_fact(
            entity_key="fund:Global Equity Fund",
            field_name="isin",
            value="LU0123456789",
            source_page=11,
            extraction_phase="exploration",
        )

        hint = graph_with_fund.get_extraction_context("Global Equity Fund", "isin")

        assert len(hint.existing_facts) == 1
        assert hint.existing_facts[0].value == "LU0123456789"

    def test_includes_cross_references(self):
        """Context includes cross-references that hint at the field."""
        store = GraphStore()
        store._add_entity(
            entity_type=EntityType.FUND,
            entity_id="Test Fund",
            name="Test Fund",
            properties={"section_pages": [10]},
            source_phase="exploration",
        )
        store._add_relation(
            relation_type=RelationType.REFERENCES,
            subject_key="fund:Test Fund",
            object_key="external:kiid",
            properties={
                "field_hint": "isin",
                "text": "For ISIN codes, see the KIID",
                "target_pages": [],
                "is_external": True,
                "external_doc": "KIID",
            },
            source_phase="exploration",
        )

        hint = store.get_extraction_context("Test Fund", "isin")

        assert len(hint.cross_refs) == 1
        assert hint.cross_refs[0].is_external is True

    def test_confidence_higher_with_table(self, graph_with_fund_and_table):
        """Confidence is higher when table hint is available."""
        hint = graph_with_fund_and_table.get_extraction_context("Global Equity Fund", "isin")

        assert hint.confidence == 0.9

    def test_confidence_lower_without_pages(self, empty_graph):
        """Confidence is lower when no pages are found."""
        hint = empty_graph.get_extraction_context("Unknown Fund", "isin")

        assert hint.confidence == 0.5
        assert hint.pages == []


# =============================================================================
# get_fund_context Tests
# =============================================================================


class TestGetFundContext:
    """Tests for get_fund_context method."""

    def test_returns_none_for_unknown_fund(self, empty_graph):
        """Unknown fund returns None."""
        context = empty_graph.get_fund_context("Unknown Fund")
        assert context is None

    def test_returns_entity(self, graph_with_fund):
        """Context includes the fund entity."""
        context = graph_with_fund.get_fund_context("Global Equity Fund")

        assert context is not None
        assert context.entity.name == "Global Equity Fund"

    def test_returns_section_pages(self, graph_with_fund):
        """Context includes section pages."""
        context = graph_with_fund.get_fund_context("Global Equity Fund")

        assert context.section_pages == [10, 11, 12]

    def test_includes_isin_table(self, graph_with_fund_and_table):
        """Context includes ISIN table hint when available."""
        context = graph_with_fund_and_table.get_fund_context("Global Equity Fund")

        assert context.isin_table is not None
        assert context.isin_table.table_id == "isin_50"

    def test_includes_fee_table(self):
        """Context includes fee table hint when available."""
        store = GraphStore()
        store._add_entity(
            entity_type=EntityType.FUND,
            entity_id="Test Fund",
            name="Test Fund",
            properties={"section_pages": [10]},
            source_phase="exploration",
        )
        store._add_entity(
            entity_type=EntityType.TABLE,
            entity_id="fee_60",
            name="Fee Table",
            properties={
                "pages": [60],
                "columns": ["Fund", "Fee"],
                "lookup_column": "Fund",
                "is_consolidated": True,
            },
            source_phase="exploration",
        )
        store._add_relation(
            relation_type=RelationType.HAS_FEE_IN,
            subject_key="fund:Test Fund",
            object_key="table:fee_60",
            source_phase="exploration",
        )

        context = store.get_fund_context("Test Fund")

        assert context.fee_table is not None

    def test_includes_share_classes(self):
        """Context includes share class entities."""
        store = GraphStore()
        store._add_entity(
            entity_type=EntityType.FUND,
            entity_id="Test Fund",
            name="Test Fund",
            properties={"section_pages": [10]},
            source_phase="exploration",
        )
        store._add_entity(
            entity_type=EntityType.SHARE_CLASS,
            entity_id="A (acc) USD",
            name="A (acc) USD",
            properties={"isin": "LU0123456789"},
            source_phase="exploration",
        )
        store._add_relation(
            relation_type=RelationType.HAS_SHARE_CLASS,
            subject_key="fund:Test Fund",
            object_key="share_class:A (acc) USD",
            source_phase="exploration",
        )

        context = store.get_fund_context("Test Fund")

        assert len(context.share_classes) == 1
        assert context.share_classes[0].name == "A (acc) USD"

    def test_includes_existing_facts(self, graph_with_fund):
        """Context includes all facts for the fund."""
        graph_with_fund._add_fact(
            entity_key="fund:Global Equity Fund",
            field_name="management_fee",
            value="1.50%",
            source_page=50,
            extraction_phase="extraction",
        )

        context = graph_with_fund.get_fund_context("Global Equity Fund")

        assert len(context.existing_facts) == 1


# =============================================================================
# record_table Tests
# =============================================================================


class TestRecordTable:
    """Tests for store.record_table."""

    def test_creates_table_entity(self, empty_graph):
        """record_table creates a table entity."""
        table = empty_graph.record_table(
            table_type="isin",
            pages=[50, 51],
            columns=["Fund Name", "ISIN"],
            lookup_column="Fund Name",
            is_consolidated=True,
            source_phase="exploration",
        )

        assert table.entity_type == EntityType.TABLE
        assert table.id == "isin_50"
        assert table.properties["pages"] == [50, 51]

    def test_consolidated_table_links_to_all_funds(self, graph_with_fund):
        """Consolidated table creates relations to all existing funds."""
        graph_with_fund._add_entity(
            entity_type=EntityType.FUND,
            entity_id="Another Fund",
            name="Another Fund",
            properties={},
            source_phase="exploration",
        )

        graph_with_fund.record_table(
            table_type="isin",
            pages=[50],
            columns=["Fund Name", "ISIN"],
            lookup_column="Fund Name",
            is_consolidated=True,
            source_phase="exploration",
        )

        relations = graph_with_fund.query_relations(relation_type=RelationType.HAS_ISIN_IN)
        assert len(relations) == 2

    def test_non_consolidated_table_links_to_specific_fund(self, graph_with_fund):
        """Non-consolidated table links only to specified fund."""
        graph_with_fund._add_entity(
            entity_type=EntityType.FUND,
            entity_id="Another Fund",
            name="Another Fund",
            properties={},
            source_phase="exploration",
        )

        graph_with_fund.record_table(
            table_type="isin",
            pages=[11],
            columns=["Share Class", "ISIN"],
            lookup_column="Share Class",
            is_consolidated=False,
            source_phase="exploration",
            belongs_to_fund="Global Equity Fund",
        )

        relations = graph_with_fund.query_relations(relation_type=RelationType.HAS_ISIN_IN)
        assert len(relations) == 1
        assert relations[0].subject_key == "fund:Global Equity Fund"

    def test_fee_table_uses_correct_relation_type(self, graph_with_fund):
        """Fee tables create HAS_FEE_IN relations."""
        graph_with_fund.record_table(
            table_type="fee",
            pages=[60],
            columns=["Fund", "Management Fee"],
            lookup_column="Fund",
            is_consolidated=True,
            source_phase="exploration",
        )

        relations = graph_with_fund.query_relations(relation_type=RelationType.HAS_FEE_IN)
        assert len(relations) == 1


# =============================================================================
# record_fund Tests
# =============================================================================


class TestRecordFund:
    """Tests for store.record_fund."""

    def test_creates_fund_entity(self, empty_graph):
        """record_fund creates a fund entity."""
        fund = empty_graph.record_fund(
            name="New Fund",
            section_pages=[20, 21, 22],
            has_dedicated_section=True,
            source_phase="exploration",
            source_page=20,
        )

        assert fund.entity_type == EntityType.FUND
        assert fund.name == "New Fund"
        assert fund.properties["section_pages"] == [20, 21, 22]
        assert fund.properties["has_dedicated_section"] is True

    def test_uses_section_pages_as_source(self, empty_graph):
        """When source_page not given, uses section_pages as source."""
        fund = empty_graph.record_fund(
            name="New Fund",
            section_pages=[20, 21],
            has_dedicated_section=True,
            source_phase="exploration",
        )

        assert fund.source_pages == [20, 21]


# =============================================================================
# link_funds_to_table Tests
# =============================================================================


class TestLinkFundsToTable:
    """Tests for store.link_funds_to_table."""

    def test_links_all_funds(self, empty_graph):
        """Links all existing funds to the table."""
        empty_graph.record_fund("Fund A", [10], True, "exploration")
        empty_graph.record_fund("Fund B", [20], True, "exploration")
        empty_graph._add_entity(
            entity_type=EntityType.TABLE,
            entity_id="isin_50",
            name="ISIN Table",
            properties={},
            source_phase="exploration",
        )

        empty_graph.link_funds_to_table("isin_50", "isin", "exploration")

        relations = empty_graph.query_relations(relation_type=RelationType.HAS_ISIN_IN)
        assert len(relations) == 2

    def test_does_not_duplicate_relations(self, graph_with_fund_and_table):
        """Does not create duplicate relations."""
        initial_count = len(graph_with_fund_and_table.relations)

        graph_with_fund_and_table.link_funds_to_table("isin_50", "isin", "exploration")

        assert len(graph_with_fund_and_table.relations) == initial_count


# =============================================================================
# Utility Method Tests
# =============================================================================


class TestInferLookupColumn:
    """Tests for _infer_lookup_column method."""

    def test_prefers_fund_name(self, empty_graph):
        """Prefers 'Fund Name' when available."""
        columns = ["ISIN", "Fund Name", "Currency"]
        result = empty_graph._infer_lookup_column(columns)
        assert result == "Fund Name"

    def test_case_insensitive(self, empty_graph):
        """Matching is case insensitive."""
        columns = ["isin", "fund name", "currency"]
        result = empty_graph._infer_lookup_column(columns)
        assert result == "fund name"

    def test_fallback_order(self, empty_graph):
        """Falls back through candidates in order."""
        columns = ["ISIN", "Sub-Fund", "Currency"]
        result = empty_graph._infer_lookup_column(columns)
        assert result == "Sub-Fund"

    def test_first_column_fallback(self, empty_graph):
        """Falls back to first column when no match."""
        columns = ["ISIN", "Currency", "Amount"]
        result = empty_graph._infer_lookup_column(columns)
        assert result == "ISIN"


class TestStats:
    """Tests for stats method."""

    def test_empty_graph_stats(self, empty_graph):
        """Empty graph has zero counts."""
        stats = empty_graph.stats()
        assert stats["entities"] == 0
        assert stats["relations"] == 0
        assert stats["facts"] == 0

    def test_populated_graph_stats(self, graph_with_fund_and_table):
        """Stats reflect graph contents."""
        graph_with_fund_and_table._add_fact(
            entity_key="fund:Global Equity Fund",
            field_name="isin",
            value="LU0123456789",
            source_page=11,
            extraction_phase="extraction",
        )

        stats = graph_with_fund_and_table.stats()

        assert stats["entities"] == 2
        assert stats["entities_by_type"]["fund"] == 1
        assert stats["entities_by_type"]["table"] == 1
        assert stats["relations"] == 1
        assert stats["facts"] == 1


class TestSummary:
    """Tests for summary method."""

    def test_summary_format(self, graph_with_fund_and_table):
        """Summary returns human-readable string."""
        summary = graph_with_fund_and_table.summary()

        assert "Knowledge Graph:" in summary
        assert "Entities: 2" in summary
        assert "funds: 1" in summary
        assert "tables: 1" in summary
