"""Tests for KnowledgeContext cross-reference integration."""

import pytest
from unittest.mock import MagicMock

from extractor.core.field_strategy import KnowledgeContext, FieldStrategy
from extractor.core.document_knowledge import DocumentKnowledge, Finding, FindingType
from extractor.pydantic_models.graph_models import RelationType, Relation


def _make_context(relations=None, knowledge=None):
    """Build a KnowledgeContext with a mocked GraphStore."""
    store = MagicMock()
    store.query_relations.return_value = relations or []
    knowledge = knowledge or DocumentKnowledge()
    return KnowledgeContext(knowledge=knowledge, store=store)


# --- Feature 1: cross-ref tests ---

def test_cross_ref_provides_pages():
    rel = Relation(
        relation_type=RelationType.REFERENCES,
        subject_key="fund:Fund X",
        object_key="appendix:E",
        properties={"field_hint": "redemption_fee", "target_pages": [150, 151]},
    )
    ctx = _make_context(relations=[rel])
    strategy = ctx.get_field_strategy("redemption_fee", "Fund X")
    assert strategy.strategy == "text_extraction"
    assert strategy.pages == [150, 151]
    assert strategy.confidence == 0.85


def test_cross_ref_skipped_when_external():
    rel = Relation(
        relation_type=RelationType.REFERENCES,
        subject_key="fund:Fund X",
        object_key="appendix:E",
        properties={"field_hint": "redemption_fee", "target_pages": [150], "is_external": True},
    )
    ctx = _make_context(relations=[rel])
    # _check_cross_refs should skip this because is_external is truthy
    result = ctx._check_cross_refs("redemption_fee", "Fund X")
    assert result is None


def test_cross_ref_no_fund_name():
    ctx = _make_context()
    strategy = ctx.get_field_strategy("redemption_fee", None)
    # Should fall through to search (no fund_name → cross-ref skipped)
    assert strategy.strategy == "search"


def test_cross_ref_no_match():
    rel = Relation(
        relation_type=RelationType.REFERENCES,
        subject_key="fund:Fund X",
        object_key="appendix:E",
        properties={"field_hint": "management_fee", "target_pages": [100]},
    )
    ctx = _make_context(relations=[rel])
    strategy = ctx.get_field_strategy("redemption_fee", "Fund X")
    # field_hint doesn't match → falls through to search
    assert strategy.strategy == "search"


def test_table_beats_cross_ref():
    """Table finding should win over cross-ref (higher in chain)."""
    knowledge = DocumentKnowledge()
    knowledge.add_finding(Finding(
        finding_type=FindingType.TABLE_LOCATION,
        description="Fee table",
        source_agent="explorer",
        pages=[200],
        field_name="management_fee",
        metadata={"columns": ["Fund Name", "Management Fee"]},
    ))
    rel = Relation(
        relation_type=RelationType.REFERENCES,
        subject_key="fund:Fund X",
        object_key="appendix:E",
        properties={"field_hint": "management_fee", "target_pages": [150]},
    )
    ctx = _make_context(relations=[rel], knowledge=knowledge)
    strategy = ctx.get_field_strategy("management_fee", "Fund X")
    assert strategy.strategy == "table_lookup"
    assert strategy.confidence == 0.9
