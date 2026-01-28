"""Tests for resolver question-awareness in field_searchers."""

import pytest
from unittest.mock import MagicMock, AsyncMock

from extractor.core.document_knowledge import DocumentKnowledge, QuestionPriority
from extractor.core.field_searchers import ISINResolver


# --- Feature 2: resolver + question history tests ---

def test_resolver_excludes_searched_pages():
    knowledge = DocumentKnowledge()
    # Add a finding so get_pages_for_field returns pages
    from extractor.core.document_knowledge import Finding, FindingType
    knowledge.add_finding(Finding(
        finding_type=FindingType.SECTION_LOCATION,
        description="ISIN section",
        source_agent="explorer",
        pages=[10, 11, 12],
        field_name="isin",
    ))
    # Post a question saying pages 10,11 were already searched
    knowledge.ask_question(
        question="Could not find isin for Fund A",
        field_name="isin",
        entity_name="Fund A",
        source_agent="extraction_Fund A",
        pages_searched=[10, 11],
    )

    resolver = ISINResolver()
    pages = resolver._get_pages_from_knowledge(knowledge, "Fund A")
    assert pages == [12]


def test_resolver_no_questions_returns_all():
    knowledge = DocumentKnowledge()
    from extractor.core.document_knowledge import Finding, FindingType
    knowledge.add_finding(Finding(
        finding_type=FindingType.SECTION_LOCATION,
        description="ISIN section",
        source_agent="explorer",
        pages=[10, 11, 12],
        field_name="isin",
    ))
    resolver = ISINResolver()
    pages = resolver._get_pages_from_knowledge(knowledge, "Fund A")
    assert pages == [10, 11, 12]


def test_get_pages_already_searched_empty():
    knowledge = DocumentKnowledge()
    result = knowledge.get_pages_already_searched("isin", "Fund A")
    assert result == set()


def test_get_pages_already_searched_unions_multiple():
    knowledge = DocumentKnowledge()
    knowledge.ask_question(
        question="Q1",
        field_name="isin",
        entity_name="Fund A",
        source_agent="agent1",
        pages_searched=[1, 2],
    )
    knowledge.ask_question(
        question="Q2",
        field_name="isin",
        entity_name="Fund A",
        source_agent="agent2",
        pages_searched=[3, 4],
    )
    result = knowledge.get_pages_already_searched("isin", "Fund A")
    assert result == {1, 2, 3, 4}
