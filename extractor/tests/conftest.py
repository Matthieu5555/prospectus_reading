"""Pytest configuration and shared fixtures.

Provides reusable test fixtures for:
- Mock PDF readers
- Mock LLM responses
- Sample exploration notes
- Sample extracted data
"""

import pytest
from unittest.mock import MagicMock, patch
from dataclasses import dataclass

from extractor.pydantic_models.provenance import ExtractedValue, NOT_FOUND, NotFoundReason


# =============================================================================
# Mock PDF Reader
# =============================================================================


@pytest.fixture
def mock_pdf_reader():
    """Create a mock PDFReader with sample pages."""
    reader = MagicMock()
    reader.filename = "test_prospectus.pdf"
    reader.page_count = 100

    # Sample page content
    page_content = {
        1: "Table of Contents\n1. Introduction\n2. Fund Information\n3. Fees",
        2: "JPMorgan Investment Funds\nUmbrella Information\nManagement Company: JPM Asset Management",
        10: "Global Equity Fund\nInvestment Objective: Long-term capital growth",
        11: "Global Equity Fund\nShare Classes:\n- A (acc) USD - ISIN: LU0123456789\n- A (dist) EUR - ISIN: LU0123456790",
        50: "Fee Table\nGlobal Equity Fund\nManagement Fee: 1.50%\nEntry Fee: 5.00%",
    }

    def read_pages(start, end):
        result = []
        for page in range(start, end + 1):
            if page in page_content:
                result.append(f"--- Page {page} ---\n{page_content[page]}")
            else:
                result.append(f"--- Page {page} ---\nGeneric content for page {page}")
        return "\n\n".join(result)

    reader.read_pages = read_pages
    reader.get_page_chunks = lambda chunk_size=30: [(1, 30), (31, 60), (61, 90), (91, 100)]
    reader.search_text = lambda pattern: [{"page": 11, "text": "ISIN: LU0123456789"}]

    return reader


# =============================================================================
# Mock LLM Client
# =============================================================================


@pytest.fixture
def mock_llm_response():
    """Factory for creating mock LLM responses."""
    @dataclass
    class MockResponse:
        content: dict

    def _create_response(content: dict):
        return MockResponse(content=content)

    return _create_response


@pytest.fixture
def mock_llm_client(mock_llm_response):
    """Create a mock LLMClient that returns predefined responses."""
    with patch("extractor.core.llm_client.acompletion") as mock_completion:
        # Default response
        mock_completion.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content='{"result": "test"}'))],
            usage=MagicMock(prompt_tokens=100, completion_tokens=50)
        )
        yield mock_completion


# =============================================================================
# Sample Extracted Values
# =============================================================================


@pytest.fixture
def sample_extracted_value():
    """Create a sample ExtractedValue with full provenance."""
    return ExtractedValue(
        value="1.50%",
        source_page=50,
        source_quote="Management Fee: 1.50%",
        rationale="Extracted from fee table",
        confidence=0.95,
    )


@pytest.fixture
def sample_not_found_value():
    """Create a NOT_FOUND ExtractedValue."""
    return ExtractedValue(
        value=NOT_FOUND,
        source_page=None,
        source_quote=None,
        rationale="Field not found in reviewed pages",
        confidence=0.0,
        not_found_reason=NotFoundReason.EXTRACTION_FAILED,
    )


@pytest.fixture
def sample_external_ref_value():
    """Create a NOT_FOUND value with external reference."""
    return ExtractedValue(
        value=NOT_FOUND,
        source_page=5,
        source_quote="For ISIN codes, see the KIID document",
        rationale="Explicitly referenced to external document",
        confidence=0.8,
        not_found_reason=NotFoundReason.IN_EXTERNAL_DOC,
        external_reference="KIID",
    )


# =============================================================================
# Sample Fund Data
# =============================================================================


@pytest.fixture
def sample_fund_data():
    """Create sample fund extraction data."""
    return {
        "name": "Global Equity Fund",
        "investment_objective": {
            "value": "Long-term capital growth through equity investments",
            "source_page": 10,
            "confidence": 0.9,
        },
        "investment_restrictions": {
            "value": "max 10% single issuer; min 70% equities",
            "source_page": 15,
            "confidence": 0.85,
        },
        "leverage_policy": {
            "value": "NOT_FOUND",
            "not_found_reason": "not_in_document",
        },
        "share_classes": [
            {
                "name": "A (acc) USD",
                "isin": {"value": "LU0123456789", "source_page": 11, "confidence": 0.95},
                "currency": {"value": "USD", "source_page": 11, "confidence": 0.95},
                "management_fee": {"value": "1.50%", "source_page": 50, "confidence": 0.9},
            },
            {
                "name": "A (dist) EUR",
                "isin": {"value": "LU0123456790", "source_page": 11, "confidence": 0.95},
                "currency": {"value": "EUR", "source_page": 11, "confidence": 0.95},
                "management_fee": {"value": "1.50%", "source_page": 50, "confidence": 0.9},
            },
        ],
    }


@pytest.fixture
def sample_exploration_notes():
    """Create sample exploration notes."""
    from extractor.pydantic_models.pipeline import (
        ExplorationNotes, FundMention, TableInfo, CrossReference, PageIndexEntry
    )

    return ExplorationNotes(
        page_start=1,
        page_end=30,
        toc_pages=[1],
        umbrella_info_pages=[2, 3],
        funds_mentioned=[
            FundMention(name="Global Equity Fund", page=10, has_dedicated_section=True),
            FundMention(name="Global Bond Fund", page=20, has_dedicated_section=True),
        ],
        tables=[
            TableInfo(
                table_type="fee_table",
                page_start=50,
                page_end=52,
                columns=["Fund Name", "Management Fee", "Entry Fee"],
            ),
        ],
        cross_references=[
            CrossReference(
                source_page=5,
                text="See KIID for detailed ISIN information",
                target_description="KIID document",
            ),
        ],
        page_index=[
            PageIndexEntry(page=10, content_type="fund_section", fund_name="Global Equity Fund"),
            PageIndexEntry(page=20, content_type="fund_section", fund_name="Global Bond Fund"),
            PageIndexEntry(page=50, content_type="fee_table", description="Management fees"),
        ],
        observations=["Document has clear structure", "Fee table spans multiple pages"],
    )
