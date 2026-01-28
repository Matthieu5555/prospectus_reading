"""Tests for extractor.pydantic_models.provenance module.

Tests the ExtractedValue model and related functionality:
- ExtractedValue creation and validation
- NOT_FOUND handling with reasons
- Factory methods (from_simple, extraction_failed, etc.)
- Serialization
"""

from extractor.pydantic_models.provenance import (
    ExtractedValue,
    NotFoundReason,
    NOT_FOUND,
)


# =============================================================================
# ExtractedValue creation tests
# =============================================================================


class TestExtractedValueCreation:
    """Tests for ExtractedValue creation."""

    def test_basic_creation(self):
        """Create ExtractedValue with basic fields."""
        ev = ExtractedValue(
            value="1.50%",
            source_page=50,
            confidence=0.9,
        )
        assert ev.value == "1.50%"
        assert ev.source_page == 50
        assert ev.confidence == 0.9

    def test_full_provenance(self):
        """Create ExtractedValue with full provenance."""
        ev = ExtractedValue(
            value="Global Equity Fund",
            source_page=10,
            source_quote="The fund is named Global Equity Fund",
            rationale="Extracted from fund description section",
            confidence=0.95,
        )
        assert ev.source_quote == "The fund is named Global Equity Fund"
        assert ev.rationale == "Extracted from fund description section"

    def test_not_found_with_reason(self):
        """Create NOT_FOUND with explicit reason."""
        ev = ExtractedValue(
            value=NOT_FOUND,
            source_page=None,
            confidence=0.0,
            not_found_reason=NotFoundReason.NOT_IN_DOCUMENT,
        )
        assert ev.is_not_found
        assert ev.not_found_reason == NotFoundReason.NOT_IN_DOCUMENT

    def test_not_found_with_external_reference(self):
        """Create NOT_FOUND with external document reference."""
        ev = ExtractedValue(
            value=NOT_FOUND,
            source_page=5,
            source_quote="For ISIN codes, see KIID",
            confidence=0.8,
            not_found_reason=NotFoundReason.IN_EXTERNAL_DOC,
            external_reference="KIID",
        )
        assert ev.external_reference == "KIID"
        assert ev.not_found_reason == NotFoundReason.IN_EXTERNAL_DOC


# =============================================================================
# Factory method tests
# =============================================================================


class TestExtractedValueFactoryMethods:
    """Tests for ExtractedValue factory methods."""

    def test_from_simple_string(self):
        """Create from simple string value."""
        ev = ExtractedValue.from_simple("Test Value")
        assert ev.value == "Test Value"
        # from_simple uses DEFAULT_UNVERIFIED confidence (0.5) since no full provenance
        assert ev.confidence == 0.5

    def test_from_simple_with_page(self):
        """Create from simple value with page number."""
        ev = ExtractedValue.from_simple("Test", page=10)
        assert ev.value == "Test"
        assert ev.source_page == 10

    def test_from_simple_not_found(self):
        """from_simple with NOT_FOUND creates extraction_failed."""
        ev = ExtractedValue.from_simple(NOT_FOUND)
        assert ev.is_not_found
        assert ev.not_found_reason == NotFoundReason.EXTRACTION_FAILED

    def test_extraction_failed(self):
        """Create extraction_failed value."""
        ev = ExtractedValue.extraction_failed("Could not find ISIN")
        assert ev.is_not_found
        assert ev.not_found_reason == NotFoundReason.EXTRACTION_FAILED
        assert "ISIN" in ev.rationale

    def test_not_in_document(self):
        """Create not_in_document value."""
        ev = ExtractedValue.not_in_document("No dividend policy found")
        assert ev.is_not_found
        assert ev.not_found_reason == NotFoundReason.NOT_IN_DOCUMENT

    def test_not_applicable(self):
        """Create not_applicable value."""
        ev = ExtractedValue.not_applicable("Hedging not relevant for USD share class")
        assert ev.is_not_found
        assert ev.not_found_reason == NotFoundReason.NOT_APPLICABLE

    def test_in_external_doc(self):
        """Create in_external_doc value."""
        ev = ExtractedValue.in_external_doc(
            rationale="See KIID for details",
            external_doc="KIID",
        )
        assert ev.is_not_found
        assert ev.not_found_reason == NotFoundReason.IN_EXTERNAL_DOC
        assert ev.external_reference == "KIID"
        assert "KIID" in ev.rationale


# =============================================================================
# Property tests
# =============================================================================


class TestExtractedValueProperties:
    """Tests for ExtractedValue properties."""

    def test_is_not_found_true_for_not_found(self):
        """is_not_found returns True for NOT_FOUND value."""
        ev = ExtractedValue(value=NOT_FOUND, confidence=0.0)
        assert ev.is_not_found is True

    def test_is_not_found_false_for_value(self):
        """is_not_found returns False for valid value."""
        ev = ExtractedValue(value="1.50%", confidence=0.9)
        assert ev.is_not_found is False

    def test_is_not_found_false_for_empty_string(self):
        """is_not_found returns False for empty string (valid value)."""
        ev = ExtractedValue(value="", confidence=0.9)
        assert ev.is_not_found is False

    def test_is_not_found_false_for_zero(self):
        """is_not_found returns False for zero (valid value)."""
        ev = ExtractedValue(value=0, confidence=0.9)
        assert ev.is_not_found is False

    def test_is_not_found_false_for_false(self):
        """is_not_found returns False for boolean False (valid value)."""
        ev = ExtractedValue(value=False, confidence=0.9)
        assert ev.is_not_found is False

    def test_is_actionable_not_found_for_extraction_failed(self):
        """is_actionable_not_found returns True for EXTRACTION_FAILED."""
        ev = ExtractedValue.extraction_failed("Test")
        assert ev.is_actionable_not_found is True

    def test_is_actionable_not_found_false_for_not_in_document(self):
        """is_actionable_not_found returns False for NOT_IN_DOCUMENT."""
        ev = ExtractedValue.not_in_document("Test")
        assert ev.is_actionable_not_found is False

    def test_is_actionable_not_found_false_for_external_doc(self):
        """is_actionable_not_found returns False for IN_EXTERNAL_DOC."""
        ev = ExtractedValue.in_external_doc("See KIID for ISINs", "KIID")
        assert ev.is_actionable_not_found is False


# =============================================================================
# NotFoundReason tests
# =============================================================================


class TestNotFoundReason:
    """Tests for NotFoundReason enum."""

    def test_all_reasons_exist(self):
        """All expected NOT_FOUND reasons exist."""
        assert NotFoundReason.NOT_IN_DOCUMENT
        assert NotFoundReason.NOT_APPLICABLE
        assert NotFoundReason.IN_EXTERNAL_DOC
        assert NotFoundReason.INHERITED
        assert NotFoundReason.EXTRACTION_FAILED

    def test_reasons_are_strings(self):
        """Reasons serialize to strings."""
        assert NotFoundReason.NOT_IN_DOCUMENT.value == "not_in_document"
        assert NotFoundReason.EXTRACTION_FAILED.value == "extraction_failed"


# =============================================================================
# Serialization tests
# =============================================================================


class TestExtractedValueSerialization:
    """Tests for ExtractedValue serialization."""

    def test_model_dump_includes_all_fields(self):
        """model_dump includes all fields."""
        ev = ExtractedValue(
            value="Test",
            source_page=10,
            source_quote="Quote",
            rationale="Rationale",
            confidence=0.9,
        )
        data = ev.model_dump()

        assert data["value"] == "Test"
        assert data["source_page"] == 10
        assert data["source_quote"] == "Quote"
        assert data["rationale"] == "Rationale"
        assert data["confidence"] == 0.9

    def test_model_dump_excludes_none(self):
        """model_dump can exclude None values."""
        ev = ExtractedValue(value="Test", confidence=0.9)
        data = ev.model_dump(exclude_none=True)

        assert "source_page" not in data
        assert "source_quote" not in data

    def test_not_found_serialization(self):
        """NOT_FOUND values serialize correctly."""
        ev = ExtractedValue.extraction_failed("Missing data")
        data = ev.model_dump()

        assert data["value"] == NOT_FOUND
        assert data["not_found_reason"] == "extraction_failed"
