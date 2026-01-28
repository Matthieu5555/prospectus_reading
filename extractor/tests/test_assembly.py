"""Tests for assembly module components.

Tests the refactored assembly sub-modules:
- converters: to_extracted_value, to_name_extracted_value
- statistics: count_provenance, build_cost_summary
- graph_builder: build_umbrella, build_subfund, build_share_class
"""


from extractor.pydantic_models import (
    ExtractedValue,
    NOT_FOUND,
    NotFoundReason,
    Umbrella,
    SubFund,
    ShareClass,
)
from extractor.phases.assembly.converters import (
    not_found_reason_from_string,
    to_name_extracted_value,
    to_extracted_value,
)
from extractor.phases.assembly.graph_builder import (
    build_umbrella,
    build_share_class,
    build_subfund,
)


# =============================================================================
# Converter Tests
# =============================================================================


class TestNotFoundReasonFromString:
    """Tests for not_found_reason_from_string converter."""

    def test_returns_extraction_failed_for_none(self):
        result = not_found_reason_from_string(None)
        assert result == NotFoundReason.EXTRACTION_FAILED

    def test_returns_extraction_failed_for_empty_string(self):
        result = not_found_reason_from_string("")
        assert result == NotFoundReason.EXTRACTION_FAILED

    def test_parses_not_in_document(self):
        result = not_found_reason_from_string("not_in_document")
        assert result == NotFoundReason.NOT_IN_DOCUMENT

    def test_parses_not_applicable(self):
        result = not_found_reason_from_string("not_applicable")
        assert result == NotFoundReason.NOT_APPLICABLE

    def test_parses_in_external_doc(self):
        result = not_found_reason_from_string("in_external_doc")
        assert result == NotFoundReason.IN_EXTERNAL_DOC

    def test_parses_inherited(self):
        result = not_found_reason_from_string("inherited")
        assert result == NotFoundReason.INHERITED

    def test_parses_case_insensitive(self):
        result = not_found_reason_from_string("NOT_IN_DOCUMENT")
        assert result == NotFoundReason.NOT_IN_DOCUMENT

    def test_returns_extraction_failed_for_unknown(self):
        result = not_found_reason_from_string("unknown_reason")
        assert result == NotFoundReason.EXTRACTION_FAILED


class TestToNameExtractedValue:
    """Tests for to_name_extracted_value converter."""

    def test_returns_default_for_none(self):
        result = to_name_extracted_value(None, default="Unknown")
        assert isinstance(result, ExtractedValue)
        assert result.value == "Unknown"

    def test_returns_existing_extracted_value(self):
        ev = ExtractedValue(value="Test Fund", source_page=5)
        result = to_name_extracted_value(ev)
        assert result is ev

    def test_converts_provenance_dict(self):
        data = {
            "value": "Test Fund",
            "source_page": 10,
            "source_quote": "The Test Fund",
            "confidence": 0.95,
        }
        result = to_name_extracted_value(data)
        assert isinstance(result, ExtractedValue)
        assert result.value == "Test Fund"
        assert result.source_page == 10
        assert result.confidence == 0.95

    def test_converts_simple_string(self):
        result = to_name_extracted_value("Test Fund")
        assert isinstance(result, ExtractedValue)
        assert result.value == "Test Fund"

    def test_returns_default_for_not_found(self):
        result = to_name_extracted_value(NOT_FOUND, default="Unknown")
        assert isinstance(result, ExtractedValue)
        assert result.value == "Unknown"

    def test_returns_default_for_empty_string(self):
        result = to_name_extracted_value("", default="Unknown")
        assert isinstance(result, ExtractedValue)
        assert result.value == "Unknown"


class TestToExtractedValue:
    """Tests for to_extracted_value converter."""

    def test_returns_none_for_none(self):
        result = to_extracted_value(None)
        assert result is None

    def test_converts_provenance_dict(self):
        data = {
            "value": "1.50%",
            "source_page": 50,
            "source_quote": "Management Fee: 1.50%",
            "rationale": "Found in fee table",
            "confidence": 0.9,
        }
        result = to_extracted_value(data, "management_fee")
        assert isinstance(result, ExtractedValue)
        assert result.value == "1.50%"
        assert result.source_page == 50
        assert result.confidence == 0.9

    def test_converts_not_found_with_reason(self):
        data = {
            "value": NOT_FOUND,
            "not_found_reason": "in_external_doc",
            "external_reference": "KIID",
        }
        result = to_extracted_value(data)
        assert isinstance(result, ExtractedValue)
        assert result.is_not_found
        assert result.not_found_reason == NotFoundReason.IN_EXTERNAL_DOC
        assert result.external_reference == "KIID"

    def test_converts_legacy_string(self):
        result = to_extracted_value("1.50%", "management_fee")
        assert isinstance(result, ExtractedValue)
        assert result.value == "1.50%"

    def test_converts_legacy_bool(self):
        result = to_extracted_value(True, "currency_hedged")
        assert isinstance(result, ExtractedValue)
        assert result.value is True

    def test_converts_legacy_not_found(self):
        result = to_extracted_value(NOT_FOUND, "isin")
        assert isinstance(result, ExtractedValue)
        assert result.is_not_found
        assert result.not_found_reason == NotFoundReason.EXTRACTION_FAILED


# =============================================================================
# Graph Builder Tests
# =============================================================================


class TestBuildUmbrella:
    """Tests for build_umbrella function."""

    def test_builds_umbrella_from_raw_data(self):
        data = {
            "name": "JPMorgan Investment Funds",
            "legal_form": {"value": "SICAV", "source_page": 2},
            "domicile": "Luxembourg",
            "management_company": {"value": "JPM Asset Management", "source_page": 3},
        }
        umbrella = build_umbrella(data, default_name="Unknown")

        assert isinstance(umbrella, Umbrella)
        assert umbrella.legal_form.value == "SICAV"
        assert umbrella.legal_form.source_page == 2
        assert umbrella.domicile.value == "Luxembourg"

    def test_uses_default_name_when_missing(self):
        data = {}
        umbrella = build_umbrella(data, default_name="Default Umbrella")

        name = umbrella.name
        if isinstance(name, ExtractedValue):
            assert name.value == "Default Umbrella"
        else:
            assert name == "Default Umbrella"


class TestBuildShareClass:
    """Tests for build_share_class function."""

    def test_builds_share_class_from_raw_data(self):
        data = {
            "name": "A (acc) USD",
            "isin": {"value": "LU0123456789", "source_page": 100},
            "currency": "USD",
            "management_fee": "1.50%",
            "distribution_policy": {"value": "Accumulating", "source_page": 15},
        }
        sc = build_share_class(data)

        assert isinstance(sc, ShareClass)
        assert sc.isin.value == "LU0123456789"
        assert sc.isin.source_page == 100
        assert sc.currency.value == "USD"
        assert sc.management_fee.value == "1.50%"

    def test_handles_missing_fields(self):
        data = {"name": "A (acc) USD"}
        sc = build_share_class(data)

        assert isinstance(sc, ShareClass)
        assert sc.isin is None


class TestBuildSubfund:
    """Tests for build_subfund function."""

    def test_builds_subfund_with_share_classes(self):
        data = {
            "name": "Global Equity Fund",
            "investment_objective": {
                "value": "Long-term capital growth",
                "source_page": 10,
            },
            "share_classes": [
                {"name": "A (acc) USD", "isin": "LU0123456789"},
                {"name": "A (dist) EUR", "isin": "LU0123456790"},
            ],
        }
        subfund = build_subfund(data)

        assert isinstance(subfund, SubFund)
        assert subfund.investment_objective.value == "Long-term capital growth"
        assert len(subfund.share_classes) == 2
        assert subfund.share_classes[0].isin.value == "LU0123456789"

    def test_handles_empty_share_classes(self):
        data = {
            "name": "Global Bond Fund",
            "investment_objective": "Fixed income returns",
        }
        subfund = build_subfund(data)

        assert isinstance(subfund, SubFund)
        assert len(subfund.share_classes) == 0
