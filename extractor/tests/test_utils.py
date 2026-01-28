"""Tests for value helpers and fund name utilities.

Tests the core utility functions:
- is_not_found: Check if value represents NOT_FOUND
- get_raw_value: Extract raw value from provenance dict
- normalize_fund_name: Normalize fund names for comparison
- fund_names_match: Check if two fund names refer to the same fund
"""

from extractor.core.value_helpers import is_not_found, get_raw_value
from extractor.core.fund_names import normalize_fund_name, fund_names_match


# =============================================================================
# is_not_found tests
# =============================================================================


class TestIsNotFound:
    """Tests for is_not_found function."""

    def test_none_is_not_found(self):
        """None should be considered NOT_FOUND."""
        assert is_not_found(None) is True

    def test_not_found_string_is_not_found(self):
        """Literal 'NOT_FOUND' string should be considered NOT_FOUND."""
        assert is_not_found("NOT_FOUND") is True

    def test_provenance_dict_with_not_found_is_not_found(self):
        """Provenance dict with value='NOT_FOUND' should be considered NOT_FOUND."""
        assert is_not_found({"value": "NOT_FOUND", "source_page": None}) is True

    def test_empty_string_is_not_not_found(self):
        """Empty string is not NOT_FOUND (it's a valid empty value)."""
        assert is_not_found("") is False

    def test_valid_string_is_not_not_found(self):
        """A valid string value should not be considered NOT_FOUND."""
        assert is_not_found("1.50%") is False

    def test_provenance_dict_with_value_is_not_not_found(self):
        """Provenance dict with a valid value should not be NOT_FOUND."""
        assert is_not_found({"value": "1.50%", "source_page": 10}) is False

    def test_zero_is_not_not_found(self):
        """Zero is a valid value, not NOT_FOUND."""
        assert is_not_found(0) is False

    def test_false_is_not_not_found(self):
        """False is a valid boolean, not NOT_FOUND."""
        assert is_not_found(False) is False

    def test_empty_list_is_not_not_found(self):
        """Empty list is valid, not NOT_FOUND."""
        assert is_not_found([]) is False


# =============================================================================
# get_raw_value tests
# =============================================================================


class TestGetRawValue:
    """Tests for get_raw_value function."""

    def test_plain_string_returned_as_is(self):
        """Plain string values are returned unchanged."""
        assert get_raw_value("Test Fund") == "Test Fund"

    def test_provenance_dict_extracts_value(self):
        """Value is extracted from provenance dict."""
        data = {"value": "Test Fund", "source_page": 1, "confidence": 0.9}
        assert get_raw_value(data) == "Test Fund"

    def test_none_returns_default(self):
        """None input returns the default value."""
        assert get_raw_value(None, "Default") == "Default"

    def test_none_with_no_default_returns_none(self):
        """None input with no default returns None."""
        assert get_raw_value(None) is None

    def test_integer_returned_as_is(self):
        """Integer values are returned unchanged."""
        assert get_raw_value(42) == 42

    def test_boolean_returned_as_is(self):
        """Boolean values are returned unchanged."""
        assert get_raw_value(True) is True
        assert get_raw_value(False) is False

    def test_dict_without_value_key_returned_as_is(self):
        """Dict without 'value' key is returned unchanged."""
        data = {"name": "Test", "page": 10}
        assert get_raw_value(data) == data

    def test_nested_provenance_extracts_first_level(self):
        """Only extracts value from first level, not recursively."""
        data = {"value": {"nested": "data"}, "source_page": 1}
        assert get_raw_value(data) == {"nested": "data"}


# =============================================================================
# normalize_fund_name tests
# =============================================================================


class TestNormalizeFundName:
    """Tests for normalize_fund_name function."""

    def test_lowercase_conversion(self):
        """Names are converted to lowercase."""
        assert normalize_fund_name("Global EQUITY Fund") == "global equity fund"

    def test_whitespace_trimming(self):
        """Leading/trailing whitespace is trimmed."""
        assert normalize_fund_name("  Global Fund  ") == "global fund"

    def test_multiple_spaces_collapsed(self):
        """Multiple spaces are collapsed to single spaces."""
        assert normalize_fund_name("Global   Equity    Fund") == "global equity fund"

    def test_hyphen_replaced_with_space(self):
        """Hyphens are replaced with spaces."""
        assert normalize_fund_name("JPM-Global-Fund") == "jpm global fund"

    def test_en_dash_replaced(self):
        """En-dashes ( – ) are replaced with spaces."""
        assert normalize_fund_name("JPM Investment – Global Fund") == "jpm investment global fund"

    def test_em_dash_replaced(self):
        """Em-dashes ( — ) are replaced with spaces."""
        assert normalize_fund_name("JPM — Global") == "jpm global"

    def test_spaced_hyphen_replaced(self):
        """Spaced hyphens ( - ) are replaced with single space."""
        assert normalize_fund_name("JPM Investment - Global Fund") == "jpm investment global fund"

    def test_complex_name_normalized(self):
        """Complex names with multiple separators are normalized correctly."""
        name = "  JPMorgan Investment Funds - Global Equity Fund  "
        assert normalize_fund_name(name) == "jpmorgan investment funds global equity fund"


# =============================================================================
# fund_names_match tests
# =============================================================================


class TestFundNamesMatch:
    """Tests for fund_names_match function."""

    def test_exact_match(self):
        """Identical names match."""
        assert fund_names_match("Global Equity Fund", "Global Equity Fund") is True

    def test_case_insensitive_match(self):
        """Names match regardless of case."""
        assert fund_names_match("Global Equity Fund", "GLOBAL EQUITY FUND") is True

    def test_normalized_match(self):
        """Names match after normalization."""
        assert fund_names_match(
            "Global Equity Fund",
            "  global  equity  fund  "
        ) is True

    def test_suffix_match_short_in_long(self):
        """Short name matches as suffix of longer name."""
        assert fund_names_match(
            "JPMorgan Investment Funds - Global Equity Fund",
            "Global Equity Fund"
        ) is True

    def test_suffix_match_long_in_short(self):
        """Longer name containing shorter name matches."""
        assert fund_names_match(
            "Global Equity Fund",
            "JPMorgan Investment Funds - Global Equity Fund"
        ) is True

    def test_fuzzy_match_high_similarity(self):
        """Names with high similarity match via fuzzy matching."""
        # Typo in name still matches (above 0.85 threshold)
        assert fund_names_match(
            "Global Equity Fund",
            "Global Equty Fund"  # Missing 'i'
        ) is True

    def test_fuzzy_match_low_similarity_fails(self):
        """Names with low similarity don't match."""
        assert fund_names_match(
            "Global Equity Fund",
            "Fixed Income Strategy"
        ) is False

    def test_different_funds_dont_match(self):
        """Different funds with similar structure don't match."""
        assert fund_names_match(
            "Global Equity Fund",
            "Global Bond Fund"
        ) is False

    def test_custom_threshold(self):
        """Custom threshold can be specified."""
        # With default threshold (0.85), this might match
        # With strict threshold (0.95), it shouldn't
        assert fund_names_match(
            "Global Equity Fund",
            "Global Equty Fund",
            threshold=0.99
        ) is False

    def test_umbrella_prefix_matches_fund(self):
        """Fund with umbrella prefix matches fund name alone."""
        assert fund_names_match(
            "JPM Investment Funds - Europe Select Equity",
            "europe select equity"
        ) is True

    def test_empty_strings_match_due_to_suffix_rule(self):
        """Empty string matches via suffix rule (any string.endswith('') is True).

        Note: This reflects current implementation behavior. The suffix check
        `norm_target.endswith(norm_candidate)` returns True when candidate
        normalizes to empty since any string ends with empty string in Python.
        """
        assert fund_names_match("Global Fund", "") is True
        assert fund_names_match("", "Global Fund") is True

    def test_both_empty_strings_match(self):
        """Two empty strings are considered a match (exact match)."""
        assert fund_names_match("", "") is True
