"""Tests for extractor.core.constraint_parser module.

Tests the constraint parsing functions:
- parse_fund_constraints: Parse fund-level constraints
- parse_share_class_constraints: Parse share class constraints
- Helper functions for extracting numeric values
"""

from extractor.core.constraint_parser import (
    parse_fund_constraints,
    parse_share_class_constraints,
    _extract_percentage,
    _extract_amount,
    _extract_days,
    _detect_comparator,
    _extract_hedge_ratio,
)
from extractor.pydantic_models.provenance import ExtractedValue, NOT_FOUND


# =============================================================================
# Helper function tests
# =============================================================================


class TestExtractPercentage:
    """Tests for _extract_percentage helper."""

    def test_simple_percentage(self):
        """Extract simple percentage like '10%'."""
        assert _extract_percentage("max 10%") == 10.0

    def test_percentage_with_space(self):
        """Extract percentage with space like '10 %'."""
        assert _extract_percentage("up to 10 %") == 10.0

    def test_decimal_percentage(self):
        """Extract decimal percentage like '1.5%'."""
        assert _extract_percentage("fee of 1.5%") == 1.5

    def test_percentage_word(self):
        """Extract percentage written as word."""
        assert _extract_percentage("max 10 percent") == 10.0

    def test_no_percentage_returns_none(self):
        """Return None when no percentage found."""
        assert _extract_percentage("no limits specified") is None

    def test_large_percentage(self):
        """Extract large percentages like leverage ratios."""
        assert _extract_percentage("max 200% gross exposure") == 200.0


class TestExtractAmount:
    """Tests for _extract_amount helper."""

    def test_eur_amount(self):
        """Extract EUR amount."""
        amount, currency = _extract_amount("EUR 5,000")
        assert amount == 5000.0
        assert currency == "EUR"

    def test_usd_amount(self):
        """Extract USD amount."""
        amount, currency = _extract_amount("USD 1,000,000")
        assert amount == 1000000.0
        assert currency == "USD"

    def test_dollar_symbol(self):
        """Extract amount with $ symbol."""
        amount, currency = _extract_amount("$10,000")
        assert amount == 10000.0
        assert currency == "USD"

    def test_euro_symbol(self):
        """Extract amount with € symbol."""
        amount, currency = _extract_amount("€5,000")
        assert amount == 5000.0
        assert currency == "EUR"

    def test_amount_after_currency(self):
        """Extract when amount comes after currency code."""
        amount, currency = _extract_amount("minimum 5000 EUR")
        assert amount == 5000.0
        assert currency == "EUR"

    def test_no_amount_returns_none(self):
        """Return None when no amount found."""
        amount, currency = _extract_amount("institutional investors only")
        assert amount is None
        assert currency is None


class TestExtractDays:
    """Tests for _extract_days helper."""

    def test_numeric_days(self):
        """Extract numeric days like '30 days'."""
        assert _extract_days("30 days") == 30

    def test_days_singular(self):
        """Extract singular 'day'."""
        assert _extract_days("1 day") == 1

    def test_one_month(self):
        """Extract 'one month' as 30 days."""
        assert _extract_days("one month") == 30

    def test_three_months(self):
        """Extract 'three months' as 90 days."""
        assert _extract_days("three months") == 90

    def test_six_months(self):
        """Extract 'six months' as 180 days."""
        assert _extract_days("six months") == 180

    def test_one_year(self):
        """Extract 'one year' as 365 days."""
        assert _extract_days("one year") == 365

    def test_no_period_returns_none(self):
        """Return None when no period found."""
        assert _extract_days("no minimum holding") is None


class TestDetectComparator:
    """Tests for _detect_comparator helper."""

    def test_max_keyword(self):
        """Detect 'max' as maximum comparator."""
        assert _detect_comparator("max 10% single issuer") == "max"

    def test_not_more_than(self):
        """Detect 'not more than' as maximum."""
        assert _detect_comparator("not more than 10%") == "max"

    def test_shall_not_exceed(self):
        """Detect 'shall not exceed' as maximum."""
        assert _detect_comparator("shall not exceed 5%") == "max"

    def test_min_keyword(self):
        """Detect 'min' as minimum comparator."""
        assert _detect_comparator("min 70% equities") == "min"

    def test_at_least(self):
        """Detect 'at least' as minimum."""
        assert _detect_comparator("at least 80% investment grade") == "min"

    def test_no_comparator(self):
        """Return None when no comparator found."""
        assert _detect_comparator("bonds and equities") is None


class TestExtractHedgeRatio:
    """Tests for _extract_hedge_ratio helper."""

    def test_fully_hedged(self):
        """'Fully hedged' should return 100%."""
        assert _extract_hedge_ratio("fully hedged to EUR") == 100.0

    def test_full_hedge(self):
        """'Full hedge' should return 100%."""
        assert _extract_hedge_ratio("full hedge ratio") == 100.0

    def test_percentage_hedge(self):
        """Extract numeric hedge ratio."""
        assert _extract_hedge_ratio("95% hedge ratio") == 95.0

    def test_no_hedge_info(self):
        """Return None when no hedge info found."""
        assert _extract_hedge_ratio("currency exposure") is None


# =============================================================================
# parse_fund_constraints tests
# =============================================================================


class TestParseFundConstraints:
    """Tests for parse_fund_constraints function."""

    def test_empty_inputs_returns_empty_list(self):
        """No constraints when all inputs are None."""
        result = parse_fund_constraints(None, None, None)
        assert result == []

    def test_not_found_inputs_returns_empty_list(self):
        """No constraints when all inputs are NOT_FOUND."""
        not_found = ExtractedValue(value=NOT_FOUND, confidence=0.0)
        result = parse_fund_constraints(not_found, not_found, not_found)
        assert result == []

    def test_single_investment_restriction(self):
        """Parse single investment restriction."""
        restrictions = ExtractedValue(
            value="max 10% single issuer",
            source_page=50,
            confidence=0.9,
        )
        result = parse_fund_constraints(restrictions, None, None)

        assert len(result) == 1
        assert result[0].constraint_type == "concentration_limit"
        assert result[0].numeric_value == 10.0
        assert result[0].comparator == "max"

    def test_multiple_restrictions_semicolon_separated(self):
        """Parse multiple restrictions separated by semicolons."""
        restrictions = ExtractedValue(
            value="max 10% single issuer; min 70% equities; max 20% emerging markets",
            source_page=50,
            confidence=0.9,
        )
        result = parse_fund_constraints(restrictions, None, None)

        assert len(result) == 3
        types = [c.constraint_type for c in result]
        assert "concentration_limit" in types

    def test_leverage_policy_var(self):
        """Parse leverage policy with VaR."""
        leverage = ExtractedValue(
            value="VaR limit 20%",
            source_page=60,
            confidence=0.85,
        )
        result = parse_fund_constraints(None, leverage, None)

        assert len(result) == 1
        assert result[0].constraint_type == "var_limit"
        assert result[0].numeric_value == 20.0

    def test_leverage_policy_gross_exposure(self):
        """Parse leverage policy with gross exposure."""
        leverage = ExtractedValue(
            value="max 200% gross exposure",
            source_page=60,
            confidence=0.85,
        )
        result = parse_fund_constraints(None, leverage, None)

        assert len(result) == 1
        assert result[0].constraint_type == "leverage_limit"
        assert result[0].numeric_value == 200.0

    def test_derivatives_usage_policy(self):
        """Parse derivatives usage policy."""
        derivatives = ExtractedValue(
            value="hedging only, no speculation",
            source_page=70,
            confidence=0.8,
        )
        result = parse_fund_constraints(None, None, derivatives)

        assert len(result) == 1
        assert result[0].constraint_type == "derivative_policy"
        assert result[0].rule == "hedging only, no speculation"

    def test_combined_constraints(self):
        """Parse all constraint types together."""
        restrictions = ExtractedValue(value="max 10% single issuer", source_page=50, confidence=0.9)
        leverage = ExtractedValue(value="max 150% exposure", source_page=60, confidence=0.85)
        derivatives = ExtractedValue(value="permitted for hedging", source_page=70, confidence=0.8)

        result = parse_fund_constraints(restrictions, leverage, derivatives)

        assert len(result) == 3


# =============================================================================
# parse_share_class_constraints tests
# =============================================================================


class TestParseShareClassConstraints:
    """Tests for parse_share_class_constraints function."""

    def test_empty_inputs_returns_empty_list(self):
        """No constraints when all inputs are None."""
        result = parse_share_class_constraints(None, None, None, None, None)
        assert result == []

    def test_hedging_details(self):
        """Parse hedging details into constraint."""
        hedging = ExtractedValue(
            value="Hedged to EUR using FX forwards, targeting 100% hedge ratio",
            source_page=30,
            confidence=0.9,
        )
        result = parse_share_class_constraints(None, hedging, None, None, None)

        assert len(result) == 1
        assert result[0].constraint_type == "hedging_policy"
        assert result[0].numeric_value == 100.0

    def test_currency_hedged_true_no_details(self):
        """Parse hedged=True with no details."""
        hedged = ExtractedValue(value=True, source_page=30, confidence=0.9)
        result = parse_share_class_constraints(hedged, None, None, None, None)

        assert len(result) == 1
        assert result[0].constraint_type == "hedging_policy"
        assert "hedged" in result[0].rule.lower()

    def test_currency_hedged_false_no_constraint(self):
        """No hedging constraint when hedged=False."""
        hedged = ExtractedValue(value=False, source_page=30, confidence=0.9)
        result = parse_share_class_constraints(hedged, None, None, None, None)

        assert len(result) == 0

    def test_investor_restrictions(self):
        """Parse investor restrictions."""
        investor = ExtractedValue(
            value="Institutional investors only",
            source_page=25,
            confidence=0.85,
        )
        result = parse_share_class_constraints(None, None, investor, None, None)

        assert len(result) == 1
        assert result[0].constraint_type == "investor_eligibility"

    def test_minimum_investment(self):
        """Parse minimum investment."""
        min_invest = ExtractedValue(
            value="EUR 5,000",
            source_page=25,
            confidence=0.9,
        )
        result = parse_share_class_constraints(None, None, None, min_invest, None)

        assert len(result) == 1
        assert result[0].constraint_type == "minimum_investment"
        assert result[0].numeric_value == 5000.0
        assert result[0].unit == "EUR"

    def test_holding_period(self):
        """Parse minimum holding period."""
        holding = ExtractedValue(
            value="30 days",
            source_page=26,
            confidence=0.85,
        )
        result = parse_share_class_constraints(None, None, None, None, holding)

        assert len(result) == 1
        assert result[0].constraint_type == "holding_period"
        assert result[0].numeric_value == 30.0
        assert result[0].unit == "days"

    def test_combined_share_class_constraints(self):
        """Parse all share class constraint types."""
        hedging = ExtractedValue(value="Fully hedged to USD", source_page=30, confidence=0.9)
        investor = ExtractedValue(value="Retail investors", source_page=25, confidence=0.85)
        min_invest = ExtractedValue(value="USD 1,000", source_page=25, confidence=0.9)
        holding = ExtractedValue(value="one month", source_page=26, confidence=0.85)

        result = parse_share_class_constraints(None, hedging, investor, min_invest, holding)

        assert len(result) == 4
        types = [c.constraint_type for c in result]
        assert "hedging_policy" in types
        assert "investor_eligibility" in types
        assert "minimum_investment" in types
        assert "holding_period" in types
