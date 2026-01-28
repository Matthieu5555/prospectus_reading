"""Parse flat fields into typed Constraint objects.

Converts text like "max 10% single issuer; max 200% gross exposure"
into typed Constraint objects with parsed numeric values.
"""

import re
from extractor.pydantic_models.constraints import Constraint
from extractor.pydantic_models.provenance import ExtractedValue


# Keywords for classifying restriction text into constraint types.
# Order matters: first match wins.
RESTRICTION_KEYWORDS: dict[str, list[str]] = {
    "concentration_limit": ["single issuer", "issuer", "concentration", "counterparty"],
    "geographic_limit": ["geographic", "country", "region", "eu ", "european", "emerging"],
    "sector_limit": ["sector", "industry"],
    "liquidity_limit": ["liquid", "liquidity"],
    "asset_limit": ["bond", "equity", "debt", "fixed income", "stock"],
}

# Keywords indicating max vs min comparator
MAX_KEYWORDS = ["max", "not more than", "no more than", "shall not exceed", "up to"]
MIN_KEYWORDS = ["min", "at least"]


def parse_fund_constraints(
    investment_restrictions: ExtractedValue | None,
    leverage_policy: ExtractedValue | None,
    derivatives_usage: ExtractedValue | None,
) -> list[Constraint]:
    """Parse fund-level flat fields into typed constraints.

    Args:
        investment_restrictions: Raw text like "max 10% single issuer; min 70% bonds"
        leverage_policy: Raw text like "max 200% gross exposure"
        derivatives_usage: Raw text like "hedging only, no speculation"

    Returns:
        List of typed Constraint objects.
    """
    constraints = []

    if investment_restrictions and not investment_restrictions.is_not_found:
        constraints.extend(_parse_investment_restrictions(investment_restrictions))

    if leverage_policy and not leverage_policy.is_not_found:
        constraints.extend(_parse_leverage_policy(leverage_policy))

    if derivatives_usage and not derivatives_usage.is_not_found:
        constraints.extend(_parse_derivatives_usage(derivatives_usage))

    return constraints


def parse_share_class_constraints(
    currency_hedged: ExtractedValue | None,
    hedging_details: ExtractedValue | None,
    investor_restrictions: ExtractedValue | None,
    minimum_investment: ExtractedValue | None,
    minimum_holding_period: ExtractedValue | None,
) -> list[Constraint]:
    """Parse share class flat fields into typed constraints.

    Args:
        currency_hedged: Boolean indicating if hedged
        hedging_details: Raw text like "Hedged to EUR using FX forwards"
        investor_restrictions: Raw text like "Institutional only; min EUR 1M"
        minimum_investment: Raw text like "EUR 5,000"
        minimum_holding_period: Raw text like "30 days"

    Returns:
        List of typed Constraint objects.
    """
    constraints = []

    if currency_hedged or hedging_details:
        constraints.extend(_parse_hedging(currency_hedged, hedging_details))

    if investor_restrictions and not investor_restrictions.is_not_found:
        constraints.extend(_parse_investor_eligibility(investor_restrictions))

    if minimum_investment and not minimum_investment.is_not_found:
        constraints.extend(_parse_minimum_investment(minimum_investment))

    if minimum_holding_period and not minimum_holding_period.is_not_found:
        constraints.extend(_parse_holding_period(minimum_holding_period))

    return constraints


# Internal parsers

def _parse_investment_restrictions(extracted_value: ExtractedValue) -> list[Constraint]:
    """Parse investment restrictions into multiple constraints."""
    constraints = []
    text = str(extracted_value.value)

    parts = [p.strip() for p in text.split(";") if p.strip()]

    for part in parts:
        constraint_type, numeric, comparator = _classify_restriction(part)
        constraints.append(Constraint(
            constraint_type=constraint_type,
            rule=part,
            numeric_value=numeric,
            unit="percent" if numeric else None,
            comparator=comparator,
            source=extracted_value,
        ))

    return constraints


def _parse_leverage_policy(extracted_value: ExtractedValue) -> list[Constraint]:
    """Parse leverage policy into constraints."""
    constraints = []
    text = str(extracted_value.value)

    numeric = _extract_percentage(text)

    if "var" in text.lower() or "value at risk" in text.lower():
        constraints.append(Constraint(
            constraint_type="var_limit",
            rule=text,
            numeric_value=numeric,
            unit="percent" if numeric else None,
            comparator="max" if numeric else None,
            source=extracted_value,
        ))
    else:
        constraints.append(Constraint(
            constraint_type="leverage_limit",
            rule=text,
            numeric_value=numeric,
            unit="percent" if numeric else None,
            comparator="max" if numeric else None,
            source=extracted_value,
        ))

    return constraints


def _parse_derivatives_usage(extracted_value: ExtractedValue) -> list[Constraint]:
    """Parse derivatives usage into constraints."""
    text = str(extracted_value.value)

    if "limit" in text.lower() or "%" in text:
        constraint_type = "derivative_limit"
        numeric = _extract_percentage(text)
    else:
        constraint_type = "derivative_policy"
        numeric = None

    return [Constraint(
        constraint_type=constraint_type,
        rule=text,
        numeric_value=numeric,
        unit="percent" if numeric else None,
        comparator="max" if numeric else None,
        source=extracted_value,
    )]


def _parse_hedging(
    currency_hedged: ExtractedValue | None,
    hedging_details: ExtractedValue | None,
) -> list[Constraint]:
    """Parse hedging into constraints."""
    constraints = []

    # If explicitly not hedged, no constraint
    if currency_hedged and not currency_hedged.is_not_found:
        if currency_hedged.value in (False, "false", "False"):
            return []

    # If hedging details exist, create constraint
    if hedging_details and not hedging_details.is_not_found:
        text = str(hedging_details.value)
        target = _extract_hedge_ratio(text)

        constraints.append(Constraint(
            constraint_type="hedging_policy",
            rule=text,
            numeric_value=target,
            unit="percent" if target else None,
            comparator="exact" if target else None,
            source=hedging_details,
        ))
    elif currency_hedged and currency_hedged.value is True:
        # Hedged but no details
        constraints.append(Constraint(
            constraint_type="hedging_policy",
            rule="Currency hedged (details not specified)",
            source=currency_hedged,
        ))

    return constraints


def _parse_investor_eligibility(extracted_value: ExtractedValue) -> list[Constraint]:
    """Parse investor eligibility into constraints."""
    return [Constraint(
        constraint_type="investor_eligibility",
        rule=str(extracted_value.value),
        source=extracted_value,
    )]


def _parse_minimum_investment(extracted_value: ExtractedValue) -> list[Constraint]:
    """Parse minimum investment into constraint."""
    text = str(extracted_value.value)
    amount, currency = _extract_amount(text)

    return [Constraint(
        constraint_type="minimum_investment",
        rule=text,
        numeric_value=amount,
        unit=currency or "currency",
        comparator="min" if amount else None,
        source=extracted_value,
    )]


def _parse_holding_period(extracted_value: ExtractedValue) -> list[Constraint]:
    """Parse holding period into constraint."""
    text = str(extracted_value.value)
    days = _extract_days(text)

    return [Constraint(
        constraint_type="holding_period",
        rule=text,
        numeric_value=float(days) if days else None,
        unit="days" if days else None,
        comparator="min" if days else None,
        source=extracted_value,
    )]


# Helper functions

def _classify_restriction(text: str) -> tuple[str, float | None, str | None]:
    """Classify a restriction clause into constraint type with numeric value.

    Uses RESTRICTION_KEYWORDS mapping to determine constraint type.
    Detects min/max comparator from keywords like "max 10%" or "at least 70%".

    Returns:
        (constraint_type, numeric_value, comparator)
    """
    text_lower = text.lower()
    numeric = _extract_percentage(text)

    comparator = _detect_comparator(text_lower)

    # Match against keyword patterns (first match wins)
    for constraint_type, keywords in RESTRICTION_KEYWORDS.items():
        if any(keyword in text_lower for keyword in keywords):
            # Concentration limits default to "max" if no explicit comparator
            if constraint_type == "concentration_limit" and comparator is None:
                comparator = "max"
            return constraint_type, numeric, comparator

    # Default to asset_limit
    return "asset_limit", numeric, comparator


def _detect_comparator(text_lower: str) -> str | None:
    """Detect min/max comparator from text."""
    for keyword in MAX_KEYWORDS:
        if keyword in text_lower:
            return "max"
    for keyword in MIN_KEYWORDS:
        if keyword in text_lower:
            return "min"
    return None


def _extract_percentage(text: str) -> float | None:
    """Extract percentage from text.

    Handles: "10%", "10 %", "10 percent", "up to 10%"
    """
    # Try pattern like "10%" or "10 %"
    match = re.search(r"(\d+(?:\.\d+)?)\s*%", text)
    if match:
        return float(match.group(1))

    # Try pattern like "10 percent"
    match = re.search(r"(\d+(?:\.\d+)?)\s*percent", text.lower())
    if match:
        return float(match.group(1))

    return None


def _extract_hedge_ratio(text: str) -> float | None:
    """Extract hedge ratio from hedging details.

    Handles: "100% hedge ratio", "targeting 100%", "fully hedged"
    """
    if "fully hedged" in text.lower() or "full hedge" in text.lower():
        return 100.0

    match = re.search(r"(\d+(?:\.\d+)?)\s*%", text)
    if match:
        return float(match.group(1))

    return None


def _extract_amount(text: str) -> tuple[float | None, str | None]:
    """Extract currency amount.

    Handles: "EUR 5,000", "5000 USD", "$1,000,000"
    """
    # Pattern: currency + amount
    match = re.search(r"(EUR|USD|GBP|CHF|\$|€|£)\s*([\d,]+(?:\.\d+)?)", text)
    if match:
        currency = match.group(1)
        amount = float(match.group(2).replace(",", ""))
        currency_map = {"$": "USD", "€": "EUR", "£": "GBP"}
        currency = currency_map.get(currency, currency)
        return amount, currency

    # Pattern: amount + currency
    match = re.search(r"([\d,]+(?:\.\d+)?)\s*(EUR|USD|GBP|CHF)", text)
    if match:
        amount = float(match.group(1).replace(",", ""))
        return amount, match.group(2)

    return None, None


def _extract_days(text: str) -> int | None:
    """Extract days from holding period.

    Handles: "30 days", "30-day", "one month"
    """
    # Try numeric days
    match = re.search(r"(\d+)\s*(?:day|days)", text.lower())
    if match:
        return int(match.group(1))

    # Try word forms
    word_to_days = {
        "one month": 30,
        "1 month": 30,
        "three months": 90,
        "3 months": 90,
        "six months": 180,
        "6 months": 180,
        "one year": 365,
        "1 year": 365,
    }
    text_lower = text.lower()
    for word, days in word_to_days.items():
        if word in text_lower:
            return days

    return None
