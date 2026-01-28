"""Pydantic schemas for extraction phase (Phase 3) intermediate data.

These models represent raw LLM output during extraction, before
conversion to the final typed models with full provenance.
"""

from typing import Literal, Any
from pydantic import BaseModel, Field


# Type Alias for LLM Output

# Raw LLM output can be in various formats:
# - Legacy: simple values like "1.50%", "EUR", True
# - Provenance: {"value": "1.50%", "source_page": 45, "rationale": "..."}
#
# We use a union type to accept both during extraction.
# The assembly phase then converts to properly typed ExtractedValue.

RawFieldValue = str | int | float | bool | dict[str, Any] | list[Any] | None
"""Type for raw field values from LLM extraction.

Can be:
- Simple primitive (str, int, float, bool)
- Provenance dict with "value" key
- List for multi-value fields
- None for missing fields
"""


# Raw Extraction Models

class RawShareClassData(BaseModel):
    """Raw share class data from LLM extraction.

    Fields accept both legacy format (simple values) and provenance format
    (dict with value, source_page, rationale, etc).

    Attributes:
        name: Share class name/identifier (required, with provenance)
        isin: ISIN code
        currency: Base currency
        currency_hedged: Whether currency hedged
        hedging_details: Hedging description
        distribution_policy: Accumulating or distributing
        dividend_frequency: How often dividends paid
        dividend_dates: Specific dividend dates
        investor_type: Target investor type
        investor_restrictions: Restrictions on investors
        minimum_investment: Minimum initial investment
        minimum_subsequent: Minimum additional investment
        minimum_holding_period: Required holding period
        management_fee: Annual management fee
        performance_fee: Performance-based fee
        entry_fee: Subscription fee
        exit_fee: Redemption fee
        ongoing_charges: Total expense ratio
        dealing_frequency: How often can trade
        dealing_cutoff: Order cutoff time
        valuation_point: NAV calculation time
        settlement_period: Trade settlement days
        listing: Exchange listing info
        launch_date: Share class launch date
    """

    name: RawFieldValue = Field(description="Share class name/identifier with provenance")

    # Core identification
    isin: RawFieldValue = None
    currency: RawFieldValue = None
    currency_hedged: RawFieldValue = None
    hedging_details: RawFieldValue = None

    # Distribution
    distribution_policy: RawFieldValue = None
    dividend_frequency: RawFieldValue = None
    dividend_dates: RawFieldValue = None

    # Investor requirements
    investor_type: RawFieldValue = None
    investor_restrictions: RawFieldValue = None
    minimum_investment: RawFieldValue = None
    minimum_subsequent: RawFieldValue = None
    minimum_holding_period: RawFieldValue = None

    # Fees
    management_fee: RawFieldValue = None
    performance_fee: RawFieldValue = None
    entry_fee: RawFieldValue = None
    exit_fee: RawFieldValue = None
    ongoing_charges: RawFieldValue = None

    # Dealing
    dealing_frequency: RawFieldValue = None
    dealing_cutoff: RawFieldValue = None
    valuation_point: RawFieldValue = None
    settlement_period: RawFieldValue = None

    # Other
    listing: RawFieldValue = None
    launch_date: RawFieldValue = None

    class Config:
        extra = "allow"  # Allow additional fields from LLM


class RawSubFundData(BaseModel):
    """Raw sub-fund data from LLM extraction.

    Fields accept both legacy format (simple values) and provenance format
    (dict with value, source_page, rationale, etc).

    Attributes:
        name: Sub-fund name (required, with provenance)
        investment_objective: Fund's investment objective
        investment_policy: How the fund invests
        benchmark: Reference benchmark
        benchmark_usage: How benchmark is used
        asset_class: Primary asset class
        geographic_focus: Geographic investment focus
        sector_focus: Sector investment focus
        currency_base: Base currency
        risk_profile: Risk level indicator
        inception_date: Fund inception date
        leverage_policy: Leverage/borrowing limits
        derivatives_usage: How derivatives are used
        investment_restrictions: Investment constraints
        share_classes: List of share classes
    """

    name: RawFieldValue = Field(description="Sub-fund name with provenance")

    # Investment details
    investment_objective: RawFieldValue = None
    investment_policy: RawFieldValue = None
    benchmark: RawFieldValue = None
    benchmark_usage: RawFieldValue = None

    # Classification
    asset_class: RawFieldValue = None
    geographic_focus: RawFieldValue = None
    sector_focus: RawFieldValue = None

    # Operational
    currency_base: RawFieldValue = None
    risk_profile: RawFieldValue = None
    inception_date: RawFieldValue = None

    # Binding constraints (CRITICAL for PMS)
    leverage_policy: RawFieldValue = None
    derivatives_usage: RawFieldValue = None
    investment_restrictions: RawFieldValue = None

    # Share classes
    share_classes: list[RawShareClassData] = Field(default_factory=list)

    class Config:
        extra = "allow"  # Allow additional fields from LLM


class RawUmbrellaData(BaseModel):
    """Raw umbrella data from LLM extraction.

    Fields accept both legacy format (simple values) and provenance format
    (dict with value, source_page, rationale, etc).

    Attributes:
        name: Umbrella fund name
        legal_form: Legal structure (e.g., SICAV)
        product_type: Product type (e.g., UCITS)
        domicile: Country of domicile
        inception_date: Umbrella inception date
        management_company: Management company name
        depositary: Depositary/custodian name
        auditor: Auditor name
        regulator: Regulatory authority
        registered_office: Registered address
    """

    name: RawFieldValue = None
    legal_form: RawFieldValue = None
    product_type: RawFieldValue = None
    domicile: RawFieldValue = None
    inception_date: RawFieldValue = None
    management_company: RawFieldValue = None
    depositary: RawFieldValue = None
    auditor: RawFieldValue = None
    regulator: RawFieldValue = None
    registered_office: RawFieldValue = None

    class Config:
        extra = "allow"


class ExtractionPhaseResult(BaseModel):
    """Result from a single extraction operation.

    Wraps the raw data with metadata about the extraction.

    Attributes:
        success: Whether extraction succeeded
        entity_type: Type of entity (umbrella, subfund, shareclass)
        entity_name: Name of the entity
        data: The extracted data
        error_message: Error details if failed
        pages_read: Pages that were read
        search_performed: Whether search was used
    """

    success: bool = True
    entity_type: Literal["umbrella", "subfund", "shareclass"]
    entity_name: str
    data: RawSubFundData | RawUmbrellaData | RawShareClassData | None = None
    error_message: str | None = None
    pages_read: list[int] = Field(default_factory=list)
    search_performed: bool = False

    @classmethod
    def ok(
        cls,
        entity_type: Literal["umbrella", "subfund", "shareclass"],
        entity_name: str,
        data: RawSubFundData | RawUmbrellaData | RawShareClassData,
        pages: list[int] | None = None,
    ) -> "ExtractionPhaseResult":
        """Create successful result."""
        return cls(
            success=True,
            entity_type=entity_type,
            entity_name=entity_name,
            data=data,
            pages_read=pages or [],
        )

    @classmethod
    def fail(
        cls,
        entity_type: Literal["umbrella", "subfund", "shareclass"],
        entity_name: str,
        error: str,
    ) -> "ExtractionPhaseResult":
        """Create failed result."""
        return cls(
            success=False,
            entity_type=entity_type,
            entity_name=entity_name,
            error_message=error,
        )
