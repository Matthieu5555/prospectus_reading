"""Constraint models for binding compliance rules.

Constraints represent binding rules that govern fund behavior:
- Investment limits (asset allocation, concentration)
- Leverage limits (gross exposure, VaR)
- Derivative policies (hedging only, EPM)
- Share class restrictions (hedging, eligibility)

Each constraint carries full provenance via ExtractedValue.
"""

from enum import Enum
from typing import Literal, get_args
from pydantic import BaseModel, Field, computed_field

from extractor.pydantic_models.provenance import ExtractedValue


# Constraint Field Constants

# Fields containing fund-level constraints (investment limits, leverage, derivatives)
CONSTRAINT_FIELDS = ["investment_restrictions", "leverage_policy", "derivatives_usage"]

# Human-readable descriptions for constraint fields (used in prompts)
CONSTRAINT_FIELD_DESCRIPTIONS = {
    "investment_restrictions": "investment limits and prohibitions",
    "leverage_policy": "leverage limits",
    "derivatives_usage": "derivatives usage policy",
}


class BindingStatus(str, Enum):
    """How binding a constraint is - critical for PMS compliance rules.

    This is inspired by GraphRAG's claim verification status, adapted
    for prospectus constraints.
    """

    BINDING = "binding"  # "shall not exceed", "must not" - hard limit, auditable
    GUIDANCE = "guidance"  # "will generally not", "normally" - soft guidance
    DISCRETIONARY = "discretionary"  # "may at its discretion" - not really a constraint
    UNKNOWN = "unknown"  # Couldn't determine binding status


# Constraint types at fund level
FundConstraintType = Literal[
    # Asset allocation limits
    "asset_limit",          # "Min 70% investment grade bonds"
    "concentration_limit",  # "Max 10% single issuer"
    "geographic_limit",     # "Min 80% EU securities"
    "sector_limit",         # "Max 30% financials"
    "liquidity_limit",      # "Min 10% liquid assets"

    # Risk limits
    "leverage_limit",       # "Gross exposure max 200%"
    "var_limit",            # "99% 20-day VaR max 20%"
    "duration_limit",       # "Duration 3-7 years"

    # Derivative policies
    "derivative_policy",    # "Hedging only, no speculation"
    "derivative_limit",     # "Max 100% NAV in derivatives"

    # Other
    "esg_policy",           # "ESG integration required"
    "exclusion_policy",     # "No tobacco, controversial weapons"
]


# Constraint types at share class level
ShareClassConstraintType = Literal[
    # Currency hedging
    "hedging_policy",       # "Hedged to EUR via FX forwards"
    "hedging_target",       # "Target 100% hedge ratio"

    # Investor restrictions
    "investor_eligibility", # "Institutional only"
    "minimum_investment",   # "Min EUR 1,000,000 initial"
    "minimum_subsequent",   # "Min EUR 10,000 subsequent"
    "holding_period",       # "30 day minimum holding"

    # Geographic restrictions
    "distribution_country", # "Not available in US"
]


# All predefined constraint types (for is_predefined check)
PREDEFINED_CONSTRAINT_TYPES: set[str] = (
    set(get_args(FundConstraintType)) | set(get_args(ShareClassConstraintType))
)


class Constraint(BaseModel):
    """A binding rule with full provenance.

    Constraints are extracted compliance rules that must be tracked.
    Each carries:
    - Type: What kind of constraint
    - Rule: Human-readable description
    - Binding status: How enforceable (BINDING, GUIDANCE, DISCRETIONARY)
    - Numeric value: If quantifiable (10.0 for "max 10%")
    - Source: Full provenance (page, quote, rationale, confidence)

    Examples:
        Constraint(
            constraint_type="concentration_limit",
            rule="Maximum 10% in any single issuer",
            binding_status=BindingStatus.BINDING,
            numeric_value=10.0,  # Stored as literal percentage value
            source=ExtractedValue(value="10%", source_page=45, ...)
        )

        Constraint(
            constraint_type="hedging_policy",
            rule="Hedged to EUR using FX forwards, targeting 100% hedge ratio",
            binding_status=BindingStatus.GUIDANCE,
            numeric_value=100.0,  # Stored as literal percentage value
            source=ExtractedValue(value="100% hedge ratio", source_page=67, ...)
        )
    """

    constraint_type: str = Field(
        description="Type of constraint - one of FundConstraintType or ShareClassConstraintType"
    )

    rule: str = Field(
        description="Human-readable constraint rule, e.g. 'Maximum 10% in any single issuer'"
    )

    binding_status: BindingStatus = Field(
        default=BindingStatus.UNKNOWN,
        description="How binding this constraint is - BINDING (auditable), GUIDANCE (soft), DISCRETIONARY (flexible)"
    )

    numeric_value: float | None = Field(
        default=None,
        description="Quantified value if applicable, e.g. 10.0 for '10%', 200.0 for '200%'"
    )

    unit: str | None = Field(
        default=None,
        description="Unit for numeric value: 'percent', 'days', 'currency', 'years'"
    )

    comparator: Literal["min", "max", "exact", "range"] | None = Field(
        default=None,
        description="How to interpret numeric_value: min=at least, max=at most, exact=exactly"
    )

    secondary_value: float | None = Field(
        default=None,
        description="For range constraints, the upper bound (numeric_value is lower)"
    )

    source: ExtractedValue = Field(
        description="Full provenance - page, quote, rationale, confidence"
    )

    @computed_field
    @property
    def is_predefined(self) -> bool:
        """True if constraint_type is from our predefined schema, False if LLM-discovered."""
        return self.constraint_type in PREDEFINED_CONSTRAINT_TYPES


class ConstraintSet(BaseModel):
    """Collection of constraints with inheritance semantics.

    At fund level: Constraints apply to all share classes.
    At share class level: Constraints override or extend fund-level.

    Empty list means "inherits all from parent" (for share classes).
    """

    constraints: list[Constraint] = Field(
        default_factory=list,
        description="List of binding constraints"
    )

    def by_type(self, constraint_type: str) -> list[Constraint]:
        """Get all constraints of a specific type."""
        return [c for c in self.constraints if c.constraint_type == constraint_type]

    def has_type(self, constraint_type: str) -> bool:
        """Check if any constraint of this type exists."""
        return any(c.constraint_type == constraint_type for c in self.constraints)

    @property
    def investment_limits(self) -> list[Constraint]:
        """Get all investment limit constraints."""
        limit_types = {"asset_limit", "concentration_limit", "geographic_limit",
                       "sector_limit", "liquidity_limit"}
        return [c for c in self.constraints if c.constraint_type in limit_types]

    @property
    def risk_limits(self) -> list[Constraint]:
        """Get all risk limit constraints."""
        risk_types = {"leverage_limit", "var_limit", "duration_limit"}
        return [c for c in self.constraints if c.constraint_type in risk_types]

    @property
    def derivative_constraints(self) -> list[Constraint]:
        """Get all derivative-related constraints."""
        deriv_types = {"derivative_policy", "derivative_limit"}
        return [c for c in self.constraints if c.constraint_type in deriv_types]


# Helper functions for constraint creation

def asset_limit(
    rule: str,
    percentage: float,
    comparator: Literal["min", "max"] = "max",
    source: ExtractedValue | None = None,
) -> Constraint:
    """Create an asset allocation limit constraint."""
    return Constraint(
        constraint_type="asset_limit",
        rule=rule,
        numeric_value=percentage,
        unit="percent",
        comparator=comparator,
        source=source or ExtractedValue.extraction_failed(rule),
    )


def concentration_limit(
    rule: str,
    percentage: float,
    source: ExtractedValue | None = None,
) -> Constraint:
    """Create a concentration limit constraint (always max)."""
    return Constraint(
        constraint_type="concentration_limit",
        rule=rule,
        numeric_value=percentage,
        unit="percent",
        comparator="max",
        source=source or ExtractedValue.extraction_failed(rule),
    )


def leverage_limit(
    rule: str,
    percentage: float,
    source: ExtractedValue | None = None,
) -> Constraint:
    """Create a leverage limit constraint (always max)."""
    return Constraint(
        constraint_type="leverage_limit",
        rule=rule,
        numeric_value=percentage,
        unit="percent",
        comparator="max",
        source=source or ExtractedValue.extraction_failed(rule),
    )


def hedging_policy(
    rule: str,
    target_ratio: float | None = None,
    source: ExtractedValue | None = None,
) -> Constraint:
    """Create a hedging policy constraint."""
    return Constraint(
        constraint_type="hedging_policy",
        rule=rule,
        numeric_value=target_ratio,
        unit="percent" if target_ratio else None,
        comparator="exact" if target_ratio else None,
        source=source or ExtractedValue.extraction_failed(rule),
    )


def investor_eligibility(
    rule: str,
    source: ExtractedValue | None = None,
) -> Constraint:
    """Create an investor eligibility constraint."""
    return Constraint(
        constraint_type="investor_eligibility",
        rule=rule,
        source=source or ExtractedValue.extraction_failed(rule),
    )


def minimum_investment_constraint(
    rule: str,
    amount: float,
    currency: str = "EUR",
    source: ExtractedValue | None = None,
) -> Constraint:
    """Create a minimum investment constraint."""
    return Constraint(
        constraint_type="minimum_investment",
        rule=rule,
        numeric_value=amount,
        unit=currency,
        comparator="min",
        source=source or ExtractedValue.extraction_failed(rule),
    )


def holding_period(
    rule: str,
    days: int,
    source: ExtractedValue | None = None,
) -> Constraint:
    """Create a holding period constraint."""
    return Constraint(
        constraint_type="holding_period",
        rule=rule,
        numeric_value=float(days),
        unit="days",
        comparator="min",
        source=source or ExtractedValue.extraction_failed(rule),
    )
