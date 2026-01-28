"""Pydantic models for UCITS prospectus entity extraction output.

These are the final output models representing the extracted entity graph:
- Umbrella: Top-level legal entity
- SubFund: Asset side - the actual portfolio
- ShareClass: Liability side - what investors buy
- ProspectusGraph: Complete extraction result

ALL extracted fields are wrapped in ExtractedValue for full provenance tracking.
"""

from typing import Literal
from pydantic import BaseModel, Field

from extractor.pydantic_models.provenance import ExtractedValue
from extractor.pydantic_models.constraints import Constraint


# Type alias for optional extracted value
Extracted = ExtractedValue | None


# Final Output Models
# All fields wrapped in ExtractedValue for full provenance tracking.

class ShareClass(BaseModel):
    """Liability side - what investors buy.

    All fields are wrapped in ExtractedValue for provenance, including `name`.
    """

    # Identifier - now supports ExtractedValue for provenance tracking
    name: ExtractedValue | str = Field(description="Share class identifier, e.g. 'A (acc) EUR'")

    # Core identifiers
    isin: Extracted = Field(default=None, description="ISIN code, e.g. 'LU1234567890'")
    currency: Extracted = Field(default=None, description="Share class currency, e.g. 'EUR', 'USD'")

    # Hedging
    currency_hedged: Extracted = Field(default=None, description="Whether currency is hedged (true/false)")
    hedging_details: Extracted = Field(
        default=None,
        description="Hedging methodology, e.g. 'Hedged to EUR using FX forwards, targeting 100% hedge ratio'"
    )

    # Distribution
    distribution_policy: Extracted = Field(default=None, description="Accumulating or Distributing")
    dividend_frequency: Extracted = Field(default=None, description="Quarterly, Annually, Monthly, etc.")
    dividend_dates: Extracted = Field(
        default=None,
        description="Specific dividend payment dates, e.g. '15 Mar, 15 Jun, 15 Sep, 15 Dec'"
    )

    # Investor restrictions
    investor_type: Extracted = Field(default=None, description="Retail, Institutional, All")
    investor_restrictions: Extracted = Field(
        default=None,
        description="Specific investor eligibility, e.g. 'Institutional only; minimum EUR 1,000,000'"
    )
    minimum_investment: Extracted = Field(default=None, description="Minimum initial investment amount")
    minimum_subsequent: Extracted = Field(default=None, description="Minimum subsequent investment")
    minimum_holding_period: Extracted = Field(
        default=None,
        description="Required holding period, e.g. '30 days', '90 days'"
    )

    # Fees
    management_fee: Extracted = Field(default=None, description="Annual management fee percentage")
    performance_fee: Extracted = Field(default=None, description="Performance fee if any")
    entry_fee: Extracted = Field(default=None, description="Entry/subscription fee")
    exit_fee: Extracted = Field(default=None, description="Exit/redemption fee")
    ongoing_charges: Extracted = Field(default=None, description="TER/OCF - total ongoing charges")

    # Dealing
    dealing_frequency: Extracted = Field(default=None, description="Daily, Weekly, etc.")
    dealing_cutoff: Extracted = Field(
        default=None,
        description="Order cut-off time, e.g. '12:00 CET T-1', '14:00 Luxembourg time'"
    )
    valuation_point: Extracted = Field(default=None, description="NAV calculation time")
    settlement_period: Extracted = Field(default=None, description="T+1, T+2, T+3, etc.")

    # Other
    listing: Extracted = Field(default=None, description="Stock exchange listing if any")
    launch_date: Extracted = Field(default=None, description="Share class launch date")

    # Constraints (binding rules specific to this share class)
    constraints: list[Constraint] = Field(
        default_factory=list,
        description="Binding constraints specific to this share class (hedging, eligibility, etc.)"
    )


class SubFund(BaseModel):
    """Asset side - the actual portfolio.

    All fields are wrapped in ExtractedValue for provenance, including `name`.
    """

    # Identifier - now supports ExtractedValue for provenance tracking
    name: ExtractedValue | str = Field(description="Sub-fund name, e.g. 'Global Bond Fund'")

    # Investment strategy (lower priority - often marketing fluff)
    investment_objective: Extracted = Field(default=None, description="What the fund aims to achieve")
    investment_policy: Extracted = Field(default=None, description="How the fund invests")

    # Benchmark
    benchmark: Extracted = Field(default=None, description="Reference benchmark index")
    benchmark_usage: Extracted = Field(
        default=None,
        description="Target to outperform, Tracking, Reference only"
    )

    # Classification
    asset_class: Extracted = Field(default=None, description="Equity, Fixed Income, Multi-Asset, Money Market")
    geographic_focus: Extracted = Field(default=None, description="Global, Europe, US, Emerging Markets, etc.")
    sector_focus: Extracted = Field(default=None, description="Technology, Healthcare, Financials, etc.")
    currency_base: Extracted = Field(default=None, description="Base currency of the fund")
    risk_profile: Extracted = Field(default=None, description="SRRI 1-7 risk rating")

    # CRITICAL: Global constraint extractions (parsed into `constraints` list below)
    # These are the LLM's text extractions at fund level, kept for auditability.
    # The `constraints` list below contains the structured/parsed version.
    global_investment_restrictions: Extracted = Field(
        default=None,
        description="Fund-level investment limits - asset class limits, concentration limits, prohibitions"
    )
    global_leverage_policy: Extracted = Field(
        default=None,
        description="Fund-level leverage limits - gross exposure, commitment approach, VaR limits"
    )
    global_derivatives_usage: Extracted = Field(
        default=None,
        description="Fund-level derivatives policy - hedging only, EPM, investment purposes"
    )

    # Parsed constraints (derived from global_* fields above by constraint_parser.py)
    constraints: list[Constraint] = Field(
        default_factory=list,
        description="Structured constraints parsed from global extractions. Each has is_predefined=True/False."
    )

    # Dates
    inception_date: Extracted = Field(default=None, description="Fund inception date")

    # Nested entities
    share_classes: list[ShareClass] = Field(default_factory=list)


class Umbrella(BaseModel):
    """Top-level legal entity.

    All fields are wrapped in ExtractedValue for provenance, including `name`.
    """

    # Identifier - now supports ExtractedValue for provenance tracking
    name: ExtractedValue | str = Field(description="Official umbrella fund name")

    # Legal structure
    legal_form: Extracted = Field(default=None, description="SICAV, FCP, OEIC, Unit Trust")
    product_type: Extracted = Field(default=None, description="UCITS, UCI, AIF")
    domicile: Extracted = Field(default=None, description="Luxembourg, Ireland, France, etc.")

    # Dates
    inception_date: Extracted = Field(default=None, description="Umbrella inception date")

    # Service providers
    management_company: Extracted = Field(default=None, description="Name of management company")
    depositary: Extracted = Field(default=None, description="Name of depositary/custodian bank")
    auditor: Extracted = Field(default=None, description="Name of auditor")
    regulator: Extracted = Field(default=None, description="Regulatory authority")
    registered_office: Extracted = Field(default=None, description="Registered office address")

    # Umbrella-level constraints (apply to ALL sub-funds)
    constraints: list[Constraint] = Field(
        default_factory=list,
        description="Investment restrictions binding all sub-funds (borrowing limits, short sale rules, diversification requirements, etc.)"
    )


class ExternalDocReference(BaseModel):
    """Reference to information in an external document.

    When the prospectus says "See KIID for ISINs" or "Refer to Annual Report
    for performance data", we capture that here for traceability.
    """

    document_name: str = Field(description="External document name, e.g. 'KIID', 'Annual Report', 'Supplement'")
    field_name: str = Field(description="Which field is referenced externally, e.g. 'isin', 'performance_fee'")
    entity_name: str | None = Field(default=None, description="Specific fund/share class, or None if applies to all")
    source_page: int = Field(description="Page where the reference was found")
    source_quote: str = Field(description="Verbatim text of the reference")


class DiscoveredField(BaseModel):
    """A field discovered by the LLM that's not in our standard schema.

    These are PMS-relevant fields the LLM identifies beyond what we asked for.
    Used to inform schema evolution and capture valuable data we didn't anticipate.
    """

    field_name: str = Field(description="Suggested field name, e.g. 'swing_pricing_threshold'")
    value: str = Field(description="The extracted value")
    category: Literal[
        "trade_execution",
        "risk_management",
        "compliance",
        "portfolio_construction",
        "operational",
        "other"
    ] = Field(description="Category for PMS use case")
    entity_name: str | None = Field(default=None, description="Fund/share class this applies to, or None if umbrella-level")
    source_page: int | None = Field(default=None, description="Page where found")
    source_quote: str | None = Field(default=None, description="Verbatim text")
    rationale: str | None = Field(default=None, description="Why this field is valuable")
    pms_use_case: str | None = Field(default=None, description="How a PMS would use this data")


class SchemaSuggestion(BaseModel):
    """A suggestion to add a field to our standard schema.

    When the LLM finds the same bonus field across many funds, it suggests
    promoting it to a standard schema field.
    """

    suggested_field: str = Field(description="Field name to add")
    suggested_location: Literal["Umbrella", "SubFund", "ShareClass"] = Field(
        description="Where in the schema to add it"
    )
    rationale: str = Field(description="Why this should be a standard field")
    occurrence_count: int = Field(default=1, description="How many times this was found")


class DocumentStructure(BaseModel):
    """Document structure discovered during exploration.

    This captures the LLM's understanding of how the prospectus is organized,
    which can be valuable for auditing and understanding extraction context.
    """

    toc_pages: list[int] = Field(
        default_factory=list,
        description="Pages containing table of contents or fund list"
    )
    umbrella_info_pages: list[int] = Field(
        default_factory=list,
        description="Pages with umbrella-level info (legal entity, depositary, etc.)"
    )
    fund_dedicated_sections: dict[str, list[int]] = Field(
        default_factory=dict,
        description="Map of fund name to dedicated section pages"
    )


class TableLocation(BaseModel):
    """A table discovered in the document."""

    table_type: str = Field(description="Type: isin, fee, share_class, fund_list, other")
    page_start: int
    page_end: int
    columns: list[str] = Field(default_factory=list, description="Column headers if visible")
    notes: str = Field(default="", description="Additional observations about this table")


class ExplorationSummary(BaseModel):
    """Summary of document exploration - what the LLM learned about document structure.

    This preserves valuable information from the exploration phase that helps
    understand how extraction decisions were made.
    """

    # Document structure
    document_structure: DocumentStructure = Field(
        default_factory=DocumentStructure,
        description="How the document is organized"
    )

    # Tables found
    tables_discovered: list[TableLocation] = Field(
        default_factory=list,
        description="Structured tables found during exploration"
    )

    # Cross-references found
    cross_references: list[str] = Field(
        default_factory=list,
        description="References to other sections found (e.g., 'See Appendix E for ISINs')"
    )

    # Explorer observations
    observations: list[str] = Field(
        default_factory=list,
        description="Free-form notes from explorers about document structure"
    )

    # Coverage
    pages_explored: list[tuple[int, int]] = Field(
        default_factory=list,
        description="Page ranges covered by explorers"
    )


class AgentCostBreakdown(BaseModel):
    """Cost breakdown for a single agent type."""

    calls: int = Field(description="Number of API calls")
    prompt_tokens: int = Field(description="Total input tokens")
    completion_tokens: int = Field(description="Total output tokens")
    cost_usd: float = Field(description="Cost in USD")


class CostSummary(BaseModel):
    """Complete cost/token tracking for the extraction run.

    This enables operational monitoring, cost estimation, and anomaly detection.
    """

    total_calls: int = Field(description="Total API calls across all agents")
    total_tokens: int = Field(description="Total tokens (prompt + completion)")
    prompt_tokens: int = Field(description="Total input tokens")
    completion_tokens: int = Field(description="Total output tokens")
    total_cost_usd: float = Field(description="Total cost in USD")
    by_agent: dict[str, AgentCostBreakdown] = Field(
        default_factory=dict,
        description="Cost breakdown by agent type (explorer, planner, extractor, critic)"
    )


class UnresolvedQuestion(BaseModel):
    """A question that agents couldn't resolve during extraction.

    Preserves what the system tried to find but couldn't, enabling:
    - Audit trail for NOT_FOUND fields
    - Debugging extraction failures
    - Identifying document gaps
    """

    question: str = Field(description="The question, e.g. 'Where are dividend dates for Fund X?'")
    field_name: str = Field(description="Field this relates to")
    entity_name: str = Field(description="Fund/share class this is about")
    priority: Literal["high", "medium", "low"] = Field(description="Question priority")
    pages_searched: list[int] = Field(default_factory=list, description="Pages that were searched")
    source_agent: str = Field(description="Agent that posted the question")


class FundPageAssignment(BaseModel):
    """Pages assigned to a specific fund for extraction."""

    fund_name: str = Field(description="Fund name")
    dedicated_pages: list[int] = Field(default_factory=list, description="Pages with fund's dedicated section")
    isin_lookup_pages: list[int] = Field(default_factory=list, description="Pages to look up ISINs")
    fee_lookup_pages: list[int] = Field(default_factory=list, description="Pages to look up fees")


class ExtractionPlanSummary(BaseModel):
    """Summary of the extraction plan - which pages were assigned to each fund.

    Enables understanding of extraction coverage and debugging missing fields.
    """

    umbrella_pages: list[int] = Field(default_factory=list, description="Pages searched for umbrella info")
    fund_assignments: list[FundPageAssignment] = Field(
        default_factory=list,
        description="Page assignments for each fund"
    )
    broadcast_tables: list[dict] = Field(
        default_factory=list,
        description="Tables extracted once and applied to all funds"
    )
    parallel_safe: bool = Field(default=True, description="Whether funds were extracted in parallel")


class ExtractionMetadata(BaseModel):
    """Metadata about the extraction process."""

    source_file: str
    total_pages: int
    extraction_date: str
    agent_notes: list[str] = Field(default_factory=list)
    # New fields for provenance tracking
    extraction_version: str = Field(default="1.0", description="Pipeline version used")
    total_extractions: int = Field(default=0, description="Total ExtractedValue instances")
    high_confidence_count: int = Field(default=0, description="Extractions with confidence >= 0.8")
    low_confidence_count: int = Field(default=0, description="Extractions with confidence < 0.5")
    not_found_count: int = Field(default=0, description="Fields that were NOT_FOUND")
    # Cost tracking
    cost_summary: CostSummary | None = Field(default=None, description="Token and cost tracking")


class ProspectusGraph(BaseModel):
    """Complete extracted entity graph from a prospectus."""

    umbrella: Umbrella
    sub_funds: list[SubFund] = Field(default_factory=list)
    metadata: ExtractionMetadata

    # Exploration summary - what the LLM learned about document structure
    exploration: ExplorationSummary = Field(
        default_factory=ExplorationSummary,
        description="Summary of document exploration - tables found, page ranges, observations"
    )

    # Aggregated external references - when prospectus points to other documents
    external_references: list[ExternalDocReference] = Field(
        default_factory=list,
        description="All references to external documents (KIID, Annual Report, etc.) found during extraction"
    )

    # Schema discovery - bonus fields found beyond our standard schema
    discovered_fields: list[DiscoveredField] = Field(
        default_factory=list,
        description="PMS-relevant fields discovered beyond the standard schema"
    )
    schema_suggestions: list[SchemaSuggestion] = Field(
        default_factory=list,
        description="Suggestions for new fields to add to the standard schema"
    )

    # Extraction audit trail
    unresolved_questions: list[UnresolvedQuestion] = Field(
        default_factory=list,
        description="Questions agents couldn't resolve - audit trail for NOT_FOUND fields"
    )
    extraction_plan: ExtractionPlanSummary | None = Field(
        default=None,
        description="Which pages were assigned to each fund - for debugging missing fields"
    )


