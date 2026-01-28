"""Pydantic schemas for planning phase (Phase 2).

These schemas define the extraction plan produced by the planner.
"""

from typing import Literal
from pydantic import BaseModel, Field


class PageLookup(BaseModel):
    """Instructions for looking up data in a table.

    Attributes:
        table_pages: Pages containing the table
        lookup_column: Column to match on (e.g., 'Fund Name')
        lookup_value: Value to match (e.g., 'Global Bond Fund')
    """

    table_pages: list[int] = Field(description="Pages containing the table")
    lookup_column: str = Field(
        default="",
        description="Column to match on (e.g., 'Fund Name')"
    )
    lookup_value: str = Field(
        default="",
        description="Value to match (e.g., 'Global Bond Fund')"
    )


class FundExtractionTask(BaseModel):
    """Task to extract one sub-fund's data.

    Each fund gets a dedicated task with specific pages to read
    and lookup instructions for broadcast tables.

    Attributes:
        fund_name: Exact fund name to extract
        dedicated_pages: Pages with this fund's dedicated section
        isin_lookup: Where to find ISINs for this fund
        fee_lookup: Where to find fees for this fund
        share_class_lookup: Where to find share class details
    """

    fund_name: str = Field(description="Exact fund name")

    dedicated_pages: list[int] = Field(
        default_factory=list,
        description="Pages with this fund's dedicated section"
    )

    isin_lookup: PageLookup | None = Field(
        default=None,
        description="Where to find ISINs for this fund"
    )
    fee_lookup: PageLookup | None = Field(
        default=None,
        description="Where to find fees for this fund"
    )
    share_class_lookup: PageLookup | None = Field(
        default=None,
        description="Where to find share class details"
    )


class BroadcastTable(BaseModel):
    """A table to extract once and distribute to all funds.

    Broadcast tables contain data for multiple funds in a single table,
    e.g., fee schedules or ISIN lists.

    Attributes:
        table_type: Type of data in table (isin, fee, share_class)
        pages: Pages containing the table
        extraction_priority: Lower = extract first
        notes: Extraction hints
    """

    table_type: Literal["isin", "fee", "share_class"] = Field(
        description="Type of data in table"
    )
    pages: list[int] = Field(description="Pages containing the table")
    extraction_priority: int = Field(
        default=1,
        description="Lower = extract first"
    )
    notes: str = Field(default="", description="Extraction hints")


class PlannerOutput(BaseModel):
    """Complete extraction plan from planner.

    This is the main output of the planning phase and guides
    all subsequent extraction work.

    Attributes:
        umbrella_name: Name of the umbrella fund
        total_funds: Total number of sub-funds
        fund_names: Complete deduplicated list of all fund names
        umbrella_pages: Pages with umbrella-level info
        fund_tasks: Extraction task for each fund
        broadcast_tables: Tables to extract once and distribute
        fund_name_variants: Maps variant fund names to canonical form
        parallel_safe: True if funds don't share pages
        observations: Notes for debugging
    """

    umbrella_name: str = Field(description="Name of the umbrella fund")
    total_funds: int = Field(description="Total number of sub-funds")
    fund_names: list[str] = Field(
        description="Complete deduplicated list of all fund names"
    )

    umbrella_pages: list[int] = Field(
        description="Pages with umbrella-level info"
    )

    fund_tasks: list[FundExtractionTask] = Field(
        description="Extraction task for each fund"
    )

    broadcast_tables: list[BroadcastTable] = Field(
        default_factory=list,
        description="Tables to extract once and distribute"
    )

    fund_name_variants: dict[str, str] = Field(
        default_factory=dict,
        description="Maps variant fund names to their canonical form"
    )

    parallel_safe: bool = Field(
        default=True,
        description="True if funds don't share pages (can extract in parallel)"
    )
    observations: list[str] = Field(
        default_factory=list,
        description="Notes for debugging"
    )
