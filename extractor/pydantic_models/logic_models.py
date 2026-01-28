"""Document logic models for structural pattern analysis.

These models capture how a specific prospectus is organized,
answering questions like "where are ISINs?" and "where are fees?"
once rather than having each extractor figure it out independently.
"""

from typing import Literal
from pydantic import BaseModel, Field


class ISINStrategy(BaseModel):
    """Strategy for extracting ISINs from this document.

    Determines whether to use table lookup (for consolidated tables),
    text extraction (for inline ISINs), or skip (for external documents).
    """

    location: Literal["consolidated_table", "per_fund_inline", "appendix", "external"] = Field(
        description="Where ISINs are located in this document ('external' means in KIID/other doc)"
    )
    external_doc: str | None = Field(
        default=None,
        description="Name of external document if location='external' (e.g., 'KIID')"
    )
    table_pages: list[int] | None = Field(
        default=None,
        description="Pages containing the ISIN table (if consolidated)"
    )
    lookup_column: str | None = Field(
        default=None,
        description="Column to match fund/share class name against"
    )
    value_column: str = Field(
        default="ISIN",
        description="Column containing the ISIN values"
    )


class FeeStrategy(BaseModel):
    """Strategy for extracting fees from this document.

    Many prospectuses have a master fee table in an appendix
    while fund sections just reference it.
    """

    location: Literal["consolidated_table", "per_fund_inline", "appendix", "external"] = Field(
        description="Where fees are located in this document ('external' means in KIID/other doc)"
    )
    external_doc: str | None = Field(
        default=None,
        description="Name of external document if location='external'"
    )
    table_pages: list[int] | None = Field(
        default=None,
        description="Pages containing the fee table (if consolidated)"
    )
    lookup_column: str | None = Field(
        default=None,
        description="Column to match fund/share class name against"
    )
    fee_columns: dict[str, str] = Field(
        default_factory=dict,
        description="Maps fee type to column name (e.g., 'management_fee' -> 'Management Fee')"
    )


class CrossReferenceResolution(BaseModel):
    """A resolved cross-reference from a fund section to another location.

    When a fund section says "see Appendix E for fee details",
    this captures both the reference text and the resolved target pages.
    """

    reference_text: str = Field(
        description="Original reference text (e.g., 'See Appendix E')"
    )
    target_pages: list[int] = Field(
        description="Resolved page numbers for this reference"
    )
    field_hint: str | None = Field(
        default=None,
        description="What kind of data this reference points to (fee, isin, etc.)"
    )
    source_page: int | None = Field(
        default=None,
        description="Page where the reference was found"
    )


class DocumentLogic(BaseModel):
    """Explicit understanding of how this specific prospectus is organized.

    This is the output of the Document Logic phase, synthesized from
    exploration notes. It becomes the source of truth for how to
    extract data from this document.

    Attributes:
        isin_strategy: How to extract ISINs
        fee_strategy: How to extract fees
        resolved_cross_references: Cross-refs with resolved page targets
        appendix_map: Maps appendix names to page ranges
        section_boundaries: Maps section types to page ranges
        observations: Free-form notes about document structure
    """

    isin_strategy: ISINStrategy = Field(
        description="How to extract ISINs from this document"
    )
    fee_strategy: FeeStrategy = Field(
        description="How to extract fees from this document"
    )
    resolved_cross_references: list[CrossReferenceResolution] = Field(
        default_factory=list,
        description="Cross-references with resolved page targets"
    )
    appendix_map: dict[str, tuple[int, int]] = Field(
        default_factory=dict,
        description="Maps appendix names to (start_page, end_page)"
    )
    section_boundaries: dict[str, tuple[int, int]] = Field(
        default_factory=dict,
        description="Maps section types to (start_page, end_page)"
    )
    observations: list[str] = Field(
        default_factory=list,
        description="Free-form notes about document structure"
    )

    def get_isin_pages(self) -> list[int]:
        """Get pages where ISINs can be found."""
        if self.isin_strategy.table_pages:
            return self.isin_strategy.table_pages
        return []

    def get_fee_pages(self) -> list[int]:
        """Get pages where fees can be found."""
        if self.fee_strategy.table_pages:
            return self.fee_strategy.table_pages
        return []

    def should_use_table_lookup(self, field: str) -> bool:
        """Check if a field should use table lookup vs text extraction."""
        if field == "isin":
            return self.isin_strategy.location == "consolidated_table"
        if field == "fee":
            return self.fee_strategy.location == "consolidated_table"
        return False

    def is_field_external(self, field: str) -> bool:
        """Check if a field is documented in an external document."""
        if field == "isin":
            return self.isin_strategy.location == "external"
        if field == "fee":
            return self.fee_strategy.location == "external"
        return False

    def get_external_doc(self, field: str) -> str | None:
        """Get the external document name for a field, if applicable."""
        if field == "isin" and self.isin_strategy.location == "external":
            return self.isin_strategy.external_doc
        if field == "fee" and self.fee_strategy.location == "external":
            return self.fee_strategy.external_doc
        return None
