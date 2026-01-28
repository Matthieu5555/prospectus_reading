"""Recipe models for field-level extraction strategies.

Recipes specify HOW to extract each field for each fund.
Instead of generic "read these pages and extract everything",
recipes dispatch to the right extraction method per field:
- Table lookup for ISINs/fees in consolidated tables
- Text extraction for investment objectives in fund sections
- Cross-reference following for data in appendices
"""

from typing import Literal
from pydantic import BaseModel, Field


# Source types determine which extraction path to use
SourceType = Literal["text_extraction", "table_lookup", "cross_reference", "inherited"]


class TableLookupSource(BaseModel):
    """Instructions for looking up a value in a parsed table.

    Used when data is in a consolidated table (ISIN tables, fee tables).
    The TableExtractor parses the table once and caches it - subsequent
    lookups are instant dictionary lookups.

    Attributes:
        table_type: Type of table (isin, fee, share_class).
        table_pages: Pages containing the table.
        lookup_column: Column to match on (e.g., "Fund Name").
        lookup_value: Value to match (usually the fund name).
        target_columns: Which columns to extract from the matched row.
    """

    table_type: str = Field(description="Type of table: isin, fee, share_class")
    table_pages: list[int] = Field(description="Pages containing the table")
    lookup_column: str = Field(description="Column to match fund name against")
    lookup_value: str = Field(description="Value to match (usually fund name)")
    target_columns: list[str] = Field(
        default_factory=list,
        description="Columns to extract from matched row"
    )


class TextExtractionSource(BaseModel):
    """Instructions for LLM-based text extraction.

    Used when data is embedded in prose or small per-fund tables.
    Falls back to this when table lookup isn't appropriate.

    Attributes:
        pages: Pages to read for extraction.
        field_hint: What the LLM should look for.
        context: Additional context to help extraction.
    """

    pages: list[int] = Field(description="Pages to read for extraction")
    field_hint: str = Field(
        default="",
        description="Helps LLM know what to look for"
    )
    context: str = Field(
        default="",
        description="Additional context (e.g., expected format)"
    )


class CrossReferenceSource(BaseModel):
    """Instructions for following a cross-reference.

    Used when the fund section says "see Appendix E for fee details".
    The reference has been resolved to actual page numbers.

    Attributes:
        reference_text: Original reference text for debugging.
        target_pages: Resolved page numbers to read.
        extraction_type: How to extract from target (text or table).
    """

    reference_text: str = Field(description="Original reference text")
    target_pages: list[int] = Field(description="Resolved target pages")
    extraction_type: Literal["text", "table"] = Field(
        default="text",
        description="How to extract from target pages"
    )


class InheritedSource(BaseModel):
    """Instructions for inheriting from umbrella level.

    Used when a field applies to all funds (e.g., depositary, auditor).
    The value is copied from umbrella extraction.

    Attributes:
        field_name: Name of umbrella field to inherit from.
    """

    field_name: str = Field(description="Umbrella field to inherit from")


class FieldStrategy(BaseModel):
    """Strategy for extracting a single field.

    Each field gets its own strategy based on DocumentLogic analysis.
    The extraction phase dispatches to different code paths based on source_type.

    Attributes:
        field_name: Name of the field to extract.
        source_type: Which extraction method to use.
        source: Source-specific instructions (depends on source_type).
        required: Whether this field is required (affects error handling).
        fallback_to_text: If table lookup fails, try text extraction.
    """

    field_name: str = Field(description="Field to extract: isin, management_fee, etc.")
    source_type: SourceType = Field(description="Extraction method to use")
    source: TableLookupSource | TextExtractionSource | CrossReferenceSource | InheritedSource | None = Field(
        default=None,
        description="Source-specific extraction instructions"
    )
    required: bool = Field(
        default=False,
        description="Whether extraction failure is an error"
    )
    fallback_to_text: bool = Field(
        default=True,
        description="Try text extraction if primary method fails"
    )


class FundExtractionRecipe(BaseModel):
    """Complete extraction recipe for one fund.

    Replaces FundExtractionTask with explicit per-field strategies.
    Instead of "read these pages and extract everything", this says
    exactly how to get each piece of data.

    Attributes:
        fund_name: Exact fund name.
        dedicated_pages: Pages with fund-specific content.
        field_strategies: Per-field extraction instructions.
        share_class_strategy: How to extract share classes.
        notes: Debugging notes from planner.
    """

    fund_name: str = Field(description="Exact fund name to extract")
    dedicated_pages: list[int] = Field(
        default_factory=list,
        description="Pages with this fund's dedicated section"
    )
    field_strategies: list[FieldStrategy] = Field(
        default_factory=list,
        description="Per-field extraction instructions"
    )
    share_class_strategy: FieldStrategy | None = Field(
        default=None,
        description="Strategy for extracting share classes"
    )
    notes: list[str] = Field(
        default_factory=list,
        description="Debugging notes from recipe generation"
    )

    def get_strategy(self, field_name: str) -> FieldStrategy | None:
        """Get the strategy for a specific field."""
        for strategy in self.field_strategies:
            if strategy.field_name == field_name:
                return strategy
        return None

    def get_table_lookup_fields(self) -> list[FieldStrategy]:
        """Get all fields that use table lookup."""
        return [s for s in self.field_strategies if s.source_type == "table_lookup"]

    def get_text_extraction_fields(self) -> list[FieldStrategy]:
        """Get all fields that use text extraction."""
        return [s for s in self.field_strategies if s.source_type == "text_extraction"]


class ExtractionRecipeSet(BaseModel):
    """Complete set of recipes for a document.

    Contains recipes for all funds plus document-wide settings.

    Attributes:
        umbrella_name: Name of the umbrella fund.
        fund_recipes: Recipe for each fund.
        broadcast_tables: Tables to parse upfront.
    """

    umbrella_name: str = Field(description="Umbrella fund name")
    fund_recipes: list[FundExtractionRecipe] = Field(
        default_factory=list,
        description="Extraction recipe for each fund"
    )
    broadcast_tables: list[TableLookupSource] = Field(
        default_factory=list,
        description="Tables to parse upfront for lookups"
    )

    def get_recipe(self, fund_name: str) -> FundExtractionRecipe | None:
        """Get recipe for a specific fund."""
        for recipe in self.fund_recipes:
            if recipe.fund_name == fund_name:
                return recipe
        return None
