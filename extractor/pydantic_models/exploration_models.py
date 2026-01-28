"""Pydantic schemas for exploration phase (Phase 1).

These schemas define the output from explorers and document structure models.
"""

from typing import Literal
from pydantic import BaseModel, Field


# Basic Exploration Models

class FundMention(BaseModel):
    """A sub-fund mentioned in an explorer's page range.

    Attributes:
        name: Exact fund name as written in document
        page: First page where this fund appears
        has_dedicated_section: True if fund has its own section with details
    """

    name: str = Field(description="Exact fund name as written in document")
    page: int = Field(description="First page where this fund appears")
    has_dedicated_section: bool = Field(
        default=False,
        description="True if fund has its own section with details"
    )


class TableDiscovery(BaseModel):
    """A table discovered during exploration.

    Attributes:
        table_type: Type of data in the table (isin, fee, share_class, fund_list, other)
        page_start: First page of the table
        page_end: Last page of the table
        columns: Column headers if visible
        has_fund_name_column: True if table has a column to look up by fund name
        belongs_to_fund: For per-fund tables, which fund owns this table
        row_count_estimate: Estimated number of data rows
        notes: Additional observations about the table

    Table Ownership Logic:
        - If has_fund_name_column=True → consolidated table, belongs_to_fund=None
        - If has_fund_name_column=False → per-fund table, belongs_to_fund should be set
    """

    table_type: Literal["isin", "fee", "share_class", "fund_list", "other"] = Field(
        description="Type of data in the table"
    )
    page_start: int = Field(description="First page of the table")
    page_end: int = Field(description="Last page of the table")
    columns: list[str] = Field(
        default_factory=list,
        description="Column headers if visible"
    )
    has_fund_name_column: bool = Field(
        default=False,
        description="True if table has a column to look up by fund name (consolidated table)"
    )
    belongs_to_fund: str | None = Field(
        default=None,
        description="For per-fund tables (no fund name column), which fund owns this table"
    )
    row_count_estimate: int | None = Field(
        default=None,
        description="Estimated number of data rows"
    )
    notes: str = Field(default="", description="Additional observations")


class CrossReference(BaseModel):
    """A cross-reference found in the document.

    Cross-references can be:
    - Internal: Points to another section within this document ("See Appendix E")
    - External: Points to a separate document ("See KIID", "Annual Report", URLs)

    Attributes:
        text: The cross-reference text, e.g., 'See Appendix E'
        source_page: Page where cross-reference appears
        target_description: What the reference points to
        is_external: True if points to external document
        external_doc: External document name if applicable
        field_hint: What field this relates to (isin, fee, etc.)
        target_page: Target page number for internal references
    """

    text: str = Field(description="The cross-reference text, e.g., 'See Appendix E'")
    source_page: int = Field(description="Page where cross-reference appears")
    target_description: str = Field(
        default="",
        description="What the reference points to"
    )

    # Classification fields (all have defaults for backward compatibility)
    is_external: bool = Field(
        default=False,
        description="True if points to external document (KIID, Annual Report, URL)"
    )
    external_doc: str | None = Field(
        default=None,
        description="External document name: 'KIID', 'Annual Report', 'Supplement', or URL"
    )
    field_hint: str | None = Field(
        default=None,
        description="What field this relates to: 'isin', 'fee', 'performance', 'risk_profile', etc."
    )
    target_page: int | None = Field(
        default=None,
        description="Target page number for internal references (e.g., 'See page 50' -> 50)"
    )


# Document Inventory Models

class FieldPresence(BaseModel):
    """Tracks where a field type exists in the document.

    Attributes:
        field_name: The field this tracks, e.g., 'isin', 'management_fee'
        pages: Pages where this field exists
        table_type: If found in a table, the table type
        confidence: How confident we are this field exists at these locations
        notes: Additional context about this field's presence
    """

    field_name: str = Field(description="The field this tracks, e.g., 'isin', 'management_fee'")
    pages: list[int] = Field(default_factory=list, description="Pages where this field exists")
    table_type: str | None = Field(
        default=None,
        description="If found in a table: 'isin', 'fee', 'share_class', etc."
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0, le=1.0,
        description="How confident we are this field exists at these locations"
    )
    notes: str = Field(default="", description="Additional context about this field's presence")


class ExternalFieldRef(BaseModel):
    """Tracks a field that is documented in an external document.

    Attributes:
        field_name: The field, e.g., 'isin', 'ongoing_charges'
        external_doc: External document name
        source_page: Page where this reference was found
        source_quote: Verbatim text of the reference
        applies_to: Which entities this applies to ('all' or specific name)
    """

    field_name: str = Field(description="The field, e.g., 'isin', 'ongoing_charges'")
    external_doc: str = Field(description="External document: 'KIID', 'Annual Report', 'Supplement'")
    source_page: int = Field(description="Page where this reference was found")
    source_quote: str = Field(default="", description="Verbatim text of the reference")
    applies_to: str = Field(
        default="all",
        description="Which entities: 'all' or specific fund/share class name"
    )


class TOCEntry(BaseModel):
    """A parsed entry from the table of contents.

    Attributes:
        section_name: Section title as written
        page_number: Page number if listed
        level: Nesting level (1=top, 2=subsection, etc.)
    """

    section_name: str = Field(description="Section title as written")
    page_number: int | None = Field(default=None, description="Page number if listed")
    level: int = Field(default=1, description="Nesting level: 1=top, 2=subsection, etc.")


class ExplicitAbsence(BaseModel):
    """A field explicitly stated as not applicable or absent.

    Attributes:
        field_name: The field, e.g., 'performance_fee'
        reason_quote: Verbatim text, e.g., 'No performance fees apply'
        source_page: Page where stated
    """

    field_name: str = Field(description="The field, e.g., 'performance_fee'")
    reason_quote: str = Field(description="Verbatim text, e.g., 'No performance fees apply'")
    source_page: int | None = Field(default=None, description="Page where stated")


class DocumentInventory(BaseModel):
    """Inventory of what data types exist in this document.

    Built during exploration and used by extractors/resolvers to:
    1. Know where to look for specific fields
    2. Determine appropriate NOT_FOUND reasons
    3. Identify when data is in external documents

    Attributes:
        fields_present: Fields found in document with their locations
        fields_absent: Fields confirmed NOT in this document
        fields_external: Fields documented in external documents
        toc_entries: Parsed table of contents entries
        explicit_absences: Fields explicitly stated as not applicable
    """

    fields_present: list[FieldPresence] = Field(
        default_factory=list,
        description="Fields found in document with their locations"
    )
    fields_absent: list[str] = Field(
        default_factory=list,
        description="Fields confirmed NOT in this document"
    )
    fields_external: list[ExternalFieldRef] = Field(
        default_factory=list,
        description="Fields documented in external documents"
    )
    toc_entries: list[TOCEntry] = Field(
        default_factory=list,
        description="Parsed table of contents entries"
    )
    explicit_absences: list[ExplicitAbsence] = Field(
        default_factory=list,
        description="Fields explicitly stated as not applicable"
    )

    def has_field(self, field_name: str) -> bool:
        """Check if a field is present in the document."""
        return any(fp.field_name == field_name for fp in self.fields_present)

    def is_external(self, field_name: str) -> bool:
        """Check if a field is documented externally."""
        return any(ef.field_name == field_name for ef in self.fields_external)

    def is_absent(self, field_name: str) -> bool:
        """Check if a field is confirmed absent."""
        return field_name in self.fields_absent

    def is_explicitly_absent(self, field_name: str) -> bool:
        """Check if a field is explicitly stated as not applicable."""
        return any(ea.field_name == field_name for ea in self.explicit_absences)

    def get_pages_for_field(self, field_name: str) -> list[int]:
        """Get pages where a field can be found."""
        for fp in self.fields_present:
            if fp.field_name == field_name:
                return fp.pages
        return []

    def get_external_ref(self, field_name: str) -> ExternalFieldRef | None:
        """Get external reference for a field if one exists."""
        for ef in self.fields_external:
            if ef.field_name == field_name:
                return ef
        return None

    def get_not_found_reason(self, field_name: str) -> str | None:
        """Suggest a NOT_FOUND reason based on inventory.

        Returns:
            - 'not_applicable' if field is explicitly absent
            - 'in_external_doc' if field is in external doc
            - 'not_in_document' if field is in fields_absent
            - None if we don't know (should be extraction_failed)
        """
        if self.is_explicitly_absent(field_name):
            return "not_applicable"
        if self.is_external(field_name):
            return "in_external_doc"
        if self.is_absent(field_name):
            return "not_in_document"
        return None


# Document Skeleton (Structure Discovery)

class SectionInfo(BaseModel):
    """A section identified in the document structure.

    Built from native PDF TOC or LLM structure discovery.
    Used for intelligent chunking and cross-reference resolution.

    Attributes:
        name: Section name, e.g., 'Appendix E', 'Fee Schedule'
        page_start: First page of this section (1-indexed)
        page_end: Last page of this section (1-indexed)
        level: Nesting level (1=top, 2=subsection, etc.)
        section_type: Section category
        fund_name: If this is a fund-specific section, the fund name
    """

    name: str = Field(description="Section name, e.g., 'Appendix E', 'Fee Schedule'")
    page_start: int = Field(description="First page of this section (1-indexed)")
    page_end: int = Field(description="Last page of this section (1-indexed)")
    level: int = Field(default=1, description="Nesting level: 1=top, 2=subsection, etc.")
    section_type: str = Field(
        default="other",
        description="Section category: 'toc', 'fund_section', 'appendix', 'table', 'other'"
    )
    fund_name: str | None = Field(
        default=None,
        description="If this is a fund-specific section, the fund name"
    )


class DocumentSkeleton(BaseModel):
    """Global document structure - available to all explorers.

    Built during structure discovery phase (before detailed exploration).
    Enables:
    - Smart chunking at section boundaries
    - Cross-reference resolution
    - Global context for explorers

    Attributes:
        toc_source: How TOC was obtained ('native', 'llm', 'none')
        toc_pages: Pages containing table of contents
        sections: All identified sections with page ranges
        appendix_map: Maps appendix names to (start_page, end_page)
        total_pages: Total pages in document
    """

    toc_source: str = Field(
        default="none",
        description="How TOC was obtained: 'native', 'llm', 'none'"
    )
    toc_pages: list[int] = Field(
        default_factory=list,
        description="Pages containing table of contents"
    )
    sections: list[SectionInfo] = Field(
        default_factory=list,
        description="All identified sections with page ranges"
    )
    appendix_map: dict[str, tuple[int, int]] = Field(
        default_factory=dict,
        description="Maps appendix names to (start_page, end_page)"
    )
    total_pages: int = Field(description="Total pages in document")

    def get_section_for_page(self, page: int) -> SectionInfo | None:
        """Find which section contains a given page."""
        for section in self.sections:
            if section.page_start <= page <= section.page_end:
                return section
        return None

    def resolve_reference(self, ref_text: str, page_index: list["PageContent"] | None = None) -> list[int]:
        """Resolve a cross-reference text to page number(s).

        Handles patterns like:
        - "See page 200" or "pages 100-105" -> extracts page numbers
        - "See Appendix E" -> looks up in appendix_map
        - "See Annex 3" or "Schedule II" -> similar to appendix
        - "See Section 5.2" -> looks up in sections
        - Named sections via page_index lookup

        Args:
            ref_text: The cross-reference text to resolve
            page_index: Optional page index from exploration for named lookups

        Returns:
            List of page numbers if resolved, empty list otherwise.
        """
        import re

        ref_lower = ref_text.lower()

        # Pattern 1: Direct page references - "page 45", "pages 100-105"
        page_match = re.search(r'page[s]?\s*(\d+)(?:\s*[-–to]\s*(\d+))?', ref_lower)
        if page_match:
            start = int(page_match.group(1))
            end = int(page_match.group(2)) if page_match.group(2) else start
            return list(range(start, end + 1))

        # Pattern 2: Appendix/Annex/Schedule references
        appendix_match = re.search(
            r'(appendix|annex|schedule|exhibit)\s*([a-z0-9]+)',
            ref_lower
        )
        if appendix_match:
            doc_type = appendix_match.group(1).title()
            identifier = appendix_match.group(2).upper()

            # Try exact key match first
            appendix_key = f"{doc_type} {identifier}"
            if appendix_key in self.appendix_map:
                start, end = self.appendix_map[appendix_key]
                return list(range(start, end + 1))

            # Try just the identifier
            for key, (start, end) in self.appendix_map.items():
                key_upper = key.upper()
                if identifier in key_upper or key_upper.endswith(identifier):
                    return list(range(start, end + 1))

            # Try matching appendix in sections
            for section in self.sections:
                if section.section_type == "appendix":
                    section_upper = section.name.upper()
                    if identifier in section_upper:
                        return list(range(section.page_start, section.page_end + 1))

        # Pattern 3: Section references - "section 5.2", "part iii"
        section_match = re.search(
            r'(section|part|chapter)\s*([\d.]+|[ivx]+)',
            ref_lower
        )
        if section_match:
            section_num = section_match.group(2)
            for section in self.sections:
                section_name_lower = section.name.lower()
                if section_num in section_name_lower:
                    return list(range(section.page_start, section.page_end + 1))

        # Pattern 4: Named section lookup via sections list
        for section in self.sections:
            section_name_lower = section.name.lower()
            words_in_ref = set(ref_lower.split())
            words_in_section = set(section_name_lower.split())
            common_words = words_in_ref & words_in_section - {"the", "and", "or", "of", "to", "in", "for", "a", "an"}
            if len(common_words) >= 2 or section_name_lower in ref_lower:
                return list(range(section.page_start, section.page_end + 1))

        # Pattern 5: Named section lookup via page_index
        if page_index:
            for entry in page_index:
                if entry.description:
                    desc_lower = entry.description.lower()
                    if any(keyword in desc_lower for keyword in ref_lower.split() if len(keyword) > 3):
                        return [entry.page]

        return []

    def resolve_reference_single(self, ref_text: str) -> int | None:
        """Resolve a cross-reference text to a single page number.

        Convenience method for backward compatibility.
        """
        pages = self.resolve_reference(ref_text)
        return pages[0] if pages else None

    def get_fund_sections(self) -> list[SectionInfo]:
        """Get all sections that are fund-specific."""
        return [s for s in self.sections if s.section_type == "fund_section" or s.fund_name]

    def get_toc_fund_names(self) -> list[str]:
        """Get canonical fund names from TOC.

        This is the AUTHORITATIVE source of fund names. All other phases
        should map their fund mentions to these canonical names.

        Returns:
            List of fund names in order of appearance in TOC.
        """
        fund_sections = self.get_fund_sections()
        return [s.fund_name or s.name for s in fund_sections]

    def get_fund_page_range(self, fund_name: str) -> tuple[int, int] | None:
        """Get page range for a specific fund from TOC.

        Args:
            fund_name: The canonical fund name (from get_toc_fund_names).

        Returns:
            (start_page, end_page) tuple, or None if not found.
        """
        for section in self.get_fund_sections():
            section_fund = section.fund_name or section.name
            if section_fund == fund_name:
                return (section.page_start, section.page_end)
        return None

    def get_appendices(self) -> list[SectionInfo]:
        """Get all appendix sections."""
        return [s for s in self.sections if s.section_type == "appendix"]

    def summary(self) -> str:
        """Human-readable summary of document structure."""
        lines = [
            f"Document Structure ({self.total_pages} pages)",
            f"  TOC source: {self.toc_source}",
            f"  Sections: {len(self.sections)}",
            f"  Appendices: {len(self.appendix_map)}",
        ]
        if self.sections:
            lines.append("  Top-level sections:")
            for s in self.sections[:10]:
                if s.level == 1:
                    lines.append(f"    - {s.name}: pages {s.page_start}-{s.page_end}")
        return "\n".join(lines)

    def to_explorer_context(self) -> str:
        """Format skeleton as context string for explorer prompts."""
        lines = ["DOCUMENT STRUCTURE:"]

        if self.appendix_map:
            lines.append("\nAppendices:")
            for name, (start, end) in sorted(self.appendix_map.items()):
                lines.append(f"  - {name}: pages {start}-{end}")

        fund_sections = self.get_fund_sections()
        if fund_sections:
            lines.append("\nFund Sections:")
            for s in fund_sections[:20]:
                lines.append(f"  - {s.name}: pages {s.page_start}-{s.page_end}")

        return "\n".join(lines)


class PageContent(BaseModel):
    """What exists on a single page - used to build page-level index.

    Attributes:
        page: Page number (1-indexed)
        content_type: Type of content on this page
        fund_name: If fund_section, which fund this page belongs to
        description: Brief description of page content
    """

    page: int = Field(description="Page number (1-indexed)")
    content_type: Literal[
        "fund_section",
        "fee_table",
        "isin_table",
        "share_class_table",
        "general_info",
        "appendix",
        "toc",
        "other",
    ] = Field(description="Type of content on this page")
    fund_name: str | None = Field(
        default=None,
        description="If fund_section, which fund this page belongs to"
    )
    description: str = Field(
        default="",
        description="Brief description of page content"
    )


# Type alias for page_index parameter
PageIndexEntry = PageContent


class ExplorationNotes(BaseModel):
    """Output from a single explorer covering a page range.

    Attributes:
        page_start: First page covered by this explorer
        page_end: Last page covered by this explorer
        toc_pages: Pages containing table of contents or fund list
        umbrella_info_pages: Pages with umbrella-level info
        funds_mentioned: Sub-funds found in this page range
        tables: Structured tables found
        cross_references: References to other sections
        observations: Free-form notes for planner context
        inventory: Inventory of what data types exist in these pages
        page_index: Page-by-page breakdown of content
    """

    page_start: int = Field(description="First page covered by this explorer")
    page_end: int = Field(description="Last page covered by this explorer")

    toc_pages: list[int] = Field(
        default_factory=list,
        description="Pages containing table of contents or fund list"
    )
    umbrella_info_pages: list[int] = Field(
        default_factory=list,
        description="Pages with umbrella-level info (legal entity, depositary, etc.)"
    )

    funds_mentioned: list[FundMention] = Field(
        default_factory=list,
        description="Sub-funds found in this page range"
    )

    tables: list[TableDiscovery] = Field(
        default_factory=list,
        description="Structured tables found"
    )

    cross_references: list[CrossReference] = Field(
        default_factory=list,
        description="References to other sections (e.g., 'See Appendix E')"
    )

    observations: list[str] = Field(
        default_factory=list,
        description="Free-form notes for planner context"
    )

    inventory: DocumentInventory = Field(
        default_factory=DocumentInventory,
        description="Inventory of what data types exist in these pages"
    )

    page_index: list[PageContent] = Field(
        default_factory=list,
        description="Page-by-page breakdown of content (source of truth for downstream phases)"
    )
