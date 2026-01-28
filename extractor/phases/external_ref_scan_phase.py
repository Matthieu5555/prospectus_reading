"""External Reference Pre-Scan phase - detects external document references.

This phase detects when field values aren't in the prospectus but live in
external documents (KIID, Annual Report, Supplement, etc.). It uses two
complementary approaches:

1. **Table cell detection**: Checks pre-scanned tables for cells containing
   "See KIID" instead of actual values. Very accurate since table structure
   is unambiguous.

2. **Proximity-based text matching**: Finds sentences where a field keyword,
   referral phrase, and external document name appear together. More flexible
   than exact regex patterns.

By detecting these early, we prevent wasted LLM calls searching for data
that doesn't exist in the document.

No LLM calls - pure heuristic scanning for speed.
"""

import re
from dataclasses import dataclass

from extractor.phases.phase_base import PhaseRunner


@dataclass
class ExternalRefResult:
    """Result from external reference pre-scan."""

    total_refs_found: int
    fields_external: dict[str, str]  # field_name -> external_doc
    pages_scanned: int
    table_refs_found: int  # References found via table cell detection
    text_refs_found: int   # References found via text proximity matching


# =============================================================================
# Vocabulary for proximity-based detection
# =============================================================================
#
# These dictionaries encode domain knowledge about how prospectuses refer to
# external documents. They're hand-curated from analyzing ~20 real prospectuses
# across different asset managers and jurisdictions.
#
# If you're adding support for a new document type or jurisdiction (e.g., US
# mutual funds, Asian markets), this is where you'd add keywords. The structure
# maps raw text patterns to canonical names, so "TER," "ongoing charges," and
# "OCF" all resolve to the same `ongoing_charges` field.
#
# (This could have been a ML classifier trained on labeled examples, but the
# vocabulary is small enough that hand-curation is faster and more transparent.
# You can see exactly what patterns are matched. A classifier would be a black
# box that might miss obvious patterns or match spurious ones.)
# =============================================================================

# Field keywords - map to canonical field names
FIELD_KEYWORDS: dict[str, list[str]] = {
    "isin": ["isin", "isin code", "isin codes", "identifier", "identifiers"],
    "performance": ["performance", "past performance", "historical performance", "returns", "track record"],
    "ongoing_charges": ["ongoing charge", "ongoing charges", "ter", "total expense ratio", "ocf", "ongoing cost"],
    "management_fee": ["management fee", "management fees", "annual management charge", "amc"],
    "entry_fee": ["entry fee", "entry charge", "subscription fee", "initial charge"],
    "exit_fee": ["exit fee", "exit charge", "redemption fee", "redemption charge"],
    "risk_profile": ["risk profile", "risk indicator", "risk level", "srri", "risk rating", "risk category"],
    "distribution_policy": ["distribution", "dividend", "income distribution"],
    # Additional fields for better coverage
    "benchmark": ["benchmark", "index", "reference index", "comparator", "performance comparator"],
    "nav": ["nav", "net asset value", "nav per share", "unit price", "share price"],
    "minimum_investment": ["minimum investment", "minimum subscription", "minimum initial", "investment minimum"],
    "fund_size": ["fund size", "aum", "assets under management", "total assets", "net assets"],
    "inception_date": ["inception", "launch date", "commencement date", "fund launch"],
    "dealing_frequency": ["dealing", "valuation day", "dealing day", "subscription day", "redemption day"],
    "volatility": ["volatility", "standard deviation", "sharpe ratio", "risk metrics"],
}

# Referral phrases that indicate redirection to another document
REFERRAL_PHRASES: list[str] = [
    "see", "refer to", "refer", "available in", "available at", "available on",
    "found in", "consult", "published in", "disclosed in", "contained in",
    "detailed in", "set out in", "specified in", "listed in", "shown in",
    "please contact", "contact", "upon request", "on request",
    "not included", "not disclosed", "not available", "not shown",
    # Additional phrases for better coverage
    "please see", "please refer", "kindly refer", "readers should refer",
    "appendix", "annex", "schedule", "exhibit",
    "full details", "further information", "more information", "additional information",
    "available from", "provided by", "obtainable from", "can be obtained",
    "separately disclosed", "published separately", "disclosed separately",
    "investors should consult", "shareholders should refer",
]

# External document names
EXTERNAL_DOCS: dict[str, str] = {
    # KIID variants -> canonical "KIID"
    "kiid": "KIID",
    "kid": "KIID",
    "key investor information": "KIID",
    "key investor document": "KIID",
    "key information document": "KIID",
    "priips": "KIID",
    # Annual Report variants
    "annual report": "Annual Report",
    "semi-annual report": "Annual Report",
    "semi annual report": "Annual Report",
    "audited accounts": "Annual Report",
    # Factsheet variants
    "factsheet": "Factsheet",
    "fund factsheet": "Factsheet",
    "fact sheet": "Factsheet",
    "monthly report": "Factsheet",
    # Other documents
    "supplement": "Supplement",
    "addendum": "Supplement",
    "prospectus": "Prospectus",
    "master prospectus": "Prospectus",
    "base prospectus": "Prospectus",
    "term sheet": "TermSheet",
    "terms and conditions": "TermSheet",
    # SFDR/regulatory
    "sfdr": "SFDR",
    "art. 6": "SFDR",
    "art. 8": "SFDR",
    "art. 9": "SFDR",
    "article 6": "SFDR",
    "article 8": "SFDR",
    "article 9": "SFDR",
    "sustainability disclosure": "SFDR",
    "pre-contractual disclosure": "SFDR",
    # Website variants
    "website": "Website",
    "www.": "Website",
    "http": "Website",
    "investor portal": "Website",
    "client portal": "Website",
    "fund centre": "Website",
    "fund center": "Website",
    # Not disclosed
    "upon request": "Upon Request",
    "on request": "Upon Request",
    "not disclosed": "Not Disclosed",
    "not available": "Not Disclosed",
    "contact the administrator": "Upon Request",
    "contact the management company": "Upon Request",
}

# Column name mappings for table detection
COLUMN_TO_FIELD: dict[str, str] = {
    "isin": "isin",
    "isin code": "isin",
    "isin codes": "isin",
    "identifier": "isin",
    "management fee": "management_fee",
    "mgmt fee": "management_fee",
    "annual fee": "management_fee",
    "entry fee": "entry_fee",
    "entry charge": "entry_fee",
    "subscription fee": "entry_fee",
    "initial charge": "entry_fee",
    "exit fee": "exit_fee",
    "exit charge": "exit_fee",
    "redemption fee": "exit_fee",
    "redemption charge": "exit_fee",
    "ongoing charges": "ongoing_charges",
    "ongoing charge": "ongoing_charges",
    "ter": "ongoing_charges",
    "ocf": "ongoing_charges",
    "total expense ratio": "ongoing_charges",
    "performance fee": "performance_fee",
    "currency": "currency",
    "ccy": "currency",
    "base currency": "currency",
    "distribution": "distribution_policy",
    "distribution policy": "distribution_policy",
    # Additional column mappings
    "share class": "share_class_name",
    "class name": "share_class_name",
    "share class name": "share_class_name",
    "inception": "inception_date",
    "launch": "inception_date",
    "launch date": "inception_date",
    "inception date": "inception_date",
    "minimum investment": "minimum_investment",
    "min investment": "minimum_investment",
    "minimum initial": "minimum_investment",
    "srri": "risk_profile",
    "risk indicator": "risk_profile",
}


def _is_reference_cell(cell_value: str) -> tuple[bool, str | None]:
    """Check if a table cell contains a reference instead of an actual value.

    Args:
        cell_value: The cell content.

    Returns:
        Tuple of (is_reference, external_doc_name).
    """
    if not cell_value:
        return False, None

    cell = cell_value.strip().lower()

    # Skip empty or placeholder cells
    if cell in ["", "-", "n/a", "na", "—", "–"]:
        return False, None

    # Real values are usually short; long text is probably not a reference
    if len(cell) > 80:
        return False, None

    # Check for external doc mentions
    for doc_keyword, doc_name in EXTERNAL_DOCS.items():
        if doc_keyword in cell:
            # Confirm it's a reference (has referral language or is very short)
            has_referral = any(phrase in cell for phrase in ["see", "refer", "available", "consult"])
            is_short = len(cell) < 30
            if has_referral or is_short:
                return True, doc_name

    return False, None


def _find_external_doc(text: str) -> str | None:
    """Find which external document is mentioned in text.

    Args:
        text: Text to search (lowercase).

    Returns:
        Canonical document name or None.
    """
    for doc_keyword, doc_name in EXTERNAL_DOCS.items():
        if doc_keyword in text:
            return doc_name
    return None


def _split_into_sentences(text: str) -> list[str]:
    """Split text into sentence-like segments for proximity matching.

    Args:
        text: Full text to split.

    Returns:
        List of segments (sentences or line-based chunks).
    """
    # Split on sentence boundaries and newlines
    # This is simple but works for prospectus text
    segments = re.split(r'[.!?\n]+', text)
    return [s.strip() for s in segments if s.strip() and len(s.strip()) > 10]


class ExternalRefScanPhase(PhaseRunner[ExternalRefResult]):
    """Pre-scan phase for external document references.

    Uses two detection methods:
    1. Table cell detection: Checks scanned tables for "See KIID" type cells
    2. Proximity matching: Finds field + referral + external doc in same sentence

    Results are written to the knowledge graph so downstream phases
    know not to waste effort searching for externally-documented fields.
    """

    name = "ExternalRefScan"

    async def run(self) -> ExternalRefResult:
        """Scan document for external reference patterns.

        Returns:
            ExternalRefResult with detected external references.
        """
        total_pages = self.context.pdf.page_count
        self.log(f"Scanning {total_pages} pages for external references")

        # Track findings
        fields_external: dict[str, str] = {}
        table_refs_found = 0
        text_refs_found = 0

        # Method 1: Table cell detection (uses pre-scanned tables from TableScan)
        table_refs_found = self._scan_tables(fields_external)

        # Method 2: Proximity-based text matching
        text_refs_found = self._scan_text_proximity(fields_external)

        # Record to knowledge graph
        self._record_to_knowledge_graph(fields_external)

        # Log summary
        total_refs = table_refs_found + text_refs_found
        if fields_external:
            self.log(f"Detected {len(fields_external)} external field(s): {list(fields_external.keys())}")
            self.log(f"  Table detection: {table_refs_found}, Text proximity: {text_refs_found}")
        else:
            self.log("No external references detected")

        self.logger.phase_result(
            self.name,
            f"{total_refs} refs, {len(fields_external)} fields external",
        )

        return ExternalRefResult(
            total_refs_found=total_refs,
            fields_external=fields_external,
            pages_scanned=total_pages,
            table_refs_found=table_refs_found,
            text_refs_found=text_refs_found,
        )

    def _scan_tables(self, fields_external: dict[str, str]) -> int:
        """Scan pre-parsed tables for reference cells.

        Args:
            fields_external: Dict to populate with findings.

        Returns:
            Number of reference cells found.
        """
        refs_found = 0
        scanned_tables = self.context.state.scanned_tables

        if not scanned_tables:
            self.log("No pre-scanned tables available", "debug")
            return 0

        for table in scanned_tables:
            # Check each column
            for col_name in table.columns:
                col_lower = col_name.lower().strip()

                # Map column to field name
                field_name = COLUMN_TO_FIELD.get(col_lower)
                if not field_name:
                    # Try partial matching
                    for col_pattern, field in COLUMN_TO_FIELD.items():
                        if col_pattern in col_lower:
                            field_name = field
                            break

                if not field_name:
                    continue

                # Already detected this field
                if field_name in fields_external:
                    continue

                # Count reference cells vs total cells
                ref_count = 0
                total_cells = 0
                detected_doc = None

                for row in table.rows:
                    cell_value = row.get(col_name, "")
                    if not cell_value:
                        continue

                    total_cells += 1
                    is_ref, doc_name = _is_reference_cell(cell_value)
                    if is_ref:
                        ref_count += 1
                        if doc_name:
                            detected_doc = doc_name

                # If significant portion of cells are references, mark field as external
                # Threshold: >30% of non-empty cells are references
                if total_cells > 0 and ref_count > total_cells * 0.3:
                    external_doc = detected_doc or "KIID"
                    fields_external[field_name] = external_doc
                    refs_found += ref_count
                    self.log(f"Table detection: {field_name} -> {external_doc} ({ref_count}/{total_cells} cells)")

        return refs_found

    def _scan_text_proximity(self, fields_external: dict[str, str]) -> int:
        """Scan text for proximity-based external references.

        Finds sentences where field keyword + referral phrase + external doc
        appear together.

        Args:
            fields_external: Dict to populate with findings.

        Returns:
            Number of references found.
        """
        refs_found = 0
        total_pages = self.context.pdf.page_count

        # Sample pages for efficiency (first 15 + last 10 pages typically have this info)
        pages_to_scan = list(range(1, min(16, total_pages + 1)))
        if total_pages > 25:
            pages_to_scan.extend(range(total_pages - 9, total_pages + 1))
        pages_to_scan = sorted(set(pages_to_scan))

        for page_num in pages_to_scan:
            try:
                page_text = self.context.pdf.read_pages(page_num, page_num)
                page_lower = page_text.lower()

                # Split into sentences for proximity matching
                sentences = _split_into_sentences(page_lower)

                for sentence in sentences:
                    # Check if sentence mentions an external document
                    external_doc = _find_external_doc(sentence)
                    if not external_doc:
                        continue

                    # Check if sentence has a referral phrase
                    has_referral = any(phrase in sentence for phrase in REFERRAL_PHRASES)
                    if not has_referral:
                        continue

                    # Check which fields are mentioned
                    for field_name, keywords in FIELD_KEYWORDS.items():
                        if field_name in fields_external:
                            continue  # Already detected

                        if any(kw in sentence for kw in keywords):
                            fields_external[field_name] = external_doc
                            refs_found += 1

                            # Get a clean quote (find original case in page_text)
                            quote = sentence[:100] + "..." if len(sentence) > 100 else sentence
                            self.log(f"Text proximity: {field_name} -> {external_doc} (page {page_num})")

                            # Record with quote for knowledge graph
                            self._record_single_ref(field_name, external_doc, page_num, quote)

            except Exception as e:
                self.log(f"Error scanning page {page_num}: {e}", "warning")

        return refs_found

    def _record_single_ref(
        self,
        field_name: str,
        external_doc: str,
        page_num: int,
        quote: str,
    ) -> None:
        """Record a single external reference to the knowledge graph.

        Args:
            field_name: Field name.
            external_doc: External document name.
            page_num: Source page.
            quote: Source quote.
        """
        knowledge = self.context.knowledge
        existing = knowledge.get_external_ref_for_field(field_name)
        if not existing:
            knowledge.record_external_reference(
                field_name=field_name,
                external_doc=external_doc,
                source_page=page_num,
                source_quote=quote[:200],
                source_agent="external_ref_scan",
            )

    def _record_to_knowledge_graph(self, fields_external: dict[str, str]) -> None:
        """Record all findings to the knowledge graph.

        Args:
            fields_external: Dict of field_name -> external_doc.
        """
        knowledge = self.context.knowledge

        for field_name, external_doc in fields_external.items():
            existing = knowledge.get_external_ref_for_field(field_name)
            if not existing:
                knowledge.record_external_reference(
                    field_name=field_name,
                    external_doc=external_doc,
                    source_page=0,  # Unknown for table detections
                    source_quote=f"Detected via table/proximity scan: {field_name} -> {external_doc}",
                    source_agent="external_ref_scan",
                )
