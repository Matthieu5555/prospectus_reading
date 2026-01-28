"""Field searchers - find missing values in prospectus documents.

When extraction fails to find a field (like ISIN, fees, or dividend dates),
these specialized searchers use pattern matching and document knowledge
to locate the missing values.

Searchers available:
- ISINResolver: Finds ISIN codes using regex patterns
- DividendResolver: Finds dividend dates and frequencies
- FeeResolver: Finds management fees and charges
- ValuationPointResolver: Finds NAV calculation times
"""

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass

from extractor.core.config import PageLimits, SearchLimits
from extractor.core.pdf_reader import PDFReader
from extractor.core.document_knowledge import DocumentKnowledge
from extractor.pydantic_models.provenance import ExtractedValue, NotFoundReason


@dataclass
class ResolverResult:
    """Result from running a resolver."""

    field_name: str
    entity_name: str
    value: ExtractedValue
    resolver_used: str
    pages_searched: list[int]


class BaseResolver(ABC):
    """Base class for field resolvers with common functionality."""

    field_name: str = "unknown"
    search_patterns: list[str] = []

    async def resolve(
        self,
        entity_name: str,
        knowledge: DocumentKnowledge,
        pdf: PDFReader,
        search_context: PDFReader,
    ) -> ExtractedValue:
        """Template method for resolution."""

        # Step 1: Check knowledge graph for known locations
        pages = self._get_pages_from_knowledge(knowledge, entity_name)

        # Step 2: If no known pages, search document
        if not pages:
            pages = self._search_for_pages(search_context)

        # Step 3: Check for external reference
        ext_ref = knowledge.get_external_ref_for_field(self.field_name)
        if ext_ref and not pages:
            return ExtractedValue.in_external_doc(
                f"Field {self.field_name} is documented in {ext_ref.external_doc}",
                ext_ref.external_doc
            )

        # Step 4: If we have pages, extract from them
        if pages:
            result = await self._extract_from_pages(entity_name, pages, pdf)
            if result and result.value != "NOT_FOUND":
                return result

        # Step 5: Return appropriate NOT_FOUND
        return self._create_not_found(entity_name, pages, knowledge)

    def _get_pages_from_knowledge(
        self, knowledge: DocumentKnowledge, entity_name: str | None = None,
    ) -> list[int]:
        """Get pages from knowledge graph findings, excluding already-searched pages."""
        pages = knowledge.get_pages_for_field(self.field_name)
        already_searched = knowledge.get_pages_already_searched(
            self.field_name, entity_name,
        )
        if already_searched:
            pages = [p for p in pages if p not in already_searched]
        return pages

    def _search_for_pages(self, search_context: PDFReader) -> list[int]:
        """Search document for relevant pages."""
        all_pages = set()
        per_pattern = max(1, SearchLimits.FIELD_RESOLVER // len(self.search_patterns)) if self.search_patterns else 0
        for pattern in self.search_patterns:
            hits = search_context.search(pattern, per_pattern)
            for hit in hits:
                all_pages.add(hit["page"])
        return sorted(all_pages)[:PageLimits.FIELD_RESOLVER_PAGES]

    @abstractmethod
    async def _extract_from_pages(
        self, entity_name: str, pages: list[int], pdf: PDFReader
    ) -> ExtractedValue | None:
        """Extract the field value from the given pages."""
        pass

    def _create_not_found(
        self, entity_name: str, pages_searched: list[int], knowledge: DocumentKnowledge
    ) -> ExtractedValue:
        """Create appropriate NOT_FOUND value."""
        if pages_searched:
            return ExtractedValue.not_found(
                f"Searched {len(pages_searched)} pages but could not find {self.field_name} for {entity_name}",
                NotFoundReason.NOT_IN_DOCUMENT
            )
        return ExtractedValue.extraction_failed(
            f"No pages identified for {self.field_name} extraction"
        )


def read_pages_if_any(pdf: PDFReader, pages: list[int]) -> str | None:
    """Safely read pages, returning None if pages list is empty."""
    if not pages:
        return None
    return pdf.read_pages(pages[0], pages[-1])


class ISINResolver(BaseResolver):
    """Resolver for ISIN codes."""

    field_name = "isin"
    search_patterns = ["ISIN", "LU0", "LU1", "LU2", "IE00", "FR00", "DE00", "GB00"]

    # ISIN regex: 2 letters + 10 alphanumeric
    ISIN_PATTERN = re.compile(r'\b([A-Z]{2}[A-Z0-9]{10})\b')

    async def _extract_from_pages(
        self, entity_name: str, pages: list[int], pdf: PDFReader
    ) -> ExtractedValue | None:
        """Extract ISINs from pages, matching to entity name."""
        text = read_pages_if_any(pdf, pages)
        if not text:
            return None

        # Find all ISINs
        isins = self.ISIN_PATTERN.findall(text)
        if not isins:
            return None

        # Try to match ISIN to entity name
        # Look for entity name near an ISIN
        entity_lower = entity_name.lower()
        lines = text.split('\n')

        for i, line in enumerate(lines):
            line_lower = line.lower()
            # Check if entity name appears in this or adjacent lines
            context = line_lower
            if i > 0:
                context = lines[i-1].lower() + " " + context
            if i < len(lines) - 1:
                context += " " + lines[i+1].lower()

            # Check for entity name match
            entity_words = entity_lower.split()
            if any(word in context for word in entity_words if len(word) > 3):
                # Found entity context, look for ISIN
                line_isins = self.ISIN_PATTERN.findall(line)
                if line_isins:
                    return ExtractedValue(
                        value=line_isins[0],
                        source_page=pages[0],  # Approximate
                        source_quote=line.strip()[:100],
                        rationale=f"Found ISIN near mention of {entity_name}",
                        confidence=0.8
                    )

        # If we found ISINs but couldn't match to entity, return first one with lower confidence
        if isins:
            return ExtractedValue(
                value=isins[0],
                source_page=pages[0],
                source_quote=f"Found {len(isins)} ISINs in document",
                rationale=f"Could not definitively match to {entity_name}",
                confidence=0.4
            )

        return None


class DividendResolver(BaseResolver):
    """Resolver for dividend-related fields."""

    field_name = "dividend_dates"
    search_patterns = [
        "dividend", "distribution", "payment date", "ex-dividend",
        "record date", "quarterly", "annually", "semi-annual"
    ]

    # Common date patterns
    DATE_PATTERNS = [
        r'\d{1,2}(?:st|nd|rd|th)?\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)',
        r'(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}',
        r'\d{1,2}/\d{1,2}',
    ]

    async def _extract_from_pages(
        self, entity_name: str, pages: list[int], pdf: PDFReader
    ) -> ExtractedValue | None:
        """Extract dividend dates from pages."""
        text = read_pages_if_any(pdf, pages)
        if not text:
            return None
        text_lower = text.lower()

        # Look for dividend/distribution sections
        dividend_keywords = ["dividend", "distribution", "payment"]

        for keyword in dividend_keywords:
            idx = text_lower.find(keyword)
            if idx == -1:
                continue

            # Get context around keyword
            start = max(0, idx - 100)
            end = min(len(text), idx + 300)
            context = text[start:end]

            # Look for dates in context
            dates_found = []
            for pattern in self.DATE_PATTERNS:
                matches = re.findall(pattern, context, re.IGNORECASE)
                dates_found.extend(matches)

            if dates_found:
                return ExtractedValue(
                    value=", ".join(dates_found[:4]),  # Max 4 dates
                    source_page=pages[0],
                    source_quote=context.strip()[:150],
                    rationale=f"Found dates near '{keyword}' keyword",
                    confidence=0.7
                )

        return None


class FeeResolver(BaseResolver):
    """Resolver for fee-related fields."""

    field_name = "management_fee"
    search_patterns = [
        "management fee", "annual fee", "ongoing charge", "TER",
        "expense ratio", "fee schedule", "charges"
    ]

    # Fee percentage pattern
    FEE_PATTERN = re.compile(r'(\d+(?:\.\d+)?)\s*%')

    async def _extract_from_pages(
        self, entity_name: str, pages: list[int], pdf: PDFReader
    ) -> ExtractedValue | None:
        """Extract fee from pages."""
        text = read_pages_if_any(pdf, pages)
        if not text:
            return None
        text_lower = text.lower()

        # Look for fee mentions
        fee_keywords = ["management fee", "annual management", "ongoing charge"]

        for keyword in fee_keywords:
            idx = text_lower.find(keyword)
            if idx == -1:
                continue

            # Get context after keyword
            context = text[idx:idx+100]

            # Look for percentage
            match = self.FEE_PATTERN.search(context)
            if match:
                fee_value = match.group(1) + "%"
                return ExtractedValue(
                    value=fee_value,
                    source_page=pages[0],
                    source_quote=context.strip()[:100],
                    rationale=f"Found fee value after '{keyword}'",
                    confidence=0.75
                )

        return None


class ValuationPointResolver(BaseResolver):
    """Resolver for valuation point/NAV calculation time."""

    field_name = "valuation_point"
    search_patterns = [
        "valuation point", "NAV calculation", "valuation time",
        "net asset value", "pricing point", "dealing"
    ]

    # Time patterns
    TIME_PATTERN = re.compile(r'(\d{1,2}[:.]\d{2})\s*(?:a\.?m\.?|p\.?m\.?|CET|GMT|UTC|hours)?', re.IGNORECASE)

    async def _extract_from_pages(
        self, entity_name: str, pages: list[int], pdf: PDFReader
    ) -> ExtractedValue | None:
        """Extract valuation point from pages."""
        text = read_pages_if_any(pdf, pages)
        if not text:
            return None
        text_lower = text.lower()

        keywords = ["valuation point", "valuation time", "nav calculation", "pricing"]

        for keyword in keywords:
            idx = text_lower.find(keyword)
            if idx == -1:
                continue

            context = text[idx:idx+150]
            match = self.TIME_PATTERN.search(context)
            if match:
                return ExtractedValue(
                    value=match.group(0),
                    source_page=pages[0],
                    source_quote=context.strip()[:100],
                    rationale=f"Found time near '{keyword}'",
                    confidence=0.7
                )

        return None


# Registry of available resolvers
FIELD_RESOLVERS: dict[str, type[BaseResolver]] = {
    "isin": ISINResolver,
    "dividend_dates": DividendResolver,
    "dividend_frequency": DividendResolver,
    "management_fee": FeeResolver,
    "ongoing_charges": FeeResolver,
    "valuation_point": ValuationPointResolver,
}


def get_resolver(field_name: str) -> BaseResolver | None:
    """Get a resolver instance for a field name."""
    resolver_class = FIELD_RESOLVERS.get(field_name)
    if resolver_class:
        resolver = resolver_class()
        resolver.field_name = field_name  # Override for fee variants
        return resolver
    return None
