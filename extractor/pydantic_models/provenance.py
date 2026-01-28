"""Provenance tracking for extracted values.

Every extracted value should be wrapped in ExtractedValue to provide:
- Source page number
- Verbatim quote from document
- LLM's rationale for extraction
- Confidence score
- NOT_FOUND reason (when applicable)
"""

from enum import Enum
from typing import Any
from pydantic import BaseModel, Field

from extractor.core.config import ConfidenceThresholds

# Sentinel for missing data - when truly not found in document
NOT_FOUND = "NOT_FOUND"


class NotFoundReason(str, Enum):
    """Typed reasons for why a value was not found.

    This distinguishes between different failure modes:
    - NOT_IN_DOCUMENT: Searched thoroughly, confirmed absent
    - NOT_APPLICABLE: Field doesn't apply (e.g., hedging_details for unhedged share)
    - IN_EXTERNAL_DOC: Explicitly referenced to external doc ("See KIID for ISINs")
    - INHERITED: Value exists at parent level (umbrella/fund), not share-specific
    - EXTRACTION_FAILED: Should exist but extractor couldn't find it (needs investigation)
    """

    NOT_IN_DOCUMENT = "not_in_document"
    NOT_APPLICABLE = "not_applicable"
    IN_EXTERNAL_DOC = "in_external_doc"
    INHERITED = "inherited"
    EXTRACTION_FAILED = "extraction_failed"

    def __str__(self) -> str:
        return self.value


class ExtractedValue(BaseModel):
    """Wrapper for any extracted value with full provenance.

    This is the core model for auditable extraction. Every field should
    be wrapped in this to provide:
    - Where the value came from (page, quote)
    - Why it was extracted (rationale)
    - How confident we are (confidence)
    - How it was found (source_type)

    Example:
        {
            "value": "max 10% single issuer",
            "source_page": 45,
            "source_quote": "The Sub-Fund shall not invest more than 10%...",
            "rationale": "Found in Investment Restrictions section",
            "confidence": 0.95,
            "source_type": "dedicated",
            "extraction_pass": 1
        }

    For NOT_FOUND values, rationale should explain why:
        {
            "value": "NOT_FOUND",
            "source_page": null,
            "source_quote": null,
            "rationale": "ISINs are listed in separate KIID document, not in prospectus",
            "confidence": 1.0,
            "source_type": null
        }
    """

    value: Any = Field(
        description="The extracted value. Can be str, int, float, bool, list, or NOT_FOUND"
    )
    source_page: int | None = Field(
        default=None,
        description="Page number where this value was found (1-indexed)"
    )
    source_quote: str | None = Field(
        default=None,
        description="Verbatim text from the document supporting this extraction"
    )
    rationale: str | None = Field(
        default=None,
        description="LLM's reasoning for this extraction decision"
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Confidence score from 0.0 to 1.0"
    )
    not_found_reason: NotFoundReason | None = Field(
        default=None,
        description="Why the value was not found (only set when value is NOT_FOUND)"
    )
    external_reference: str | None = Field(
        default=None,
        description="External document reference when reason is IN_EXTERNAL_DOC (e.g., 'KIID', 'Annual Report')"
    )
    is_discovered: bool = Field(
        default=False,
        description="True if this field was discovered/suggested by the LLM, not part of predefined schema"
    )
    extraction_pass: int = Field(
        default=1,
        description="Which pass found this value: 1 = initial extraction, 2 = cross-ref resolution"
    )
    source_type: str | None = Field(
        default=None,
        description="How the value was found: 'dedicated', 'cross_ref', 'broadcast_table', 'pattern', 'search'"
    )

    @property
    def is_found(self) -> bool:
        """Check if value was found (not NOT_FOUND)."""
        return self.value != NOT_FOUND

    @property
    def is_not_found(self) -> bool:
        """Check if value is NOT_FOUND."""
        return self.value == NOT_FOUND

    @classmethod
    def not_found(
        cls,
        rationale: str,
        reason: NotFoundReason = NotFoundReason.EXTRACTION_FAILED,
        external_ref: str | None = None,
        extraction_pass: int = 1,
    ) -> "ExtractedValue":
        """Create a NOT_FOUND value with typed reason.

        Args:
            rationale: Human-readable explanation
            reason: Typed reason from NotFoundReason enum
            external_ref: External document name if reason is IN_EXTERNAL_DOC
            extraction_pass: Which pass determined NOT_FOUND

        Returns:
            ExtractedValue with NOT_FOUND and typed reason
        """
        return cls(
            value=NOT_FOUND,
            source_page=None,
            source_quote=None,
            rationale=rationale,
            confidence=1.0,
            not_found_reason=reason,
            external_reference=external_ref,
            extraction_pass=extraction_pass,
            source_type=None,
        )

    @classmethod
    def not_in_document(cls, rationale: str) -> "ExtractedValue":
        """Create NOT_FOUND: confirmed absent from document after thorough search."""
        return cls.not_found(rationale, NotFoundReason.NOT_IN_DOCUMENT)

    @classmethod
    def not_applicable(cls, rationale: str) -> "ExtractedValue":
        """Create NOT_FOUND: field doesn't apply to this entity."""
        return cls.not_found(rationale, NotFoundReason.NOT_APPLICABLE)

    @classmethod
    def in_external_doc(cls, rationale: str, external_doc: str) -> "ExtractedValue":
        """Create NOT_FOUND: value is in external document (e.g., KIID)."""
        return cls.not_found(rationale, NotFoundReason.IN_EXTERNAL_DOC, external_doc)

    @classmethod
    def inherited(cls, rationale: str) -> "ExtractedValue":
        """Create NOT_FOUND: value inherited from parent level."""
        return cls.not_found(rationale, NotFoundReason.INHERITED)

    @classmethod
    def extraction_failed(cls, rationale: str) -> "ExtractedValue":
        """Create NOT_FOUND: extraction failed, needs investigation."""
        return cls.not_found(rationale, NotFoundReason.EXTRACTION_FAILED)

    @classmethod
    def from_simple(
        cls,
        value: Any,
        page: int | None = None,
        source_type: str | None = None,
    ) -> "ExtractedValue":
        """Create from a simple value without full provenance.

        Use this for legacy/migration - new extractions should provide full provenance.

        Args:
            value: The extracted value
            page: Optional page number
            source_type: How the value was found (dedicated, cross_ref, etc.)

        Returns:
            ExtractedValue with minimal provenance
        """
        if value == NOT_FOUND:
            return cls.extraction_failed("No rationale provided (legacy extraction)")
        return cls(
            value=value,
            source_page=page,
            source_quote=None,
            rationale=None,
            confidence=ConfidenceThresholds.DEFAULT_UNVERIFIED,
            source_type=source_type,
        )

    @property
    def is_actionable_not_found(self) -> bool:
        """Check if this NOT_FOUND needs investigation.

        Only EXTRACTION_FAILED values are actionable - others have valid reasons.
        """
        return (
            self.is_not_found and
            self.not_found_reason == NotFoundReason.EXTRACTION_FAILED
        )

    @classmethod
    def discovered(
        cls,
        value: Any,
        source_page: int | None = None,
        source_quote: str | None = None,
        rationale: str | None = None,
        confidence: float = 1.0,
        source_type: str | None = None,
    ) -> "ExtractedValue":
        """Create an ExtractedValue for a discovered (non-schema) field.

        Use this when the LLM suggests a field that's not in our predefined schema.

        Args:
            value: The extracted value
            source_page: Page where found
            source_quote: Verbatim quote
            rationale: Why this field is valuable
            confidence: Confidence score
            source_type: How the value was found

        Returns:
            ExtractedValue with is_discovered=True
        """
        return cls(
            value=value,
            source_page=source_page,
            source_quote=source_quote,
            rationale=rationale,
            confidence=confidence,
            is_discovered=True,
            source_type=source_type,
        )


# Type alias for fields that may or may not be extracted
MaybeExtracted = ExtractedValue | None
