"""Pydantic schemas for critic phase (Phase 4).

These schemas define the verification results from critics.
"""

from typing import Literal
from pydantic import BaseModel, Field


class FieldVerification(BaseModel):
    """Verification result for a single field.

    Attributes:
        field_name: Name of the field being verified
        extracted_value: The value that was extracted
        source_text: Text in document that supports this value
        confidence: Confidence level (high, medium, low, not_found)
        correction: Corrected value if extraction was wrong
        reasoning: Why this confidence level was assigned
    """

    field_name: str = Field(description="Name of the field being verified")
    extracted_value: str = Field(description="The value that was extracted")
    source_text: str | None = Field(
        default=None,
        description="Text in document that supports this value"
    )
    confidence: Literal["high", "medium", "low", "not_found"] = Field(
        description="high=exact match, medium=similar, low=no evidence, not_found=hallucinated"
    )
    correction: str | None = Field(
        default=None,
        description="Corrected value if extraction was wrong"
    )
    reasoning: str = Field(
        default="",
        description="Why this confidence level was assigned"
    )


class CriticResult(BaseModel):
    """Output from critic verification.

    Attributes:
        entity_type: Type of entity verified (umbrella, subfund, shareclass)
        entity_name: Name of the entity
        verifications: Per-field verification results
        overall_confidence: Average confidence (0.0 to 1.0)
        suggested_reread_pages: Pages to re-read if confidence is low
        critical_errors: Errors that must be fixed
    """

    entity_type: Literal["umbrella", "subfund", "shareclass"] = Field(
        description="Type of entity verified"
    )
    entity_name: str = Field(description="Name of the entity")

    verifications: list[FieldVerification] = Field(
        default_factory=list,
        description="Per-field verification results"
    )

    overall_confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="0.0 to 1.0, average confidence"
    )

    suggested_reread_pages: list[int] = Field(
        default_factory=list,
        description="Pages to re-read if confidence is low"
    )

    critical_errors: list[str] = Field(
        default_factory=list,
        description="Errors that must be fixed"
    )
