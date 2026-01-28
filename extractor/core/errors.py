"""Structured error types for the extraction pipeline.

Provides typed errors for:
- Extraction failures
- LLM API errors
- Validation errors
- Resource errors
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ErrorSeverity(Enum):
    """Severity levels for extraction errors."""
    WARNING = "warning"   # Non-fatal, extraction continued with defaults
    ERROR = "error"       # Fatal for this item, pipeline continued
    CRITICAL = "critical" # Pipeline halted


class ErrorCategory(Enum):
    """Categories of extraction errors."""
    LLM_API = "llm_api"           # OpenRouter/LiteLLM errors
    LLM_PARSE = "llm_parse"       # JSON parsing errors from LLM response
    PDF_READ = "pdf_read"         # PDF reading errors
    VALIDATION = "validation"     # Pydantic validation errors
    TIMEOUT = "timeout"           # Operation timeout
    RESOURCE = "resource"         # Resource exhaustion (memory, rate limit)
    UNKNOWN = "unknown"           # Unclassified errors


@dataclass
class ExtractionError:
    """Structured extraction error with context."""

    category: ErrorCategory
    severity: ErrorSeverity
    message: str
    phase: str                     # Pipeline phase where error occurred
    entity_type: str | None = None # "umbrella", "subfund", "share_class"
    entity_name: str | None = None # Name of the entity being extracted
    page_range: tuple[int, int] | None = None
    original_error: Exception | None = None
    retry_count: int = 0
    context: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        parts = [f"[{self.severity.value.upper()}] {self.category.value}: {self.message}"]
        if self.entity_name:
            parts.append(f"entity={self.entity_name}")
        if self.phase:
            parts.append(f"phase={self.phase}")
        if self.retry_count > 0:
            parts.append(f"retries={self.retry_count}")
        return " | ".join(parts)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "category": self.category.value,
            "severity": self.severity.value,
            "message": self.message,
            "phase": self.phase,
            "entity_type": self.entity_type,
            "entity_name": self.entity_name,
            "page_range": list(self.page_range) if self.page_range else None,
            "retry_count": self.retry_count,
            "context": self.context,
        }


@dataclass
class PipelineErrors:
    """Aggregate errors across entire pipeline run."""

    errors: list[ExtractionError] = field(default_factory=list)
    warnings: list[ExtractionError] = field(default_factory=list)
    failed_entities: list[str] = field(default_factory=list)

    def add(self, error: ExtractionError):
        """Add an error or warning."""
        if error.severity == ErrorSeverity.WARNING:
            self.warnings.append(error)
        else:
            self.errors.append(error)
            if error.entity_name and error.entity_name not in self.failed_entities:
                self.failed_entities.append(error.entity_name)

    @property
    def error_count(self) -> int:
        return len(self.errors)

    @property
    def warning_count(self) -> int:
        return len(self.warnings)

    def summary(self) -> dict:
        """Get summary statistics."""
        by_category = {}
        for error in self.errors:
            cat = error.category.value
            by_category[cat] = by_category.get(cat, 0) + 1

        return {
            "total_errors": self.error_count,
            "total_warnings": self.warning_count,
            "failed_entities": len(self.failed_entities),
            "errors_by_category": by_category,
        }

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "errors": [e.to_dict() for e in self.errors],
            "warnings": [w.to_dict() for w in self.warnings],
            "failed_entities": self.failed_entities,
            "summary": self.summary(),
        }


# Factory functions for common error types

def llm_api_error(
    message: str,
    phase: str,
    entity_name: str | None = None,
    original: Exception | None = None,
    retry_count: int = 0,
) -> ExtractionError:
    """Create an LLM API error."""
    return ExtractionError(
        category=ErrorCategory.LLM_API,
        severity=ErrorSeverity.ERROR,
        message=message,
        phase=phase,
        entity_name=entity_name,
        original_error=original,
        retry_count=retry_count,
    )


def llm_parse_error(
    message: str,
    phase: str,
    entity_name: str | None = None,
    raw_response: str | None = None,
) -> ExtractionError:
    """Create an LLM parse error."""
    return ExtractionError(
        category=ErrorCategory.LLM_PARSE,
        severity=ErrorSeverity.ERROR,
        message=message,
        phase=phase,
        entity_name=entity_name,
        context={"raw_response": raw_response[:500] if raw_response else None},
    )


def validation_error(
    message: str,
    phase: str,
    entity_name: str | None = None,
    field_name: str | None = None,
) -> ExtractionError:
    """Create a validation error."""
    return ExtractionError(
        category=ErrorCategory.VALIDATION,
        severity=ErrorSeverity.WARNING,
        message=message,
        phase=phase,
        entity_name=entity_name,
        context={"field": field_name} if field_name else {},
    )


def timeout_error(
    phase: str,
    entity_name: str | None = None,
    timeout_seconds: float | None = None,
) -> ExtractionError:
    """Create a timeout error."""
    return ExtractionError(
        category=ErrorCategory.TIMEOUT,
        severity=ErrorSeverity.ERROR,
        message=f"Operation timed out after {timeout_seconds}s" if timeout_seconds else "Operation timed out",
        phase=phase,
        entity_name=entity_name,
    )


def resource_error(
    message: str,
    phase: str,
    entity_name: str | None = None,
) -> ExtractionError:
    """Create a resource error (rate limit, memory, etc)."""
    return ExtractionError(
        category=ErrorCategory.RESOURCE,
        severity=ErrorSeverity.ERROR,
        message=message,
        phase=phase,
        entity_name=entity_name,
    )
