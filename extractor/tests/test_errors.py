"""Tests for extractor.core.errors module.

Tests the error handling infrastructure:
- ExtractionError dataclass
- PipelineErrors accumulator
- Error factory functions
- Error categorization and severity
"""

from extractor.core.errors import (
    ExtractionError,
    PipelineErrors,
    ErrorCategory,
    ErrorSeverity,
    llm_api_error,
    llm_parse_error,
    validation_error,
    timeout_error,
    resource_error,
)


# =============================================================================
# ErrorCategory and ErrorSeverity tests
# =============================================================================


class TestErrorEnums:
    """Tests for error enums."""

    def test_all_categories_exist(self):
        """All error categories are defined."""
        assert ErrorCategory.LLM_API
        assert ErrorCategory.LLM_PARSE
        assert ErrorCategory.PDF_READ
        assert ErrorCategory.VALIDATION
        assert ErrorCategory.TIMEOUT
        assert ErrorCategory.RESOURCE
        assert ErrorCategory.UNKNOWN

    def test_all_severities_exist(self):
        """All severity levels are defined."""
        assert ErrorSeverity.WARNING
        assert ErrorSeverity.ERROR
        assert ErrorSeverity.CRITICAL


# =============================================================================
# ExtractionError tests
# =============================================================================


class TestExtractionError:
    """Tests for ExtractionError dataclass."""

    def test_basic_creation(self):
        """Create basic ExtractionError."""
        error = ExtractionError(
            category=ErrorCategory.LLM_API,
            severity=ErrorSeverity.ERROR,
            message="API call failed",
            phase="extraction",
        )
        assert error.category == ErrorCategory.LLM_API
        assert error.severity == ErrorSeverity.ERROR
        assert error.message == "API call failed"
        assert error.phase == "extraction"

    def test_full_creation(self):
        """Create ExtractionError with all fields."""
        original = Exception("Connection timeout")
        error = ExtractionError(
            category=ErrorCategory.TIMEOUT,
            severity=ErrorSeverity.ERROR,
            message="Request timed out",
            phase="exploration",
            entity_type="chunk",
            entity_name="pages_1-30",
            page_range=(1, 30),
            original_error=original,
            retry_count=3,
            context={"url": "https://api.example.com"},
        )
        assert error.entity_name == "pages_1-30"
        assert error.page_range == (1, 30)
        assert error.retry_count == 3
        assert error.context["url"] == "https://api.example.com"

    def test_str_representation(self):
        """ExtractionError has readable string representation."""
        error = ExtractionError(
            category=ErrorCategory.LLM_API,
            severity=ErrorSeverity.ERROR,
            message="API failed",
            phase="extraction",
        )
        error_str = str(error)
        assert "LLM_API" in error_str or "API" in error_str
        assert "extraction" in error_str


# =============================================================================
# Factory function tests
# =============================================================================


class TestErrorFactoryFunctions:
    """Tests for error factory functions."""

    def test_llm_api_error(self):
        """llm_api_error creates correct error."""
        original = Exception("Connection refused")
        error = llm_api_error(
            message="Failed to call API",
            phase="extraction",
            entity_name="Global Fund",
            original=original,
            retry_count=2,
        )
        assert error.category == ErrorCategory.LLM_API
        assert error.severity == ErrorSeverity.ERROR
        assert error.entity_name == "Global Fund"
        assert error.retry_count == 2
        assert error.original_error == original

    def test_llm_parse_error(self):
        """llm_parse_error creates correct error."""
        error = llm_parse_error(
            message="Invalid JSON response",
            phase="exploration",
            entity_name="chunk_1",
            raw_response="not valid json",
        )
        assert error.category == ErrorCategory.LLM_PARSE
        assert error.context["raw_response"] == "not valid json"

    def test_validation_error(self):
        """validation_error creates correct error."""
        error = validation_error(
            message="ISIN format invalid",
            phase="extraction",
            entity_name="Share Class A",
            field_name="isin",
        )
        assert error.category == ErrorCategory.VALIDATION
        assert error.severity == ErrorSeverity.WARNING
        assert error.context["field"] == "isin"

    def test_timeout_error(self):
        """timeout_error creates correct error."""
        error = timeout_error(
            phase="extraction",
            entity_name="Large Fund",
            timeout_seconds=30,
        )
        assert error.category == ErrorCategory.TIMEOUT
        assert "30" in error.message or error.context.get("timeout_seconds") == 30

    def test_resource_error(self):
        """resource_error creates correct error."""
        error = resource_error(
            message="Out of memory",
            phase="assembly",
            entity_name="pipeline",
        )
        assert error.category == ErrorCategory.RESOURCE
        assert error.severity == ErrorSeverity.ERROR


# =============================================================================
# PipelineErrors tests
# =============================================================================


class TestPipelineErrors:
    """Tests for PipelineErrors accumulator."""

    def test_empty_pipeline_errors(self):
        """Empty PipelineErrors has no errors."""
        pe = PipelineErrors()
        assert len(pe.errors) == 0
        assert len(pe.warnings) == 0
        assert len(pe.failed_entities) == 0

    def test_add_error(self):
        """add() with ERROR severity adds to errors list."""
        pe = PipelineErrors()
        error = ExtractionError(
            category=ErrorCategory.LLM_API,
            severity=ErrorSeverity.ERROR,
            message="Failed",
            phase="extraction",
            entity_name="Fund A",
        )
        pe.add(error)

        assert len(pe.errors) == 1
        assert "Fund A" in pe.failed_entities

    def test_add_warning(self):
        """add() with WARNING severity adds to warnings list."""
        pe = PipelineErrors()
        warning = ExtractionError(
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.WARNING,
            message="Minor issue",
            phase="extraction",
        )
        pe.add(warning)

        assert len(pe.warnings) == 1
        assert len(pe.errors) == 0

    def test_failed_entities_tracking(self):
        """Failed entities are tracked across errors."""
        pe = PipelineErrors()

        pe.add(ExtractionError(
            category=ErrorCategory.LLM_API,
            severity=ErrorSeverity.ERROR,
            message="Failed",
            phase="extraction",
            entity_name="Fund A",
        ))
        pe.add(ExtractionError(
            category=ErrorCategory.LLM_API,
            severity=ErrorSeverity.ERROR,
            message="Also failed",
            phase="extraction",
            entity_name="Fund B",
        ))
        # Duplicate entity
        pe.add(ExtractionError(
            category=ErrorCategory.TIMEOUT,
            severity=ErrorSeverity.ERROR,
            message="Timeout",
            phase="extraction",
            entity_name="Fund A",
        ))

        # Fund A should only appear once
        assert len(pe.failed_entities) == 2
        assert "Fund A" in pe.failed_entities
        assert "Fund B" in pe.failed_entities

    def test_summary(self):
        """summary returns structured overview."""
        pe = PipelineErrors()
        pe.add(ExtractionError(
            category=ErrorCategory.LLM_API,
            severity=ErrorSeverity.ERROR,
            message="API Error",
            phase="extraction",
            entity_name="Fund A",
        ))
        pe.add(ExtractionError(
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.WARNING,
            message="Validation warning",
            phase="extraction",
        ))

        summary = pe.summary()

        assert summary["total_errors"] == 1
        assert summary["total_warnings"] == 1
        assert summary["failed_entities"] == 1
        assert "llm_api" in summary.get("errors_by_category", {})

    def test_error_count_property(self):
        """error_count property returns count of errors."""
        pe = PipelineErrors()
        assert pe.error_count == 0

        pe.add(ExtractionError(
            category=ErrorCategory.LLM_API,
            severity=ErrorSeverity.ERROR,
            message="Error",
            phase="extraction",
        ))
        assert pe.error_count == 1

    def test_warning_count_property(self):
        """warning_count property returns count of warnings."""
        pe = PipelineErrors()
        assert pe.warning_count == 0

        pe.add(ExtractionError(
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.WARNING,
            message="Warning",
            phase="extraction",
        ))
        assert pe.warning_count == 1
