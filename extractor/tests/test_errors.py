"""Tests for extractor.core.errors module.

Tests the PipelineErrors accumulator - the main error handling infrastructure.
"""

from extractor.core.errors import (
    ExtractionError,
    PipelineErrors,
    ErrorCategory,
    ErrorSeverity,
)


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
