"""Integration tests for the extraction pipeline.

These tests verify:
1. Phase boundaries and data flow
2. Error propagation across phases
3. Full pipeline execution with mocked components
"""

import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch

from extractor.phases.phase_base import (
    PhaseContext,
    ExtractionResources,
    ExtractionConfig,
    PipelineState,
)
from extractor.phases.assembly_phase import AssemblyPhase, AssemblyResult
from extractor.core import CostTracker
from extractor.pydantic_models import (
    ExplorationNotes,
    FundMention,
    TableDiscovery,
    PlannerOutput,
    FundExtractionTask,
    BroadcastTable,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_pdf_reader():
    """Create a mock PDFReader."""
    reader = MagicMock()
    reader.filename = "test_prospectus.pdf"
    reader.page_count = 100

    def read_pages(start, end):
        return f"--- Pages {start}-{end} ---\nMock content"

    reader.read_pages = read_pages
    reader.search_term = lambda term: []
    return reader


@pytest.fixture
def mock_logger():
    """Create a mock PipelineLogger."""
    logger = MagicMock()
    logger.start_phase = MagicMock()
    logger.end_phase = MagicMock()
    logger.phase_result = MagicMock()
    logger.info = MagicMock()
    logger.warning = MagicMock()
    logger.error = MagicMock()
    logger.debug = MagicMock()
    return logger


@pytest.fixture
def phase_context(mock_pdf_reader, mock_logger):
    """Create a PhaseContext for testing."""
    resources = ExtractionResources(
        pdf=mock_pdf_reader,
        semaphore=asyncio.Semaphore(5),
        logger=mock_logger,
        cost_tracker=CostTracker(),
    )

    config = ExtractionConfig(
        exploration_model="test-model",
        planner_model="test-model",
        reader_model="test-model",
        critic_model="test-model",
        use_critic=False,
        verbose=False,
    )

    state = PipelineState()

    return PhaseContext(
        resources=resources,
        config=config,
        state=state,
    )


@pytest.fixture
def sample_exploration_notes():
    """Create sample exploration notes."""
    return [
        ExplorationNotes(
            page_start=1,
            page_end=50,
            toc_pages=[1, 2],
            umbrella_info_pages=[3, 4, 5],
            funds_mentioned=[
                FundMention(name="Global Equity Fund", page=10, has_dedicated_section=True),
                FundMention(name="Global Bond Fund", page=30, has_dedicated_section=True),
            ],
            tables=[
                TableDiscovery(
                    table_type="fee",
                    page_start=80,
                    page_end=85,
                    columns=["Fund Name", "Management Fee", "Entry Fee"],
                ),
            ],
            observations=["Document has clear structure"],
        ),
    ]


@pytest.fixture
def sample_plan():
    """Create a sample planner output."""
    return PlannerOutput(
        umbrella_name="JPMorgan Investment Funds",
        total_funds=2,
        fund_names=["Global Equity Fund", "Global Bond Fund"],
        umbrella_pages=[3, 4, 5],
        fund_tasks=[
            FundExtractionTask(
                fund_name="Global Equity Fund",
                dedicated_pages=[10, 11, 12, 13, 14],
            ),
            FundExtractionTask(
                fund_name="Global Bond Fund",
                dedicated_pages=[30, 31, 32, 33, 34],
            ),
        ],
        broadcast_tables=[
            BroadcastTable(table_type="fee", pages=[80, 81, 82, 83, 84, 85]),
        ],
    )


@pytest.fixture
def sample_umbrella_data():
    """Create sample umbrella extraction data."""
    return {
        "name": {"value": "JPMorgan Investment Funds", "source_page": 3, "confidence": 0.95},
        "legal_form": {"value": "SICAV", "source_page": 3},
        "domicile": {"value": "Luxembourg", "source_page": 3},
        "management_company": {"value": "JPM Asset Management (Europe)", "source_page": 4},
        "depositary": {"value": "J.P. Morgan Bank Luxembourg S.A.", "source_page": 5},
    }


@pytest.fixture
def sample_funds_data():
    """Create sample fund extraction data."""
    return [
        {
            "name": "Global Equity Fund",
            "investment_objective": {
                "value": "Long-term capital growth through global equities",
                "source_page": 10,
                "confidence": 0.9,
            },
            "share_classes": [
                {
                    "name": "A (acc) USD",
                    "isin": {"value": "LU0123456789", "source_page": 100, "confidence": 0.95},
                    "currency": "USD",
                    "management_fee": "1.50%",
                },
                {
                    "name": "A (dist) EUR",
                    "isin": {"value": "LU0123456790", "source_page": 100, "confidence": 0.95},
                    "currency": "EUR",
                    "management_fee": "1.50%",
                },
            ],
        },
        {
            "name": "Global Bond Fund",
            "investment_objective": {
                "value": "Income and capital preservation",
                "source_page": 30,
                "confidence": 0.85,
            },
            "share_classes": [
                {
                    "name": "A (acc) USD",
                    "isin": {"value": "LU0234567890", "source_page": 100, "confidence": 0.95},
                    "currency": "USD",
                    "management_fee": "0.75%",
                },
            ],
        },
    ]


# =============================================================================
# Phase Context Tests
# =============================================================================


class TestPhaseContext:
    """Tests for PhaseContext delegation and immutability."""

    def test_reads_from_resources(self, phase_context):
        """Test that resource fields are accessible."""
        assert phase_context.pdf is not None
        assert phase_context.logger is not None
        assert phase_context.cost_tracker is not None

    def test_reads_from_config(self, phase_context):
        """Test that config fields are accessible."""
        assert phase_context.exploration_model == "test-model"
        assert phase_context.use_critic is False

    def test_reads_from_state(self, phase_context):
        """Test that state fields are accessible."""
        assert phase_context.exploration_notes == []
        assert phase_context.plan is None

    def test_writes_to_state(self, phase_context):
        """Test that state fields can be written."""
        phase_context.exploration_notes = [MagicMock()]
        assert len(phase_context.exploration_notes) == 1

    def test_blocks_writes_to_resources(self, phase_context):
        """Test that resource fields cannot be written."""
        with pytest.raises(AttributeError):
            phase_context.pdf = MagicMock()

    def test_blocks_writes_to_config(self, phase_context):
        """Test that config fields cannot be written."""
        with pytest.raises(AttributeError):
            phase_context.chunk_size = 100


# =============================================================================
# Assembly Phase Integration Tests
# =============================================================================


class TestAssemblyPhaseIntegration:
    """Integration tests for the assembly phase."""

    @pytest.mark.asyncio
    @pytest.mark.asyncio
    async def test_handles_empty_funds_data(self, phase_context, sample_exploration_notes, sample_plan):
        """Test assembly with no funds extracted."""
        phase_context.exploration_notes = sample_exploration_notes
        phase_context.plan = sample_plan
        phase_context.umbrella_data = {"name": "Test Umbrella"}
        phase_context.funds_data = []
        phase_context.broadcast_data = {}
        phase_context.critic_results = []
        phase_context.discovered_fields = []
        phase_context.schema_suggestions = []

        with patch("extractor.phases.assembly.gap_filling.fill_gaps", new_callable=AsyncMock) as mock_fill:
            mock_fill.return_value = 0

            phase = AssemblyPhase(phase_context)
            result = await phase.run()

        assert len(result.graph.sub_funds) == 0
        assert result.provenance_stats["total"] >= 0


# =============================================================================
# Data Flow Tests
# =============================================================================


class TestDataFlow:
    """Tests for data flow between phases."""

    def test_exploration_notes_flow_to_planning(
        self,
        phase_context,
        sample_exploration_notes,
    ):
        """Test that exploration notes are available for planning."""
        phase_context.exploration_notes = sample_exploration_notes

        # Verify data is accessible
        assert len(phase_context.exploration_notes) == 1
        notes = phase_context.exploration_notes[0]
        assert notes.page_start == 1
        assert notes.page_end == 50
        assert len(notes.funds_mentioned) == 2

    def test_plan_flows_to_extraction(self, phase_context, sample_plan):
        """Test that plan is available for extraction."""
        phase_context.plan = sample_plan

        # Verify data is accessible
        assert phase_context.plan.umbrella_name == "JPMorgan Investment Funds"
        assert len(phase_context.plan.fund_tasks) == 2

    def test_extracted_data_flows_to_assembly(
        self,
        phase_context,
        sample_umbrella_data,
        sample_funds_data,
    ):
        """Test that extracted data is available for assembly."""
        phase_context.umbrella_data = sample_umbrella_data
        phase_context.funds_data = sample_funds_data

        # Verify data is accessible
        assert phase_context.umbrella_data["name"]["value"] == "JPMorgan Investment Funds"
        assert len(phase_context.funds_data) == 2


# =============================================================================
# Error Propagation Tests
# =============================================================================


class TestErrorPropagation:
    """Tests for error handling and propagation."""

    def test_errors_accumulate_in_pipeline_state(self, phase_context):
        """Test that errors accumulate correctly."""
        from extractor.core.errors import ExtractionError, ErrorCategory, ErrorSeverity

        error = ExtractionError(
            category=ErrorCategory.LLM_API,
            severity=ErrorSeverity.ERROR,
            message="API call failed",
            phase="Extraction",
            entity_name="Test Fund",
        )

        phase_context.errors.add(error)

        assert phase_context.errors.error_count == 1
        assert "Test Fund" in phase_context.errors.failed_entities

    def test_warnings_do_not_add_to_failed_entities(self, phase_context):
        """Test that warnings don't mark entities as failed."""
        from extractor.core.errors import ExtractionError, ErrorCategory, ErrorSeverity

        warning = ExtractionError(
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.WARNING,
            message="Field format unusual",
            phase="Extraction",
            entity_name="Test Fund",
        )

        phase_context.errors.add(warning)

        assert phase_context.errors.warning_count == 1
        assert "Test Fund" not in phase_context.errors.failed_entities
