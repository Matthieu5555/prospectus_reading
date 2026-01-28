"""Tests for robust umbrella extraction.

Tests the two-pass umbrella extraction approach:
1. Entity info from bounded intro+outro pages
2. Constraints from TOC-guided sections or search fallback

These tests verify that:
- Entity pages are always bounded (never exceed ~20 pages)
- Constraint pages are correctly identified from TOC
- Excessive planner output is rejected
- Small documents are handled correctly
- Token limits are enforced
"""

import pytest

from extractor.core.config import UmbrellaConfig, ExtractionConfig
from extractor.core.umbrella_page_selector import (
    UmbrellaPagePlan,
    select_umbrella_pages,
    validate_umbrella_pages,
    estimate_tokens,
    check_token_limit,
    _compute_entity_pages,
    _find_constraint_sections,
    _contiguous_blocks,
)
from extractor.pydantic_models.exploration_models import (
    DocumentSkeleton,
    SectionInfo,
)


# =============================================================================
# Entity Pages Tests
# =============================================================================


class TestComputeEntityPages:
    """Tests for _compute_entity_pages function."""

    def test_large_document_bounded(self):
        """Large document returns bounded intro + outro pages."""
        pages = _compute_entity_pages(total_pages=261)

        # Should have intro (5) + outro (15) pages = 20 total
        expected_intro = list(range(1, 6))  # 1-5
        expected_outro = list(range(247, 262))  # 247-261

        assert len(pages) == 20
        assert pages[:5] == expected_intro
        assert pages[5:] == expected_outro

    def test_small_document_reads_all(self):
        """Small document reads all pages."""
        pages = _compute_entity_pages(total_pages=15)

        # Should read all 15 pages
        assert pages == list(range(1, 16))

    def test_medium_document_no_overlap(self):
        """Medium document has no overlap between intro and outro."""
        pages = _compute_entity_pages(total_pages=50)

        # Intro: 1-5, Outro: 36-50 (50 - 15 + 1 = 36)
        assert 5 not in pages[5:]  # No overlap
        assert len(pages) == 20

    def test_boundary_case_exactly_20_pages(self):
        """Document exactly 20 pages reads all."""
        pages = _compute_entity_pages(total_pages=20)

        # Should read all 20 pages
        assert pages == list(range(1, 21))

    def test_boundary_case_21_pages(self):
        """Document with 21 pages has bounded pages."""
        pages = _compute_entity_pages(total_pages=21)

        # Intro: 1-5, Outro: 7-21 (21 - 15 + 1 = 7)
        # But max(..., intro + 1) = max(7, 6) = 7
        assert 1 in pages
        assert 5 in pages
        assert 21 in pages
        assert len(pages) == 20


class TestEntityPagesBounded:
    """Tests ensuring entity pages never exceed the configured limit."""

    def test_entity_pages_never_exceed_20(self):
        """Entity pages never exceed intro + outro count."""
        max_expected = UmbrellaConfig.ENTITY_INTRO_PAGES + UmbrellaConfig.ENTITY_OUTRO_PAGES

        for total_pages in [10, 20, 50, 100, 261, 500, 1000]:
            pages = _compute_entity_pages(total_pages)
            assert len(pages) <= max_expected, f"Failed for {total_pages} pages"

    def test_entity_pages_always_sorted(self):
        """Entity pages are always sorted."""
        for total_pages in [15, 30, 100, 261]:
            pages = _compute_entity_pages(total_pages)
            assert pages == sorted(pages)


# =============================================================================
# Constraint Pages Tests
# =============================================================================


class TestFindConstraintSections:
    """Tests for _find_constraint_sections function."""

    def test_finds_investment_restrictions_section(self):
        """Finds section titled 'Investment Restrictions'."""
        skeleton = DocumentSkeleton(
            total_pages=200,
            sections=[
                SectionInfo(name="Introduction", page_start=1, page_end=10),
                SectionInfo(name="Investment Restrictions", page_start=100, page_end=120),
                SectionInfo(name="Fees and Charges", page_start=180, page_end=200),
            ],
        )

        pages = _find_constraint_sections(skeleton)

        # Should find Investment Restrictions but not Fees (non-matching title)
        assert pages == list(range(100, 121))

    def test_finds_leverage_section(self):
        """Finds section containing 'leverage' in title."""
        skeleton = DocumentSkeleton(
            total_pages=200,
            sections=[
                SectionInfo(name="Use of Leverage", page_start=50, page_end=55),
            ],
        )

        pages = _find_constraint_sections(skeleton)

        assert pages == list(range(50, 56))

    def test_finds_multiple_constraint_sections(self):
        """Finds multiple sections matching constraint patterns."""
        skeleton = DocumentSkeleton(
            total_pages=200,
            sections=[
                SectionInfo(name="Investment Policy", page_start=30, page_end=40),
                SectionInfo(name="Risk Management", page_start=60, page_end=70),
                SectionInfo(name="Derivative Instruments", page_start=80, page_end=90),
            ],
        )

        pages = _find_constraint_sections(skeleton)

        # Should include all three sections
        assert 30 in pages
        assert 40 in pages
        assert 60 in pages
        assert 70 in pages
        assert 80 in pages
        assert 90 in pages

    def test_case_insensitive_matching(self):
        """Pattern matching is case-insensitive."""
        skeleton = DocumentSkeleton(
            total_pages=100,
            sections=[
                SectionInfo(name="INVESTMENT RESTRICTIONS", page_start=20, page_end=30),
            ],
        )

        pages = _find_constraint_sections(skeleton)

        assert pages == list(range(20, 31))

    def test_no_matching_sections(self):
        """Returns empty list when no sections match."""
        skeleton = DocumentSkeleton(
            total_pages=100,
            sections=[
                SectionInfo(name="Introduction", page_start=1, page_end=10),
                SectionInfo(name="Fund Overview", page_start=11, page_end=50),
            ],
        )

        pages = _find_constraint_sections(skeleton)

        assert pages == []

    def test_empty_skeleton(self):
        """Returns empty list for empty skeleton."""
        skeleton = DocumentSkeleton(total_pages=100, sections=[])

        pages = _find_constraint_sections(skeleton)

        assert pages == []

    def test_none_skeleton(self):
        """Returns empty list for None skeleton."""
        pages = _find_constraint_sections(None)

        assert pages == []

    def test_caps_at_max_pages(self):
        """Caps constraint pages at CONSTRAINT_MAX_PAGES."""
        # Create skeleton with many large sections
        sections = [
            SectionInfo(name=f"Investment Policy Part {i}", page_start=i * 20, page_end=(i + 1) * 20 - 1)
            for i in range(10)
        ]
        skeleton = DocumentSkeleton(total_pages=500, sections=sections)

        pages = _find_constraint_sections(skeleton)

        assert len(pages) <= UmbrellaConfig.CONSTRAINT_MAX_PAGES


# =============================================================================
# Planner Validation Tests
# =============================================================================


class TestValidateUmbrellaPages:
    """Tests for validate_umbrella_pages function."""

    def test_valid_small_list(self):
        """Small page list is valid."""
        is_valid, reason = validate_umbrella_pages([1, 2, 3, 4, 5], total_pages=100)

        assert is_valid is True
        assert reason == "valid"

    def test_empty_list_is_valid(self):
        """Empty list is valid (will use heuristic)."""
        is_valid, reason = validate_umbrella_pages([], total_pages=100)

        assert is_valid is True
        assert "heuristic" in reason

    def test_reject_excessive_pages(self):
        """Rejects page list exceeding MAX_UMBRELLA_PAGES."""
        # Create list of 50 pages (> 30 limit)
        pages = list(range(1, 51))

        is_valid, reason = validate_umbrella_pages(pages, total_pages=100)

        assert is_valid is False
        assert "excessive" in reason

    def test_reject_nearly_all_pages(self):
        """Rejects page list containing nearly all document pages."""
        # 261 pages out of 261 total (the original bug case)
        pages = list(range(1, 262))

        is_valid, reason = validate_umbrella_pages(pages, total_pages=261)

        assert is_valid is False
        # Will be rejected for "excessive" (> MAX) before checking "nearly all"
        assert "excessive" in reason or "nearly all" in reason

    def test_reject_large_span(self):
        """Rejects page list with span > 80% of document."""
        # Pages 1-250 in a 261-page document
        pages = [1, 100, 250]  # Span is 250, which is > 80% of 261

        is_valid, reason = validate_umbrella_pages(pages, total_pages=261)

        assert is_valid is False
        assert "span" in reason

    def test_valid_at_boundary(self):
        """Page list at MAX_UMBRELLA_PAGES is valid."""
        max_pages = ExtractionConfig.MAX_UMBRELLA_PAGES
        pages = list(range(1, max_pages + 1))

        is_valid, reason = validate_umbrella_pages(pages, total_pages=100)

        assert is_valid is True


class TestRejectExcessivePages:
    """Tests ensuring excessive planner output is rejected."""

    def test_261_page_planner_output_rejected(self):
        """The original bug case: planner returns ALL 261 pages."""
        pages = list(range(1, 262))

        is_valid, reason = validate_umbrella_pages(pages, total_pages=261)

        assert is_valid is False

    def test_90_percent_threshold(self):
        """Pages at exactly 90% of document are rejected."""
        total = 100
        pages = list(range(1, 91))  # 90 pages = 90%

        is_valid, reason = validate_umbrella_pages(pages, total_pages=total)

        assert is_valid is False

    def test_89_percent_not_rejected(self):
        """Pages at 89% (below threshold) might be rejected for other reasons."""
        total = 100
        pages = list(range(1, 90))  # 89 pages = 89%

        # This might still be rejected for exceeding MAX_UMBRELLA_PAGES (30)
        is_valid, _ = validate_umbrella_pages(pages, total_pages=total)

        # With 89 pages > 30 max, it should be rejected
        assert is_valid is False


# =============================================================================
# Token Estimation Tests
# =============================================================================


class TestEstimateTokens:
    """Tests for estimate_tokens function."""

    def test_basic_estimation(self):
        """Basic token estimation calculation."""
        # 10 pages * 3000 chars/page / 4 chars/token = 7500 tokens
        tokens = estimate_tokens(10, chars_per_page=3000)

        assert tokens == 7500

    def test_zero_pages(self):
        """Zero pages returns zero tokens."""
        tokens = estimate_tokens(0)

        assert tokens == 0

    def test_large_document(self):
        """Large document estimation."""
        # 261 pages * 3000 chars/page / 4 chars/token = 195,750 tokens
        tokens = estimate_tokens(261, chars_per_page=3000)

        assert tokens == 195750


class TestCheckTokenLimit:
    """Tests for check_token_limit function."""

    def test_within_limit(self):
        """Pages within token limit."""
        is_safe, tokens = check_token_limit(20, chars_per_page=3000)

        assert is_safe is True
        assert tokens == 15000

    def test_exceeds_limit(self):
        """Pages exceeding token limit."""
        # 200 pages * 3000 / 4 = 150,000 tokens > 80,000 limit
        is_safe, tokens = check_token_limit(200, chars_per_page=3000)

        assert is_safe is False
        assert tokens == 150000

    def test_custom_limit(self):
        """Custom token limit is respected."""
        is_safe, _ = check_token_limit(10, max_tokens=5000)

        # 10 * 3000 / 4 = 7500 > 5000
        assert is_safe is False


# =============================================================================
# Full Page Selection Tests
# =============================================================================


class TestSelectUmbrellaPages:
    """Tests for select_umbrella_pages function."""

    def test_large_document_with_toc(self):
        """Large document with TOC uses TOC-guided constraints."""
        skeleton = DocumentSkeleton(
            total_pages=261,
            sections=[
                SectionInfo(name="Investment Restrictions", page_start=100, page_end=120),
            ],
        )

        plan = select_umbrella_pages(skeleton, total_pages=261)

        assert len(plan.entity_pages) == 20  # Bounded intro + outro
        assert plan.constraint_pages == list(range(100, 121))
        assert plan.source == "toc"

    def test_large_document_without_toc(self):
        """Large document without TOC uses heuristic."""
        plan = select_umbrella_pages(skeleton=None, total_pages=261)

        assert len(plan.entity_pages) == 20  # Bounded intro + outro
        assert plan.constraint_pages == []  # Will use search fallback
        assert plan.source == "heuristic"

    def test_small_document(self):
        """Small document reads all pages."""
        plan = select_umbrella_pages(skeleton=None, total_pages=15)

        assert plan.entity_pages == list(range(1, 16))
        assert plan.constraint_pages == []
        assert plan.source == "small_doc"

    def test_plan_summary(self):
        """Plan summary is human-readable."""
        plan = UmbrellaPagePlan(
            entity_pages=[1, 2, 3, 4, 5],
            constraint_pages=[50, 51, 52],
            source="toc",
            estimated_tokens=6000,
        )

        summary = plan.summary()

        assert "entity=5 pages" in summary
        assert "constraint=3 pages" in summary
        assert "toc" in summary

    def test_all_pages_property(self):
        """all_pages returns unique sorted pages."""
        plan = UmbrellaPagePlan(
            entity_pages=[1, 2, 3, 50, 51],
            constraint_pages=[50, 51, 52, 53],  # Overlap at 50, 51
        )

        all_pages = plan.all_pages

        assert all_pages == [1, 2, 3, 50, 51, 52, 53]
        assert plan.total_pages == 7


# =============================================================================
# Utility Function Tests
# =============================================================================


class TestContiguousBlocks:
    """Tests for _contiguous_blocks formatting function."""

    def test_single_block(self):
        """Single contiguous block."""
        result = _contiguous_blocks([1, 2, 3, 4, 5])

        assert result == "1-5"

    def test_multiple_blocks(self):
        """Multiple contiguous blocks."""
        result = _contiguous_blocks([1, 2, 3, 50, 51, 52])

        assert result == "1-3, 50-52"

    def test_single_pages(self):
        """Single pages (no blocks)."""
        result = _contiguous_blocks([1, 5, 10])

        assert result == "1, 5, 10"

    def test_mixed(self):
        """Mix of blocks and single pages."""
        result = _contiguous_blocks([1, 2, 3, 10, 20, 21, 22])

        assert result == "1-3, 10, 20-22"

    def test_empty_list(self):
        """Empty list returns 'none'."""
        result = _contiguous_blocks([])

        assert result == "none"


# =============================================================================
# Integration Tests
# =============================================================================


class TestUmbrellaExtractionIntegration:
    """Integration tests for umbrella extraction components."""

    def test_full_workflow_large_document(self):
        """Full workflow for large document like JPM prospectus."""
        total_pages = 261

        # Simulate skeleton with constraint sections
        skeleton = DocumentSkeleton(
            total_pages=total_pages,
            toc_source="native",
            sections=[
                SectionInfo(name="Legal Information", page_start=1, page_end=10),
                SectionInfo(name="Investment Objectives and Policies", page_start=50, page_end=80),
                SectionInfo(name="Investment Restrictions", page_start=81, page_end=100),
                SectionInfo(name="Risk Management", page_start=101, page_end=110),
                SectionInfo(name="Fund Details", page_start=111, page_end=240),
                SectionInfo(name="Service Providers", page_start=241, page_end=261),
            ],
        )

        # Get page plan
        plan = select_umbrella_pages(skeleton, total_pages)

        # Verify entity pages are bounded
        assert len(plan.entity_pages) <= 20

        # Verify constraint pages come from TOC
        assert plan.source == "toc"
        assert len(plan.constraint_pages) > 0
        assert len(plan.constraint_pages) <= UmbrellaConfig.CONSTRAINT_MAX_PAGES

        # Verify token estimate is reasonable
        is_safe, _ = check_token_limit(plan.total_pages)
        assert is_safe, f"Page plan would exceed token limit: {plan.total_pages} pages"

    def test_full_workflow_small_document(self):
        """Full workflow for small document."""
        total_pages = 18

        plan = select_umbrella_pages(None, total_pages)

        # Small doc should read everything
        assert plan.source == "small_doc"
        assert plan.entity_pages == list(range(1, 19))
        assert plan.constraint_pages == []
