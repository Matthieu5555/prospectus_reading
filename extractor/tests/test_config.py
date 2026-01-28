"""Tests for extractor.core.config module.

Tests the centralized configuration:
- ConfidenceThresholds: Scoring thresholds
- SearchLimits: Maximum search results
- PageLimits: Maximum pages to read
- ChunkingConfig: Document chunking settings
- DEFAULT_MODELS: Model configurations
"""

from extractor.core.config import (
    ConfidenceThresholds,
    SearchLimits,
    PageLimits,
    ChunkingConfig,
    SearchPatterns,
    DEFAULT_MODELS,
    LLMConfig,
    RetryConfig,
    RegexPatterns,
    ExtractionConfig,
)


# =============================================================================
# ConfidenceThresholds tests
# =============================================================================


class TestConfidenceThresholds:
    """Tests for ConfidenceThresholds configuration."""

    def test_high_threshold_is_valid(self):
        """HIGH threshold should be between 0.5 and 1.0."""
        assert 0.5 < ConfidenceThresholds.HIGH <= 1.0

    def test_low_threshold_is_less_than_high(self):
        """LOW threshold should be less than HIGH."""
        assert ConfidenceThresholds.LOW < ConfidenceThresholds.HIGH

    def test_fuzzy_match_is_reasonable(self):
        """FUZZY_MATCH threshold should be high enough to avoid false positives."""
        assert ConfidenceThresholds.FUZZY_MATCH >= 0.8

    def test_fallback_is_conservative(self):
        """FALLBACK should indicate uncertainty."""
        assert ConfidenceThresholds.FALLBACK <= ConfidenceThresholds.HIGH

    def test_thresholds_are_floats(self):
        """All thresholds should be floats."""
        assert isinstance(ConfidenceThresholds.HIGH, float)
        assert isinstance(ConfidenceThresholds.LOW, float)
        assert isinstance(ConfidenceThresholds.FUZZY_MATCH, float)


# =============================================================================
# SearchLimits tests
# =============================================================================


class TestSearchLimits:
    """Tests for SearchLimits configuration."""

    def test_isin_search_is_reasonable(self):
        """ISIN_SEARCH limit should allow finding all ISINs in a document."""
        assert SearchLimits.ISIN_SEARCH >= 20

    def test_fee_search_is_reasonable(self):
        """FEE_SEARCH limit should allow finding fee tables."""
        assert SearchLimits.FEE_SEARCH >= 10

    def test_constraint_search_is_reasonable(self):
        """CONSTRAINT_SEARCH limit should allow finding restrictions."""
        assert SearchLimits.CONSTRAINT_SEARCH >= 10

    def test_limits_are_integers(self):
        """All limits should be integers."""
        assert isinstance(SearchLimits.ISIN_SEARCH, int)
        assert isinstance(SearchLimits.FEE_SEARCH, int)
        assert isinstance(SearchLimits.CONSTRAINT_SEARCH, int)


# =============================================================================
# PageLimits tests
# =============================================================================


class TestPageLimits:
    """Tests for PageLimits configuration."""

    def test_isin_pages_is_reasonable(self):
        """ISIN_PAGES limit should balance coverage vs token cost."""
        assert 5 <= PageLimits.ISIN_PAGES <= 20

    def test_fee_pages_is_reasonable(self):
        """FEE_PAGES limit should balance coverage vs token cost."""
        assert 5 <= PageLimits.FEE_PAGES <= 20

    def test_limits_are_integers(self):
        """All limits should be integers."""
        assert isinstance(PageLimits.ISIN_PAGES, int)
        assert isinstance(PageLimits.FEE_PAGES, int)


# =============================================================================
# ChunkingConfig tests
# =============================================================================


class TestChunkingConfig:
    """Tests for ChunkingConfig configuration."""

    def test_chunk_size_is_reasonable(self):
        """CHUNK_SIZE should balance context vs token cost."""
        assert 20 <= ChunkingConfig.CHUNK_SIZE <= 50

    def test_min_chunk_size_less_than_chunk_size(self):
        """MIN_CHUNK_SIZE should be less than CHUNK_SIZE."""
        assert ChunkingConfig.MIN_CHUNK_SIZE < ChunkingConfig.CHUNK_SIZE

    def test_overlap_is_reasonable(self):
        """OVERLAP should be small but meaningful."""
        assert 1 <= ChunkingConfig.OVERLAP <= 10

    def test_overlap_less_than_min_chunk(self):
        """OVERLAP should be less than MIN_CHUNK_SIZE to avoid infinite loops."""
        assert ChunkingConfig.OVERLAP < ChunkingConfig.MIN_CHUNK_SIZE


# =============================================================================
# SearchPatterns tests
# =============================================================================


class TestSearchPatterns:
    """Tests for SearchPatterns configuration."""

    def test_isin_patterns_include_common_prefixes(self):
        """ISIN patterns should include common Luxembourg/Ireland prefixes."""
        assert "LU0" in SearchPatterns.ISIN
        assert "IE00" in SearchPatterns.ISIN

    def test_fee_patterns_include_key_terms(self):
        """FEE patterns should include common fee terminology."""
        fee_lower = [p.lower() for p in SearchPatterns.FEE]
        assert any("management" in p for p in fee_lower)
        assert any("fee" in p for p in fee_lower)

    def test_patterns_are_lists(self):
        """All pattern categories should be lists."""
        assert isinstance(SearchPatterns.ISIN, list)
        assert isinstance(SearchPatterns.FEE, list)
        assert isinstance(SearchPatterns.RESTRICTION, list)


# =============================================================================
# DEFAULT_MODELS tests
# =============================================================================


class TestDefaultModels:
    """Tests for DEFAULT_MODELS configuration."""

    def test_all_agent_types_have_models(self):
        """All required agent types should have configured models."""
        required_agents = ["exploration", "planning", "reader", "critic"]
        for agent in required_agents:
            assert agent in DEFAULT_MODELS
            assert isinstance(DEFAULT_MODELS[agent], str)
            assert len(DEFAULT_MODELS[agent]) > 0

    def test_models_are_valid_format(self):
        """Model strings should follow provider/model format."""
        for agent, model in DEFAULT_MODELS.items():
            # Should contain provider prefix
            assert "/" in model, f"Model {model} for {agent} should have provider prefix"


# =============================================================================
# LLMConfig tests
# =============================================================================


class TestLLMConfig:
    """Tests for LLMConfig configuration."""

    def test_temperature_is_valid(self):
        """TEMPERATURE should be between 0 and 1."""
        assert 0 <= LLMConfig.TEMPERATURE <= 1

    def test_json_format_is_correct(self):
        """RESPONSE_FORMAT should specify JSON output."""
        assert LLMConfig.RESPONSE_FORMAT == {"type": "json_object"}


# =============================================================================
# RetryConfig tests
# =============================================================================


class TestRetryConfig:
    """Tests for RetryConfig configuration."""

    def test_max_attempts_is_reasonable(self):
        """MAX_ATTEMPTS should allow recovery from transient failures."""
        assert 1 <= RetryConfig.MAX_ATTEMPTS <= 10

    def test_min_wait_is_positive(self):
        """MIN_WAIT_SECONDS should be positive for backoff to work."""
        assert RetryConfig.MIN_WAIT_SECONDS > 0

    def test_max_wait_greater_than_min(self):
        """MAX_WAIT_SECONDS should be greater than MIN_WAIT_SECONDS."""
        assert RetryConfig.MAX_WAIT_SECONDS > RetryConfig.MIN_WAIT_SECONDS


# =============================================================================
# RegexPatterns tests
# =============================================================================


class TestRegexPatterns:
    """Tests for RegexPatterns configuration."""

    def test_isin_pattern_matches_valid_isins(self):
        """ISIN pattern should match valid ISIN codes."""
        import re
        pattern = re.compile(RegexPatterns.ISIN)

        valid_isins = ["LU0123456789", "IE00B1234567", "FR0012345678"]
        for isin in valid_isins:
            assert pattern.search(isin), f"Pattern should match {isin}"

    def test_date_ordinal_pattern_matches_dates(self):
        """DATE_ORDINAL pattern should match ordinal date formats."""
        import re
        pattern = re.compile(RegexPatterns.DATE_ORDINAL)

        valid_dates = ["15th March", "1st January", "22nd Feb", "3rd Apr"]
        for date in valid_dates:
            assert pattern.search(date), f"Pattern should match {date}"


# =============================================================================
# ExtractionConfig tests
# =============================================================================


class TestExtractionConfig:
    """Tests for ExtractionConfig configuration."""

    def test_umbrella_end_pages_is_reasonable(self):
        """UMBRELLA_END_PAGES should cover typical umbrella sections."""
        assert 5 <= ExtractionConfig.UMBRELLA_END_PAGES <= 20

    def test_umbrella_min_gap_is_positive(self):
        """UMBRELLA_MIN_GAP should be positive."""
        assert ExtractionConfig.UMBRELLA_MIN_GAP > 0

    def test_priority_thresholds_are_ordered(self):
        """Priority thresholds should be ordered correctly."""
        assert ExtractionConfig.PRIORITY_MEDIUM_THRESHOLD < ExtractionConfig.PRIORITY_HIGH_THRESHOLD
