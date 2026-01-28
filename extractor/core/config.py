"""Centralized configuration for the extraction pipeline.

All magic numbers, thresholds, and configuration constants are documented here.
Each constant includes:
- What it controls
- Why this value was chosen
- What changing it affects
"""

import os
from typing import Final


# =============================================================================
# LLM Provider Configuration
# =============================================================================
#
# To switch providers, set the LLM_PROVIDER environment variable:
#   - "openrouter" (default): Uses OpenRouter API gateway
#   - "azure": Uses Azure OpenAI Service
#
# For Azure, also set:
#   - AZURE_API_KEY: Your Azure OpenAI API key
#   - AZURE_API_BASE: Your Azure endpoint (e.g., https://your-resource.openai.azure.com/)
#   - AZURE_API_VERSION: API version (e.g., 2024-02-15-preview)
#   - AZURE_DEPLOYMENT_SMART: Deployment name for smart model (default: gpt-4o)
#   - AZURE_DEPLOYMENT_FAST: Deployment name for fast model (default: gpt-4o-mini)
#
# =============================================================================

LLM_PROVIDER: Final[str] = os.environ.get("LLM_PROVIDER", "openrouter")
"""LLM provider to use. Set via LLM_PROVIDER env var.

Supported values:
- "openrouter": OpenRouter API gateway (default)
- "azure": Azure OpenAI Service
"""

# Provider-specific API key environment variables
API_KEY_ENV_VARS: Final[dict[str, str]] = {
    "openrouter": "OPENROUTER_API_KEY",
    "azure": "AZURE_API_KEY",
}

API_KEY_ENV_VAR: Final[str] = API_KEY_ENV_VARS.get(LLM_PROVIDER, "OPENROUTER_API_KEY")
"""Environment variable name for the LLM API key (provider-dependent)."""


# =============================================================================
# Model Configuration (Provider-Specific)
# =============================================================================

def _get_model_name(base_model: str) -> str:
    """Convert a base model name to provider-specific format.

    Args:
        base_model: Base model name (e.g., "gpt-4o", "gpt-4o-mini")

    Returns:
        Provider-specific model identifier.
    """
    if LLM_PROVIDER == "azure":
        # Azure uses deployment names. Check for env override, else use base name.
        deployment_env = f"AZURE_DEPLOYMENT_{base_model.upper().replace('-', '_')}"
        return f"azure/{os.environ.get(deployment_env, base_model)}"
    else:
        # OpenRouter uses openrouter/provider/model format
        return f"openrouter/openai/{base_model}"


SMART_MODEL: Final[str] = _get_model_name("gpt-4o")
"""Default 'smart' model for high-impact phases (planning, exploration).

Use --smart-model CLI flag to override. The smart model is used for phases
where reasoning quality significantly impacts downstream extraction quality.

For Azure, override deployment name with AZURE_DEPLOYMENT_GPT_4O env var.
"""

FAST_MODEL: Final[str] = _get_model_name("gpt-4o-mini")
"""Default 'fast' model for routine extraction work.

Used for per-fund extraction, critic verification, and other high-volume tasks
where cost efficiency matters more than reasoning depth.

For Azure, override deployment name with AZURE_DEPLOYMENT_GPT_4O_MINI env var.
"""

DEFAULT_MODELS: Final[dict[str, str]] = {
    "exploration": FAST_MODEL,
    "planning": FAST_MODEL,
    "reader": FAST_MODEL,
    "critic": FAST_MODEL,
}
"""Default LLM models for each agent type.

All default to the fast model for cost efficiency. Use --smart-model to enable
smarter models for high-impact phases (exploration, planning).

Why default everything to the fast model when SMART_MODEL exists? Because most
users run this on their own API budget. A single prospectus extraction can make
100+ LLM calls—defaulting to GPT-4 would surprise users with expensive bills.
The --smart-model flag is an opt-in to quality over cost. This is a conservative
default: better to start cheap and let users upgrade than to burn through their
API credits unexpectedly. (This could have been a single MODEL constant, but the
two-tier approach lets power users fine-tune cost/quality tradeoffs per phase.)

To use more capable models for specific agents:
    orchestrator = Orchestrator(pdf_path, smart_model="openrouter/openai/gpt-4o")
"""


# Confidence Thresholds

class ConfidenceThresholds:
    """Thresholds for confidence scoring throughout the pipeline.

    These thresholds classify extraction confidence into HIGH/MEDIUM/LOW:
    - HIGH (>= 0.8): Reliable extraction, suitable for automated processing
    - MEDIUM (0.5-0.8): Usable but may benefit from human review
    - LOW (< 0.5): Likely needs manual verification

    Used by: assembly_phase.py, render_extracted_values.py, critic_agent.py
    """

    HIGH: Final[float] = 0.8
    """Extractions >= this are considered reliable. Used for metadata stats."""

    LOW: Final[float] = 0.5
    """Extractions < this trigger warnings. Used for metadata stats."""

    FUZZY_MATCH: Final[float] = 0.85
    """Minimum SequenceMatcher ratio for fund name fuzzy matching.

    Lower values (e.g., 0.7) match more liberally but risk false positives.
    Higher values (e.g., 0.95) are stricter but may miss valid matches.
    0.85 balances precision and recall for fund name variants like
    "Global Bond Fund" vs "JPMorgan Investment Funds - Global Bond Fund".

    Used by: utils.py:fund_names_match()
    """

    UMBRELLA_GAP: Final[float] = 0.8
    """Threshold for detecting umbrella-level fields.

    If > 80% of funds have NOT_FOUND for a field, it's likely an
    umbrella-level field that should be extracted once and inherited.

    Used by: assembly_phase.py gap analysis
    """

    DEFAULT_UNVERIFIED: Final[float] = 0.5
    """Default confidence for extractions without full provenance.

    Applied to legacy/migrated extractions that lack source quotes.

    Used by: provenance.py:from_simple()
    """

    FALLBACK: Final[float] = 0.5
    """Fallback confidence when critic cannot determine confidence.

    Used by: critic_agent.py when confidence computation fails
    """


# Search Configuration

class SearchLimits:
    """Maximum results for different search operations.

    These control token usage by limiting how many search hits we process.
    Higher values = more thorough search but higher API costs.

    Trade-off: Completeness vs. cost. Most prospectuses have ISINs/fees
    concentrated in specific sections, so these limits rarely truncate
    relevant results.
    """

    ISIN_SEARCH: Final[int] = 50
    """Max ISIN search hits. ISINs can be scattered in appendices.
    Used by: reader_agent.py:search_and_extract_isins()
    """

    FEE_SEARCH: Final[int] = 30
    """Max fee search hits. Fees usually in dedicated tables.
    Used by: reader_agent.py:search_and_extract_fees()
    """

    CONSTRAINT_SEARCH: Final[int] = 15
    """Max constraint search hits per category. Constraints in policy sections.
    Used by: reader_agent.py:search_and_extract_constraints()
    """

    GENERAL_SEARCH: Final[int] = 20
    """Max hits for assembly phase gap-filling searches.
    Used by: assembly_phase.py umbrella constraint extraction
    """

    FIELD_RESOLVER: Final[int] = 10
    """Max hits per pattern in field resolvers.
    Used by: field_searchers.py resolver classes
    """

    DEFAULT: Final[int] = 50
    """Default max_results for general searches.
    Used by: search_context.py, pdf_reader.py
    """


class PageLimits:
    """Maximum pages to read for different operations.

    Prevents token explosion when search returns many hits across
    different pages. Reading too many pages increases cost and may
    confuse the LLM with irrelevant content.
    """

    ISIN_PAGES: Final[int] = 10
    """Max pages to read for ISIN extraction.
    Used by: reader_agent.py:search_and_extract_isins()
    """

    FEE_PAGES: Final[int] = 10
    """Max pages to read for fee extraction.
    Used by: reader_agent.py:search_and_extract_fees()
    """

    CONSTRAINT_PAGES: Final[int] = 10
    """Max pages for constraint extraction.
    Used by: reader_agent.py:search_and_extract_constraints()
    """

    GAP_FILL_PAGES: Final[int] = 15
    """Max pages for umbrella gap-filling in assembly phase.
    Used by: assembly_phase.py
    """

    FIELD_RESOLVER_PAGES: Final[int] = 15
    """Max pages for field resolvers to search.
    Used by: field_searchers.py
    """


# Chunking Configuration

class ChunkingConfig:
    """Configuration for document chunking during exploration.

    The document is split into chunks for parallel exploration.
    Chunk boundaries try to respect document structure (sections).

    Trade-offs:
    - Larger chunks: Better context but higher per-call token cost
    - Smaller chunks: More parallelism but may split related content
    - Overlap: Prevents missing content at boundaries but adds redundancy

    30 pages balances these concerns for typical 200-400 page prospectuses.
    """

    CHUNK_SIZE: Final[int] = 30
    """Target pages per exploration chunk.

    Used by: phase_base.py, orchestrator.py, cli.py, smart_chunker.py
    """

    MIN_CHUNK_SIZE: Final[int] = 10
    """Don't create chunks smaller than this (except final chunk).

    Prevents tiny chunks that lack sufficient context.
    Used by: smart_chunker.py
    """

    OVERLAP: Final[int] = 5
    """Pages of overlap between adjacent chunks.

    Ensures content at chunk boundaries isn't missed. 5 pages
    typically covers a complete section transition.
    Used by: smart_chunker.py
    """


def adaptive_chunk_size(total_pages: int, base_size: int = 30) -> int:
    """Calculate chunk size based on document length.

    Larger documents tend to have denser content and more complex structure.
    Smaller chunks prevent token overflow and improve exploration accuracy.

    Args:
        total_pages: Total document pages.
        base_size: Default chunk size for normal documents (default 30).

    Returns:
        Recommended chunk size (minimum 10, maximum base_size).

    Examples:
        >>> adaptive_chunk_size(50)   # Small doc
        30
        >>> adaptive_chunk_size(150)  # Medium doc
        25
        >>> adaptive_chunk_size(400)  # Large doc
        15
        >>> adaptive_chunk_size(600)  # Very large doc
        10
    """
    if total_pages <= 100:
        return base_size  # 30 pages - normal
    elif total_pages <= 200:
        return 25  # Slightly smaller
    elif total_pages <= 350:
        return 20  # Medium reduction
    elif total_pages <= 500:
        return 15  # Significant reduction
    else:
        return 10  # Very large documents - small chunks


# Extraction Logic Constants

class UmbrellaConfig:
    """Configuration for robust umbrella extraction.

    Uses a two-pass approach to guarantee we never exceed token limits:
    1. Entity info (name, depositary, management_company): bounded intro+outro pages
    2. Constraints (investment restrictions, leverage): TOC-guided or search fallback

    Used by: umbrella_page_selector.py, extraction_phase.py
    """

    # Token estimation
    CHARS_PER_TOKEN: Final[int] = 4
    """Approximate characters per token for estimation.

    Used for pre-flight token checks before LLM calls.
    Conservative estimate (actual is ~3.5 for English text).
    """

    MAX_UMBRELLA_INPUT_TOKENS: Final[int] = 80_000
    """Maximum tokens for a single umbrella extraction LLM call.

    Safety limit to prevent token overflow errors (128k context - response).
    Used by: umbrella_page_selector.py:estimate_tokens()
    """

    # Entity info extraction (Pass 1) - always bounded
    ENTITY_INTRO_PAGES: Final[int] = 5
    """Pages from document start for entity info (name, legal form, domicile).

    First pages contain legal disclaimers, UCITS status, fund structure.
    """

    ENTITY_OUTRO_PAGES: Final[int] = 15
    """Pages from document end for service provider info.

    Management company, depositary, auditor often in appendices.
    """

    # Constraint extraction (Pass 2) - TOC-guided
    CONSTRAINT_SECTION_PATTERNS: Final[tuple[str, ...]] = (
        "investment restriction",
        "investment policy",
        "investment objective",
        "investment limit",
        "risk management",
        "leverage",
        "borrowing",
        "derivative",
        "general investment",
        "appendix",  # Often contains investment restrictions
    )
    """TOC section names that typically contain constraint information.

    Case-insensitive matching against section.name.lower().
    Used by: umbrella_page_selector.py:_find_constraint_sections()
    """

    CONSTRAINT_MAX_PAGES: Final[int] = 40
    """Maximum pages to read for constraint extraction.

    Even with TOC guidance, cap the page count to prevent overflow.
    """

    CONSTRAINT_SEARCH_FALLBACK_PAGES: Final[int] = 20
    """Pages to search when no TOC constraint sections found.

    Uses pattern-based search as fallback.
    """


class ExtractionConfig:
    """Constants for extraction phase logic."""

    MAX_UMBRELLA_PAGES: Final[int] = 30
    """Maximum pages to read for umbrella extraction.

    Prevents token explosion when planner incorrectly assigns too many pages.
    If umbrella_pages exceeds this, we fall back to exploration's umbrella_info_pages
    or heuristics (first 5 + last 15 pages).

    Trade-off: Lower = safer but may miss info; Higher = more complete but risky.
    30 pages is generous for legal entity info (typically 5-15 pages).

    Used by: planner_agent.py:enrich_plan_with_pages()
    """

    UMBRELLA_INTRO_PAGES: Final[int] = 5
    """Default number of intro pages to read for umbrella info.

    Used as fallback when no umbrella_info_pages found in exploration.
    First few pages typically contain legal structure, UCITS status.

    Used by: planner_agent.py:enrich_plan_with_pages()
    """

    UMBRELLA_END_PAGES: Final[int] = 15
    """Pages from document end to search for umbrella info.

    Some prospectuses have legal entity details in appendices.
    Used by: extraction_phase.py, planner_agent.py
    """

    UMBRELLA_MIN_GAP: Final[int] = 5
    """Minimum gap between beginning and end sections for umbrella extraction.

    Prevents reading overlapping sections in small documents.
    Used by: extraction_phase.py
    """

    SMALL_DOCUMENT_THRESHOLD: Final[int] = 20
    """Documents <= this page count are read entirely for umbrella info.

    Small documents don't need the begin/end section heuristic.
    Used by: extraction_phase.py
    """

    PRIORITY_HIGH_THRESHOLD: Final[float] = 0.8
    """If > 80% of funds missing a field, it's HIGH priority for resolution.
    Used by: extraction_phase.py field prioritization
    """

    PRIORITY_MEDIUM_THRESHOLD: Final[float] = 0.5
    """If > 50% of funds missing a field, it's MEDIUM priority.
    Used by: extraction_phase.py field prioritization
    """


# Search Patterns

class SearchPatterns:
    """Standard search patterns by category.

    Used by SearchContext and field resolvers to find relevant pages.
    These patterns cover common prospectus terminology across jurisdictions.
    """

    ISIN: Final[list[str]] = [
        "LU0", "LU1", "LU2",  # Luxembourg (most common for UCITS)
        "IE00", "IE0",        # Ireland
        "FR00",               # France
        "DE00",               # Germany
        "GB00",               # UK
        "CH0",                # Switzerland
        "ISIN",               # Generic keyword
    ]
    """ISIN code prefixes and keywords.
    Covers major European fund domiciles.
    """

    FEE: Final[list[str]] = [
        "management fee",
        "entry fee",
        "exit fee",
        "ongoing charges",
        "TER",                # Total Expense Ratio
        "OCF",                # Ongoing Charges Figure
        "expense ratio",
    ]
    """Fee-related terms found in prospectuses."""

    RESTRICTION: Final[list[str]] = [
        "investment restriction",
        "may not invest",
        "shall not",
        "maximum",
        "minimum",
        "limit",
    ]
    """Investment restriction keywords."""

    LEVERAGE: Final[list[str]] = [
        "leverage",
        "borrowing",
        "VaR",                # Value at Risk
        "commitment approach",
        "gross exposure",
    ]
    """Leverage policy keywords."""

    DERIVATIVE: Final[list[str]] = [
        "derivative",
        "hedging",
        "forward",
        "swap",
        "option",
        "future",
    ]
    """Derivatives usage keywords."""

    DIVIDEND: Final[list[str]] = [
        "dividend",
        "distribution",
        "payment date",
        "ex-dividend",
        "record date",
        "quarterly",
        "annually",
        "semi-annual",
    ]
    """Dividend/distribution keywords for field resolvers."""


# LLM Call Configuration

class LLMConfig:
    """Default parameters for LLM API calls."""

    TEMPERATURE: Final[float] = 0.0
    """Sampling temperature for all extraction calls.

    0.0 ensures deterministic, consistent extractions.
    Higher values add creativity but reduce reproducibility.
    """

    RESPONSE_FORMAT: Final[dict[str, str]] = {"type": "json_object"}
    """Response format enforcing JSON output."""


# Retry Configuration

class RetryConfig:
    """Configuration for LLM API retry behavior.

    Uses exponential backoff: wait times are 4s, 8s, 16s, 32s, 60s (capped).
    3 attempts handle most transient failures (rate limits, timeouts).
    """

    MAX_ATTEMPTS: Final[int] = 3
    """Maximum retry attempts before giving up."""

    MIN_WAIT_SECONDS: Final[int] = 4
    """Initial wait time before first retry."""

    MAX_WAIT_SECONDS: Final[int] = 60
    """Maximum wait time between retries (caps exponential growth)."""

    MULTIPLIER: Final[int] = 1
    """Exponential backoff multiplier."""


# Regex Patterns

class RegexPatterns:
    """Compiled regex patterns used for extraction."""

    ISIN: Final[str] = r"\b([A-Z]{2}[A-Z0-9]{10})\b"
    """ISIN format: 2 letters (country) + 10 alphanumeric characters.
    Used by: field_searchers.py ISINResolver
    """

    DATE_ORDINAL: Final[str] = (
        r"\d{1,2}(?:st|nd|rd|th)?\s+"
        r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|"
        r"May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|"
        r"Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
    )
    """Date pattern like "15th March" or "1 Jan".
    Used by: field_searchers.py DividendResolver
    """

    DATE_MONTH_DAY: Final[str] = (
        r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|"
        r"May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|"
        r"Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2}"
    )
    """Date pattern like "March 15" or "Jan 1".
    Used by: field_searchers.py DividendResolver
    """
