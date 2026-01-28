"""Smart page selection for umbrella extraction.

Solves the token overflow problem by splitting umbrella extraction into two
bounded passes:
1. Entity info (name, depositary, management_company): intro + outro pages
2. Constraints (investment restrictions, leverage): TOC-guided sections

This guarantees we never exceed token limits while maximizing coverage.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from extractor.core.config import UmbrellaConfig

if TYPE_CHECKING:
    from extractor.pydantic_models.exploration_models import DocumentSkeleton

logger = logging.getLogger(__name__)


@dataclass
class UmbrellaPagePlan:
    """Plan for which pages to read for umbrella extraction.

    Separates entity info pages (always bounded) from constraint pages
    (TOC-guided with fallback).

    Attributes:
        entity_pages: Pages for Pass 1 - entity info (name, depositary, etc.)
        constraint_pages: Pages for Pass 2 - constraints (leverage, restrictions)
        source: How the plan was computed ('toc', 'heuristic', 'small_doc')
        estimated_tokens: Estimated total tokens for both passes
    """

    entity_pages: list[int] = field(default_factory=list)
    constraint_pages: list[int] = field(default_factory=list)
    source: str = "heuristic"
    estimated_tokens: int = 0

    @property
    def all_pages(self) -> list[int]:
        """All unique pages across both passes, sorted."""
        return sorted(set(self.entity_pages) | set(self.constraint_pages))

    @property
    def total_pages(self) -> int:
        """Total unique pages to read."""
        return len(self.all_pages)

    def summary(self) -> str:
        """Human-readable summary of the plan."""
        return (
            f"UmbrellaPagePlan(source={self.source}, "
            f"entity={len(self.entity_pages)} pages, "
            f"constraint={len(self.constraint_pages)} pages, "
            f"~{self.estimated_tokens:,} tokens)"
        )


def select_umbrella_pages(
    skeleton: DocumentSkeleton | None,
    total_pages: int,
    chars_per_page: int = 3000,
) -> UmbrellaPagePlan:
    """Select pages for umbrella extraction using two-pass approach.

    Strategy:
    1. Entity pages: always bounded intro + outro (first 5 + last 15)
    2. Constraint pages: TOC-guided if skeleton available, else empty (use search)

    Args:
        skeleton: DocumentSkeleton with TOC sections (may be None).
        total_pages: Total pages in document.
        chars_per_page: Estimated chars per page for token calculation.

    Returns:
        UmbrellaPagePlan with separate page lists for each pass.
    """
    # Pass 1: Entity info pages (always bounded)
    entity_pages = _compute_entity_pages(total_pages)

    # Pass 2: Constraint pages (TOC-guided)
    constraint_pages: list[int] = []
    source = "heuristic"

    if skeleton and skeleton.sections:
        constraint_pages = _find_constraint_sections(skeleton)
        if constraint_pages:
            source = "toc"
            logger.info(
                f"Found {len(constraint_pages)} constraint pages from TOC: "
                f"{_contiguous_blocks(constraint_pages)}"
            )

    # If no TOC sections found, we'll use search-based fallback in extraction
    # (don't pre-select pages here - let the search find relevant pages)

    # Estimate tokens
    all_pages = sorted(set(entity_pages) | set(constraint_pages))
    estimated_tokens = estimate_tokens(len(all_pages), chars_per_page)

    # Check if small document - read entire thing
    if total_pages <= UmbrellaConfig.ENTITY_INTRO_PAGES + UmbrellaConfig.ENTITY_OUTRO_PAGES:
        entity_pages = list(range(1, total_pages + 1))
        constraint_pages = []  # No need for separate constraint pass
        source = "small_doc"
        estimated_tokens = estimate_tokens(total_pages, chars_per_page)

    plan = UmbrellaPagePlan(
        entity_pages=entity_pages,
        constraint_pages=constraint_pages,
        source=source,
        estimated_tokens=estimated_tokens,
    )

    logger.info(f"Umbrella page plan: {plan.summary()}")
    return plan


def _compute_entity_pages(total_pages: int) -> list[int]:
    """Compute bounded intro + outro pages for entity info extraction.

    Always returns at most ENTITY_INTRO_PAGES + ENTITY_OUTRO_PAGES pages.
    Handles document overlap for small documents.

    Args:
        total_pages: Total pages in document.

    Returns:
        Sorted list of page numbers (1-indexed).
    """
    intro_count = UmbrellaConfig.ENTITY_INTRO_PAGES
    outro_count = UmbrellaConfig.ENTITY_OUTRO_PAGES

    # Small document: read everything
    if total_pages <= intro_count + outro_count:
        return list(range(1, total_pages + 1))

    # Intro pages: first N pages
    intro_pages = list(range(1, intro_count + 1))

    # Outro pages: last M pages (ensure no overlap with intro)
    outro_start = max(intro_count + 1, total_pages - outro_count + 1)
    outro_pages = list(range(outro_start, total_pages + 1))

    return sorted(set(intro_pages) | set(outro_pages))


def _find_constraint_sections(skeleton: DocumentSkeleton) -> list[int]:
    """Find pages containing constraint information using TOC sections.

    Matches section names against CONSTRAINT_SECTION_PATTERNS and returns
    the union of all matching section pages, capped at CONSTRAINT_MAX_PAGES.

    Args:
        skeleton: DocumentSkeleton with populated sections.

    Returns:
        Sorted list of page numbers (1-indexed).
    """
    if not skeleton or not skeleton.sections:
        return []

    patterns = UmbrellaConfig.CONSTRAINT_SECTION_PATTERNS
    max_pages = UmbrellaConfig.CONSTRAINT_MAX_PAGES

    matching_pages: set[int] = set()
    matched_sections: list[str] = []

    for section in skeleton.sections:
        section_name_lower = section.name.lower()

        # Check if section name matches any constraint pattern
        for pattern in patterns:
            if pattern in section_name_lower:
                # Add all pages in this section
                section_pages = list(range(section.page_start, section.page_end + 1))
                matching_pages.update(section_pages)
                matched_sections.append(f"{section.name} ({section.page_start}-{section.page_end})")
                break  # Don't double-count if multiple patterns match

    if matched_sections:
        logger.debug(f"Matched constraint sections: {matched_sections}")

    # Cap at max pages to prevent overflow
    sorted_pages = sorted(matching_pages)
    if len(sorted_pages) > max_pages:
        logger.warning(
            f"Constraint sections exceed {max_pages} pages ({len(sorted_pages)}), "
            f"truncating to first {max_pages}"
        )
        sorted_pages = sorted_pages[:max_pages]

    return sorted_pages


def estimate_tokens(page_count: int, chars_per_page: int = 3000) -> int:
    """Estimate token count for a given number of pages.

    Uses UmbrellaConfig.CHARS_PER_TOKEN for estimation.

    Args:
        page_count: Number of pages to read.
        chars_per_page: Estimated characters per page (default 3000).

    Returns:
        Estimated token count.
    """
    total_chars = page_count * chars_per_page
    return total_chars // UmbrellaConfig.CHARS_PER_TOKEN


def check_token_limit(
    page_count: int,
    chars_per_page: int = 3000,
    max_tokens: int | None = None,
) -> tuple[bool, int]:
    """Check if page count would exceed token limit.

    Args:
        page_count: Number of pages to read.
        chars_per_page: Estimated characters per page.
        max_tokens: Token limit (default: UmbrellaConfig.MAX_UMBRELLA_INPUT_TOKENS).

    Returns:
        Tuple of (is_within_limit, estimated_tokens).
    """
    if max_tokens is None:
        max_tokens = UmbrellaConfig.MAX_UMBRELLA_INPUT_TOKENS

    estimated = estimate_tokens(page_count, chars_per_page)
    return estimated <= max_tokens, estimated


def _contiguous_blocks(pages: list[int]) -> str:
    """Format pages as contiguous blocks for logging.

    Example: [1, 2, 3, 50, 51, 52] -> "1-3, 50-52"

    Args:
        pages: Sorted list of page numbers.

    Returns:
        Human-readable string of page ranges.
    """
    if not pages:
        return "none"

    blocks: list[str] = []
    start = pages[0]
    end = pages[0]

    for page in pages[1:]:
        if page == end + 1:
            end = page
        else:
            blocks.append(f"{start}-{end}" if start != end else str(start))
            start = page
            end = page

    blocks.append(f"{start}-{end}" if start != end else str(start))
    return ", ".join(blocks)


def validate_umbrella_pages(
    planner_pages: list[int],
    total_pages: int,
    max_allowed: int | None = None,
) -> tuple[bool, str]:
    """Validate umbrella pages from planner output.

    Rejects excessive page lists that would cause token overflow.

    Args:
        planner_pages: Pages returned by planner.
        total_pages: Total pages in document.
        max_allowed: Maximum allowed pages (default: ExtractionConfig.MAX_UMBRELLA_PAGES).

    Returns:
        Tuple of (is_valid, reason).
    """
    from extractor.core.config import ExtractionConfig

    if max_allowed is None:
        max_allowed = ExtractionConfig.MAX_UMBRELLA_PAGES

    if not planner_pages:
        return True, "empty (will use heuristic)"

    page_count = len(planner_pages)

    # Check if planner returned excessive pages (likely a bug)
    if page_count > max_allowed:
        return False, f"excessive ({page_count} > {max_allowed})"

    # Check if planner returned ALL pages (definitely a bug)
    if page_count >= total_pages * 0.9:
        return False, f"nearly all pages ({page_count}/{total_pages})"

    # Check if pages span most of the document
    if planner_pages:
        span = planner_pages[-1] - planner_pages[0] + 1
        if span > total_pages * 0.8:
            return False, f"span too large ({span} pages, {total_pages} total)"

    return True, "valid"
