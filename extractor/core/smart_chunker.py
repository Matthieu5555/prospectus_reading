"""Smart document chunking based on document structure.

Instead of blindly splitting a document into fixed-size chunks,
this module creates intelligent chunks that:
1. Respect section boundaries (don't split a section in half)
2. Add overlap at boundaries (catch content that spans pages)
3. Group small adjacent sections
4. Keep chunks within a maximum size
"""

from __future__ import annotations

from dataclasses import dataclass

from extractor.core.config import ChunkingConfig
from extractor.core.fund_names import strip_umbrella_prefix
from extractor.pydantic_models import DocumentSkeleton, SectionInfo


@dataclass
class ChunkInfo:
    """A chunk with metadata about what it contains."""

    page_start: int
    page_end: int
    sections_covered: list[str]  # Names of sections in this chunk
    is_appendix: bool = False    # Whether this chunk covers appendix content
    overlap_start: int = 0       # Pages of overlap at start
    overlap_end: int = 0         # Pages of overlap at end


def create_smart_chunks(
    skeleton: DocumentSkeleton,
    max_chunk_size: int = ChunkingConfig.CHUNK_SIZE,
    min_chunk_size: int = ChunkingConfig.MIN_CHUNK_SIZE,
    overlap: int = ChunkingConfig.OVERLAP,
) -> list[tuple[int, int]]:
    """Create chunks that respect document structure.

    Strategy:
    1. Use section boundaries as natural break points
    2. If a section is larger than max_chunk_size, split it with overlap
    3. Group small adjacent sections to avoid tiny chunks
    4. Add overlap at all chunk boundaries

    Args:
        skeleton: Document structure from structure discovery phase.
        max_chunk_size: Maximum pages per chunk.
        min_chunk_size: Minimum pages per chunk (will merge smaller).
        overlap: Pages of overlap between adjacent chunks.

    Returns:
        List of (start_page, end_page) tuples (1-indexed, inclusive).
    """
    if not skeleton.sections:
        # No structure info - fall back to naive chunking
        return _naive_chunks(skeleton.total_pages, max_chunk_size)

    # Build list of section boundaries
    boundaries = _get_section_boundaries(skeleton)

    # Create chunks respecting boundaries
    chunks = _create_boundary_aware_chunks(
        boundaries,
        skeleton.total_pages,
        max_chunk_size,
        min_chunk_size,
    )

    # Add overlap between chunks
    chunks_with_overlap = _add_overlap(chunks, overlap, skeleton.total_pages)

    return chunks_with_overlap


def create_smart_chunks_detailed(
    skeleton: DocumentSkeleton,
    max_chunk_size: int = ChunkingConfig.CHUNK_SIZE,
    min_chunk_size: int = ChunkingConfig.MIN_CHUNK_SIZE,
    overlap: int = ChunkingConfig.OVERLAP,
) -> list[ChunkInfo]:
    """Create chunks with detailed metadata.

    Same as create_smart_chunks but returns ChunkInfo objects
    with additional context about what each chunk contains.
    """
    if not skeleton.sections:
        chunks = _naive_chunks(skeleton.total_pages, max_chunk_size)
        return [
            ChunkInfo(
                page_start=start,
                page_end=end,
                sections_covered=[],
            )
            for start, end in chunks
        ]

    boundaries = _get_section_boundaries(skeleton)
    chunks = _create_boundary_aware_chunks(
        boundaries,
        skeleton.total_pages,
        max_chunk_size,
        min_chunk_size,
    )
    chunks_with_overlap = _add_overlap(chunks, overlap, skeleton.total_pages)

    # Build ChunkInfo objects with metadata
    result = []
    for start, end in chunks_with_overlap:
        sections = _get_sections_in_range(skeleton, start, end)
        is_appendix = any(s.section_type == "appendix" for s in sections)

        result.append(ChunkInfo(
            page_start=start,
            page_end=end,
            sections_covered=[s.name for s in sections],
            is_appendix=is_appendix,
        ))

    return result


def _naive_chunks(total_pages: int, chunk_size: int) -> list[tuple[int, int]]:
    """Fall back to naive fixed-size chunking."""
    chunks = []
    for start in range(1, total_pages + 1, chunk_size):
        end = min(start + chunk_size - 1, total_pages)
        chunks.append((start, end))
    return chunks


def _get_section_boundaries(skeleton: DocumentSkeleton) -> list[int]:
    """Extract all section start pages as potential chunk boundaries.

    Returns sorted list of page numbers where sections begin.
    """
    boundaries = set()

    # Add section start pages
    for section in skeleton.sections:
        boundaries.add(section.page_start)
        # Also add end+1 as a boundary (start of next content)
        if section.page_end < skeleton.total_pages:
            boundaries.add(section.page_end + 1)

    # Add document start and end
    boundaries.add(1)
    boundaries.add(skeleton.total_pages + 1)

    return sorted(boundaries)


def _create_boundary_aware_chunks(
    boundaries: list[int],
    total_pages: int,
    max_size: int,
    min_size: int,
) -> list[tuple[int, int]]:
    """Create chunks that respect section boundaries.

    Strategy:
    - Try to break at boundaries
    - If a section is too large, split it
    - If a section is too small, merge with adjacent
    """
    chunks = []
    current_start = 1

    while current_start <= total_pages:
        # Find the best end point for this chunk
        chunk_end = _find_chunk_end(
            current_start,
            boundaries,
            total_pages,
            max_size,
            min_size,
        )

        chunks.append((current_start, chunk_end))
        current_start = chunk_end + 1

    return chunks


def _find_chunk_end(
    start: int,
    boundaries: list[int],
    total_pages: int,
    max_size: int,
    min_size: int,
) -> int:
    """Find the best end page for a chunk starting at 'start'.

    Prefers to end at a section boundary if possible.
    """
    # Maximum possible end
    max_end = min(start + max_size - 1, total_pages)

    # Find boundaries in the valid range
    valid_boundaries = [
        b - 1 for b in boundaries
        if start + min_size <= b <= max_end + 1
    ]

    if valid_boundaries:
        # Prefer the largest boundary that fits (most content per chunk)
        return max(valid_boundaries)

    # No boundary in range - just use max_end
    return max_end


def _add_overlap(
    chunks: list[tuple[int, int]],
    overlap: int,
    total_pages: int,
) -> list[tuple[int, int]]:
    """Add overlap between adjacent chunks.

    Each chunk extends into the previous and next chunk's territory
    by 'overlap' pages (where possible).
    """
    if not chunks or overlap <= 0:
        return chunks

    result = []
    for i, (start, end) in enumerate(chunks):
        # Extend start backwards (except for first chunk)
        new_start = start
        if i > 0:
            new_start = max(1, start - overlap)

        # Extend end forwards (except for last chunk)
        new_end = end
        if i < len(chunks) - 1:
            new_end = min(total_pages, end + overlap)

        result.append((new_start, new_end))

    return result


def _get_sections_in_range(
    skeleton: DocumentSkeleton,
    start: int,
    end: int,
) -> list[SectionInfo]:
    """Get all sections that overlap with a page range."""
    return [
        s for s in skeleton.sections
        if s.page_start <= end and s.page_end >= start
    ]


def skeleton_from_native_toc(
    toc_entries: list[tuple[int, str, int]],
    total_pages: int,
) -> DocumentSkeleton:
    """Build a DocumentSkeleton from native PDF TOC.

    Args:
        toc_entries: List of (level, title, page_num) from PDFReader.get_toc()
        total_pages: Total pages in document

    Returns:
        DocumentSkeleton with sections derived from TOC
    """
    if not toc_entries:
        return DocumentSkeleton(
            toc_source="none",
            total_pages=total_pages,
        )

    sections = []
    appendix_map = {}

    # First pass: identify the fund list parent section
    # This is typically named "The Sub-Funds", "Fund Descriptions", etc.
    fund_parent_idx = _find_fund_parent_section(toc_entries)
    fund_parent_level = toc_entries[fund_parent_idx][0] if fund_parent_idx is not None else None

    # Convert TOC entries to sections
    # Each entry's page_end is the start of the next entry - 1
    for i, (level, title, page_num) in enumerate(toc_entries):
        # Determine end page (next entry's start - 1, or total_pages)
        if i + 1 < len(toc_entries):
            next_page = toc_entries[i + 1][2]
            page_end = max(page_num, next_page - 1)
        else:
            page_end = total_pages

        # Classify section type
        title_lower = title.lower()
        section_type = "other"
        fund_name = None

        if "appendix" in title_lower or "annex" in title_lower:
            section_type = "appendix"
            appendix_map[title] = (page_num, page_end)
        elif "schedule" in title_lower:
            section_type = "appendix"
            appendix_map[title] = (page_num, page_end)
        elif "fee" in title_lower or "charge" in title_lower:
            section_type = "table"
        elif _is_fund_section(i, level, title, fund_parent_idx, fund_parent_level, toc_entries):
            section_type = "fund_section"
            fund_name = strip_umbrella_prefix(title)

        sections.append(SectionInfo(
            name=title,
            page_start=page_num,
            page_end=page_end,
            level=level,
            section_type=section_type,
            fund_name=fund_name,
        ))

    return DocumentSkeleton(
        toc_source="native",
        toc_pages=[],  # Native TOC doesn't tell us which pages the TOC is on
        sections=sections,
        appendix_map=appendix_map,
        total_pages=total_pages,
    )


def _find_fund_parent_section(toc_entries: list[tuple[int, str, int]]) -> int | None:
    """Find the parent section that contains individual fund listings.

    Returns the index of the parent section, or None if not found.
    """
    fund_parent_patterns = [
        "the sub-funds",
        "sub-funds",
        "fund descriptions",
        "the funds",
        "investment objectives and policies",
    ]

    for i, (level, title, _) in enumerate(toc_entries):
        title_lower = title.lower()
        for pattern in fund_parent_patterns:
            if pattern in title_lower:
                return i

    return None


def _is_fund_section(
    idx: int,
    level: int,
    title: str,
    fund_parent_idx: int | None,
    fund_parent_level: int | None,
    toc_entries: list[tuple[int, str, int]],
) -> bool:
    """Determine if a TOC entry represents an individual fund section.

    A section is a fund section if:
    1. It's a direct child of the fund parent section (level = parent_level + 1)
    2. OR it contains "fund" in the title at level 3+
    3. AND it's not an umbrella-level section
    """
    title_lower = title.lower()

    # Exclude umbrella-level patterns
    umbrella_patterns = [
        "the sub-funds",
        "sub-funds",
        "fund descriptions",
        "the funds",
        "general",
        "investment policies",
        "appendix",
        "annex",
        "schedule",
        "glossary",
        "definitions",
        "management",
        "administration",
    ]
    for pattern in umbrella_patterns:
        if pattern in title_lower:
            return False

    # Check if direct child of fund parent section
    if fund_parent_idx is not None and fund_parent_level is not None:
        if idx > fund_parent_idx and level == fund_parent_level + 1:
            # Check we haven't passed into a different section at parent level
            for j in range(fund_parent_idx + 1, idx):
                check_level = toc_entries[j][0]
                if check_level <= fund_parent_level:
                    # We've exited the fund parent section
                    return False
            return True

    # Fallback: "fund" in title at level 3+
    if "fund" in title_lower and level >= 3:
        return True

    return False


