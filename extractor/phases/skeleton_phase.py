"""Skeleton phase - document structure detection.

Phase 0: Runs BEFORE exploration to build a DocumentSkeleton
that guides structure-aware chunking.

Strategy:
1. Check native TOC via pdf.get_toc()
2. If empty, use layout detection on first pages to find TOC/headers
3. Output DocumentSkeleton with section boundaries
"""

from dataclasses import dataclass

from extractor.phases.phase_base import PhaseRunner
from extractor.pydantic_models import DocumentSkeleton
from extractor.core import skeleton_from_native_toc


@dataclass
class SkeletonResult:
    """Result from the skeleton phase."""

    skeleton: DocumentSkeleton | None
    toc_source: str  # "native", "layout", "none"
    section_count: int


class SkeletonPhase(PhaseRunner[SkeletonResult]):
    """Phase 0: Document structure detection.

    Builds a DocumentSkeleton from:
    1. Native PDF TOC (fastest, most reliable)
    2. Layout detection (fallback for scanned/non-TOC docs)

    The skeleton guides structure-aware chunking in exploration.
    """

    name = "Skeleton"

    async def run(self) -> SkeletonResult:
        """Build document skeleton from TOC or layout detection.

        Returns:
            SkeletonResult with skeleton and source info.
        """
        total_pages = self.context.pdf.page_count
        self.start(1, model="")  # No LLM needed

        # Strategy 1: Native TOC (preferred)
        native_toc = self.context.pdf.get_toc()
        if native_toc:
            self.log(f"Found native TOC with {len(native_toc)} entries")
            skeleton = skeleton_from_native_toc(native_toc, total_pages)

            if skeleton and skeleton.sections:
                self.log(f"Built skeleton with {len(skeleton.sections)} sections")
                self._store_skeleton(skeleton)
                self.logger.phase_result(
                    "Skeleton",
                    f"{len(skeleton.sections)} sections from native TOC",
                )
                return SkeletonResult(
                    skeleton=skeleton,
                    toc_source="native",
                    section_count=len(skeleton.sections),
                )

        # Strategy 2: Layout detection (fallback)
        self.log("No native TOC, attempting layout detection")
        skeleton = await self._detect_layout_skeleton(total_pages)

        if skeleton and skeleton.sections:
            self.log(f"Layout detection found {len(skeleton.sections)} sections")
            self._store_skeleton(skeleton)
            self.logger.phase_result(
                "Skeleton",
                f"{len(skeleton.sections)} sections from layout detection",
            )
            return SkeletonResult(
                skeleton=skeleton,
                toc_source="layout",
                section_count=len(skeleton.sections),
            )

        # No skeleton available
        self.log("No document structure detected, using naive chunking")
        self.logger.phase_result("Skeleton", "none (using naive chunking)")
        return SkeletonResult(
            skeleton=None,
            toc_source="none",
            section_count=0,
        )

    async def _detect_layout_skeleton(self, total_pages: int) -> DocumentSkeleton | None:
        """Use layout detection to find document structure.

        Stub: Not yet implemented. Would render first N pages and use
        Surya LayoutPredictor to find TableOfContents and SectionHeader elements.

        Native TOC handles most prospectuses, so this is low priority.
        """
        # Not implemented - native TOC works for most documents
        self.log("Layout detection not yet implemented, skipping")
        return None

    def _store_skeleton(self, skeleton: DocumentSkeleton) -> None:
        """Store skeleton in context for downstream phases."""
        # Store full skeleton in state for Planning/Extraction phases
        self.context.skeleton = skeleton

        # Also store appendix_map in knowledge graph for cross-reference resolution
        if skeleton.appendix_map:
            self.context.knowledge.appendix_map = skeleton.appendix_map


def structure_aware_chunks(
    skeleton: DocumentSkeleton | None,
    total_pages: int,
    target_chunk_size: int = 30,
    max_chunk_size: int = 50,
) -> list[tuple[int, int, str]]:
    """Create chunks that respect document structure.

    When skeleton is available, chunks align with section boundaries.
    Large sections are split but marked as partial.
    Small sections are combined.

    Args:
        skeleton: Document skeleton with section info (or None).
        total_pages: Total pages in document.
        target_chunk_size: Ideal pages per chunk.
        max_chunk_size: Maximum pages per chunk.

    Returns:
        List of (start_page, end_page, chunk_type) tuples.
        chunk_type is "complete", "partial_section", or "naive".
    """
    if not skeleton or not skeleton.sections:
        # Fall back to naive chunking
        return _naive_chunks(total_pages, target_chunk_size)

    chunks = []
    current_chunk_pages: list[int] = []
    current_size = 0

    for section in skeleton.sections:
        section_start = section.page_start
        section_end = section.page_end or section_start
        section_size = section_end - section_start + 1

        if section_size > max_chunk_size:
            # Large section: split it, mark as partial
            if current_chunk_pages:
                # Finalize current chunk first
                chunks.append((
                    current_chunk_pages[0],
                    current_chunk_pages[-1],
                    "complete"
                ))
                current_chunk_pages = []
                current_size = 0

            # Split large section
            for start in range(section_start, section_end + 1, target_chunk_size):
                end = min(start + target_chunk_size - 1, section_end)
                chunks.append((start, end, "partial_section"))

        elif current_size + section_size > max_chunk_size:
            # Adding section exceeds max: finalize current, start new
            if current_chunk_pages:
                chunks.append((
                    current_chunk_pages[0],
                    current_chunk_pages[-1],
                    "complete"
                ))

            current_chunk_pages = list(range(section_start, section_end + 1))
            current_size = section_size

        else:
            # Add section to current chunk
            current_chunk_pages.extend(range(section_start, section_end + 1))
            current_size += section_size

    # Don't forget last chunk
    if current_chunk_pages:
        chunks.append((
            current_chunk_pages[0],
            current_chunk_pages[-1],
            "complete"
        ))

    # Handle pages not covered by sections (gaps)
    chunks = _fill_gaps(chunks, total_pages, target_chunk_size)

    return sorted(chunks, key=lambda x: x[0])


def _naive_chunks(
    total_pages: int,
    chunk_size: int,
) -> list[tuple[int, int, str]]:
    """Create fixed-size chunks (fallback when no skeleton)."""
    chunks = []
    for start in range(1, total_pages + 1, chunk_size):
        end = min(start + chunk_size - 1, total_pages)
        chunks.append((start, end, "naive"))
    return chunks


def _fill_gaps(
    chunks: list[tuple[int, int, str]],
    total_pages: int,
    chunk_size: int,
) -> list[tuple[int, int, str]]:
    """Fill gaps between chunks with naive chunks."""
    if not chunks:
        return _naive_chunks(total_pages, chunk_size)

    # Sort by start page
    chunks = sorted(chunks, key=lambda x: x[0])
    result = []
    covered = set()

    for start, end, chunk_type in chunks:
        result.append((start, end, chunk_type))
        covered.update(range(start, end + 1))

    # Find uncovered pages
    all_pages = set(range(1, total_pages + 1))
    uncovered = sorted(all_pages - covered)

    if uncovered:
        # Group consecutive uncovered pages
        gap_start = uncovered[0]
        gap_end = uncovered[0]

        for page in uncovered[1:]:
            if page == gap_end + 1:
                gap_end = page
            else:
                # End of gap, create chunk
                result.append((gap_start, gap_end, "gap"))
                gap_start = page
                gap_end = page

        # Last gap
        result.append((gap_start, gap_end, "gap"))

    return result
