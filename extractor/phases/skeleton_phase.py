"""Skeleton phase - document structure detection.

Phase 0: Runs BEFORE exploration to build a DocumentSkeleton
that guides structure-aware chunking.

Strategy:
1. Check native TOC via pdf.get_toc()
2. If empty, use LLM to extract TOC from first pages
3. Output DocumentSkeleton with section boundaries
"""

from dataclasses import dataclass

from extractor.phases.phase_base import PhaseRunner
from extractor.pydantic_models import DocumentSkeleton, SectionInfo
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

        # Strategy 2: LLM-based TOC extraction (fallback)
        self.log("No native TOC, attempting LLM extraction")
        skeleton = await self._detect_toc_with_llm(total_pages)

        if skeleton and skeleton.sections:
            self.log(f"LLM extraction found {len(skeleton.sections)} sections")
            self._store_skeleton(skeleton)
            self.logger.phase_result(
                "Skeleton",
                f"{len(skeleton.sections)} sections from LLM extraction",
            )
            return SkeletonResult(
                skeleton=skeleton,
                toc_source="llm",
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

    async def _detect_toc_with_llm(self, total_pages: int) -> DocumentSkeleton | None:
        """Use LLM to extract TOC from first pages when native TOC is missing.

        Reads the first 10-15 pages (where TOC typically lives) and uses LLM
        to identify document structure, section headings, and page numbers.

        Args:
            total_pages: Total pages in the document.

        Returns:
            DocumentSkeleton if structure was detected, None otherwise.
        """
        from extractor.core.llm_client import LLMClient
        from extractor.pydantic_models.extraction_models import LLMExtractedTOC

        # Read first 10-15 pages (usually where TOC lives)
        pages_to_read = min(15, total_pages)
        pages_text = self.context.pdf.read_pages(1, pages_to_read)

        if not pages_text or len(pages_text.strip()) < 200:
            self.log("Insufficient text for LLM TOC extraction", "warning")
            return None

        client = LLMClient(cost_tracker=self.context.cost_tracker)

        system_prompt = """You are analyzing a financial document (likely a fund prospectus).
Your task is to extract the table of contents or document structure.

Look for:
1. A literal "Table of Contents" or "Contents" section listing sections with page numbers
2. Major section headings that appear to be chapter/part divisions
3. Common prospectus sections: "Investment Objectives", "Risk Factors", "Fees and Expenses", etc.

Return structured JSON with:
- sections: List of section titles, page numbers (1-indexed), and hierarchy levels
- document_title: The main document title if visible
- confidence: 0.0-1.0 based on how clearly you could identify structure
- notes: Any issues or ambiguities

If you find a clear TOC listing, confidence should be high (0.8+).
If inferring structure from headings, confidence should be medium (0.5-0.7).
If structure is unclear, return empty sections list with low confidence."""

        user_prompt = f"""Extract the table of contents from these document pages (pages 1-{pages_to_read}):

---
{pages_text[:25000]}
---

Identify all major sections with their page numbers and hierarchy levels.
Level 1 = top-level sections, Level 2 = subsections, etc."""

        try:
            result = await client.complete_structured(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model=self.context.config.smart_model,
                response_model=LLMExtractedTOC,
                agent="skeleton_llm",
            )

            if not result.sections:
                self.log(f"LLM found no TOC sections (confidence: {result.confidence:.0%})", "warning")
                if result.notes:
                    self.log(f"LLM notes: {result.notes}", "debug")
                return None

            self.log(f"LLM extracted {len(result.sections)} TOC sections (confidence: {result.confidence:.0%})")

            # Convert to DocumentSkeleton format
            return self._llm_toc_to_skeleton(result, total_pages)

        except Exception as e:
            self.log(f"LLM TOC extraction failed: {e}", "warning")
            return None

    def _llm_toc_to_skeleton(self, toc: "LLMExtractedTOC", total_pages: int) -> DocumentSkeleton:
        """Convert LLM-extracted TOC to DocumentSkeleton format.

        Args:
            toc: LLM extraction result.
            total_pages: Total document pages (for calculating end pages).

        Returns:
            DocumentSkeleton with section boundaries.
        """
        sections = []

        for i, s in enumerate(toc.sections):
            # Calculate end page: next section's start - 1, or document end
            if i + 1 < len(toc.sections):
                end_page = max(s.page, toc.sections[i + 1].page - 1)
            else:
                end_page = total_pages

            sections.append(SectionInfo(
                title=s.title,
                level=s.level,
                page_start=s.page,
                page_end=end_page,
            ))

        return DocumentSkeleton(
            sections=sections,
            total_pages=total_pages,
            toc_source="llm",
        )

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
