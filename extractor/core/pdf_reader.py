"""PDF reader for extraction pipeline.

Pure Python + PyMuPDF.
"""

import fitz  # PyMuPDF
from pathlib import Path

from extractor.core.config import ChunkingConfig, SearchLimits, SearchPatterns


class PDFReader:
    """PDF document reader with page-level access and search capabilities.

    Includes pattern-based search functionality (formerly in SearchContext).
    """

    # Common search patterns by category - delegated to centralized config
    PATTERNS = {
        "isin": SearchPatterns.ISIN,
        "fee": SearchPatterns.FEE,
        "restriction": SearchPatterns.RESTRICTION,
        "leverage": SearchPatterns.LEVERAGE,
        "derivative": SearchPatterns.DERIVATIVE,
    }

    def __init__(self, path: str | Path):
        """Load a PDF document.

        Args:
            path: Path to the PDF file.
        """
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"PDF not found: {self.path}")

        self._doc = fitz.open(str(self.path))
        self._learned_patterns: dict[str, list[str]] = {}  # category -> found patterns

    @property
    def page_count(self) -> int:
        """Total number of pages in the document."""
        return len(self._doc)

    @property
    def filename(self) -> str:
        """Filename without path."""
        return self.path.name

    def read_page(self, page_num: int) -> str:
        """Read text from a single page (1-indexed).

        Args:
            page_num: Page number (1-indexed).

        Returns:
            Page text with header.
        """
        if page_num < 1 or page_num > self.page_count:
            return f"Error: Page {page_num} out of range (1-{self.page_count})"

        page = self._doc[page_num - 1]  # Convert to 0-indexed
        text = page.get_text()
        return f"=== PAGE {page_num} ===\n{text}"

    def read_pages(self, start: int, end: int) -> str:
        """Read text from a range of pages (1-indexed, inclusive).

        Args:
            start: First page (1-indexed).
            end: Last page (1-indexed, inclusive).

        Returns:
            Concatenated page text with headers.
        """
        # Clamp to valid range
        start = max(1, start)
        end = min(self.page_count, end)

        if start > end:
            return f"Error: Invalid range {start}-{end}"

        chunks = []
        for page_num in range(start, end + 1):
            chunks.append(self.read_page(page_num))

        return "\n\n".join(chunks)

    def search(self, term: str, max_results: int = 50) -> list[dict]:
        """Search for a term across all pages.

        Args:
            term: Search term (case-insensitive).
            max_results: Maximum number of results to return.

        Returns:
            List of {page: int, context: str} dicts.
        """
        results = []
        term_lower = term.lower()

        for i, page in enumerate(self._doc):
            if len(results) >= max_results:
                break

            text = page.get_text()
            if term_lower in text.lower():
                # Extract context lines
                lines = text.split("\n")
                matching_lines = [
                    line.strip() for line in lines
                    if term_lower in line.lower() and line.strip()
                ][:3]

                context = "; ".join(matching_lines)
                if len(context) > 200:
                    context = context[:200] + "..."

                results.append({
                    "page": i + 1,  # 1-indexed
                    "context": context,
                })

        return results

    # Backwards-compatible alias
    search_term = search

    def search_patterns(
        self,
        category: str,
        max_results: int = SearchLimits.DEFAULT,
        additional_terms: list[str] | None = None,
    ) -> list[dict]:
        """Search for patterns by category.

        Args:
            category: One of 'isin', 'fee', 'restriction', 'leverage', 'derivative'.
            max_results: Maximum results to return.
            additional_terms: Extra terms to search for.

        Returns:
            List of {page, context} dicts.
        """
        # Get base patterns for this category
        patterns = list(self.PATTERNS.get(category, []))

        # Add learned patterns (e.g., ISINs we've already found)
        if category in self._learned_patterns:
            learned = self._learned_patterns[category]
            # Prioritize learned patterns
            patterns = learned + [p for p in patterns if p not in learned]

        # Add any additional terms
        if additional_terms:
            patterns.extend(additional_terms)

        if not patterns:
            return []

        results = []
        seen_pages = set()
        per_pattern = max(1, max_results // len(patterns))

        for pattern in patterns:
            for hit in self.search(pattern, per_pattern):
                if hit["page"] not in seen_pages:
                    results.append(hit)
                    seen_pages.add(hit["page"])
                if len(results) >= max_results:
                    break
            if len(results) >= max_results:
                break

        return results

    def record_pattern(self, category: str, pattern: str):
        """Record a found pattern for learning.

        For example, if we find ISIN "LU0123456789", record "LU0" as a
        learned pattern for the 'isin' category.

        Args:
            category: Pattern category.
            pattern: The pattern to record (will extract prefix).
        """
        if not pattern or pattern == "NOT_FOUND":
            return

        # Extract prefix based on category
        if category == "isin" and len(pattern) >= 3:
            prefix = pattern[:3]
        else:
            prefix = pattern

        if category not in self._learned_patterns:
            self._learned_patterns[category] = []

        if prefix not in self._learned_patterns[category]:
            self._learned_patterns[category].append(prefix)

    def get_toc(self) -> list[tuple[int, str, int]]:
        """Extract native TOC from PDF metadata.

        PyMuPDF returns TOC as list of [level, title, page_num] where:
        - level: nesting depth (1 = top level, 2 = subsection, etc.)
        - title: section title string
        - page_num: 1-indexed page number

        Returns:
            List of (level, title, page_num) tuples.
            Empty list if no TOC embedded in PDF.
        """
        if not self._doc:
            return []
        return self._doc.get_toc()

    def get_page_chunks(self, chunk_size: int = ChunkingConfig.CHUNK_SIZE) -> list[tuple[int, int]]:
        """Generate page range chunks for parallel processing.

        Args:
            chunk_size: Number of pages per chunk.

        Returns:
            List of (start, end) tuples (1-indexed, inclusive).
        """
        chunks = []
        for start in range(1, self.page_count + 1, chunk_size):
            end = min(start + chunk_size - 1, self.page_count)
            chunks.append((start, end))
        return chunks

    def close(self):
        """Close the document."""
        if self._doc:
            self._doc.close()
            self._doc = None

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc_val, _exc_tb):
        self.close()
        return False
