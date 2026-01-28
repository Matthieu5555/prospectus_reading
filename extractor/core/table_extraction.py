"""Table extraction module using PyMuPDF.

Provides structured table parsing with caching and fuzzy lookup.
Tables are parsed once and cached - subsequent lookups are instant.

Usage:
    extractor = TableExtractor()
    table = extractor.parse_table("doc.pdf", start_page=200, end_page=205)
    row = extractor.query(table, lookup_column="Fund Name", lookup_value="Global Bond Fund")
    if row:
        isin = row.get("ISIN")
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path

import fitz  # PyMuPDF
from rapidfuzz import fuzz, process

logger = logging.getLogger(__name__)


@dataclass
class ParsedTable:
    """A parsed table with columns and rows.

    Attributes:
        columns: Column headers in order.
        rows: List of rows, each row is a dict mapping column name to cell value.
        source_pages: (start_page, end_page) that were parsed.
        raw_html: Original HTML representation (for debugging).
    """

    columns: list[str]
    rows: list[dict[str, str]]
    source_pages: tuple[int, int]
    raw_html: str = ""

    def __len__(self) -> int:
        return len(self.rows)

    def __bool__(self) -> bool:
        return len(self.rows) > 0


@dataclass
class TableExtractor:
    """Extracts and queries tables from PDFs using PyMuPDF.

    Tables are cached by (pdf_path, page_range) to avoid re-parsing.
    Queries use fuzzy matching to handle name variations.

    Attributes:
        use_llm: Unused, kept for API compatibility.
        fuzzy_threshold: Minimum score (0-100) for fuzzy matches.
    """

    use_llm: bool = False
    fuzzy_threshold: int = 80
    _cache: dict[tuple[str, int, int], ParsedTable] = field(default_factory=dict, repr=False)

    def parse_table(
        self,
        pdf_path: str | Path,
        start_page: int,
        end_page: int,
    ) -> ParsedTable:
        """Parse tables from specified pages of a PDF.

        Args:
            pdf_path: Path to the PDF file.
            start_page: First page to parse (1-indexed).
            end_page: Last page to parse (1-indexed).

        Returns:
            ParsedTable with extracted data.

        Note:
            Results are cached - calling with same args returns cached result.
        """
        pdf_path = str(Path(pdf_path).resolve())
        cache_key = (pdf_path, start_page, end_page)

        if cache_key in self._cache:
            logger.debug(f"Cache hit for {pdf_path} pages {start_page}-{end_page}")
            return self._cache[cache_key]

        logger.info(f"Parsing tables from {pdf_path} pages {start_page}-{end_page}")

        try:
            raw_tables = self._extract_tables_with_pymupdf(pdf_path, start_page, end_page)
            table = self._combine_tables(raw_tables, start_page, end_page)
        except Exception as e:
            logger.warning(f"Table extraction failed: {e}")
            table = ParsedTable(
                columns=[],
                rows=[],
                source_pages=(start_page, end_page),
                raw_html="",
            )

        self._cache[cache_key] = table
        return table

    def query(
        self,
        table: ParsedTable,
        lookup_column: str,
        lookup_value: str,
    ) -> dict[str, str] | None:
        """Find a row by fuzzy matching on a column.

        Args:
            table: ParsedTable to search.
            lookup_column: Column name to match against.
            lookup_value: Value to search for (fuzzy matched).

        Returns:
            Dict mapping column names to cell values for the best matching row,
            or None if no match above threshold.
        """
        if not table or not table.rows:
            return None

        # Find the column (fuzzy match column name too)
        column_match = self._find_column(table.columns, lookup_column)
        if not column_match:
            logger.debug(f"Column '{lookup_column}' not found in {table.columns}")
            return None

        # Build list of candidates from that column
        candidates = []
        for i, row in enumerate(table.rows):
            cell_value = row.get(column_match, "")
            if cell_value:
                candidates.append((i, cell_value))

        if not candidates:
            return None

        # Fuzzy match to find best row
        candidate_values = [c[1] for c in candidates]
        result = process.extractOne(
            lookup_value,
            candidate_values,
            scorer=fuzz.token_sort_ratio,
        )

        if result is None:
            return None

        matched_value, score, idx = result
        if score < self.fuzzy_threshold:
            logger.debug(f"Best match '{matched_value}' scored {score}, below threshold {self.fuzzy_threshold}")
            return None

        row_idx = candidates[idx][0]
        logger.debug(f"Matched '{lookup_value}' -> '{matched_value}' (score={score}) at row {row_idx}")
        return table.rows[row_idx]

    def query_all(
        self,
        table: ParsedTable,
        lookup_column: str,
        lookup_value: str,
    ) -> list[dict[str, str]]:
        """Find all rows that fuzzy match on a column.

        Useful when multiple share classes match a fund name.

        Args:
            table: ParsedTable to search.
            lookup_column: Column name to match against.
            lookup_value: Value to search for.

        Returns:
            List of matching rows (may be empty).
        """
        if not table or not table.rows:
            return []

        column_match = self._find_column(table.columns, lookup_column)
        if not column_match:
            return []

        results = []
        for row in table.rows:
            cell_value = row.get(column_match, "")
            if cell_value:
                score = fuzz.token_sort_ratio(lookup_value, cell_value)
                if score >= self.fuzzy_threshold:
                    results.append(row)

        return results

    def clear_cache(self) -> None:
        """Clear the table cache."""
        self._cache.clear()

    def _find_column(self, columns: list[str], target: str) -> str | None:
        """Find column name using fuzzy matching."""
        if not columns:
            return None

        # Exact match first
        target_lower = target.lower().strip()
        for col in columns:
            if col.lower().strip() == target_lower:
                return col

        # Fuzzy match
        result = process.extractOne(
            target,
            columns,
            scorer=fuzz.token_sort_ratio,
        )
        if result and result[1] >= self.fuzzy_threshold:
            return result[0]

        return None

    def _extract_tables_with_pymupdf(
        self,
        pdf_path: str,
        start_page: int,
        end_page: int,
    ) -> list[dict]:
        """Use PyMuPDF to extract tables from PDF pages.

        Args:
            pdf_path: Path to PDF.
            start_page: First page (1-indexed).
            end_page: Last page (1-indexed).

        Returns:
            List of raw table dicts with 'page', 'columns', 'rows'.
        """
        raw_tables = []

        doc = fitz.open(pdf_path)
        try:
            # PyMuPDF uses 0-indexed pages
            for page_num in range(start_page - 1, end_page):
                if page_num >= len(doc):
                    break

                page = doc[page_num]
                tables = page.find_tables()

                for table in tables:
                    # Extract table data
                    extracted = table.extract()
                    if not extracted or len(extracted) < 2:
                        continue

                    # First row is typically headers
                    headers = [self._clean_cell(c) for c in extracted[0] if c]
                    if not headers:
                        continue

                    # Remaining rows are data
                    rows = []
                    for row_data in extracted[1:]:
                        if len(row_data) >= len(headers):
                            row_dict = {}
                            for i, col in enumerate(headers):
                                cell_value = row_data[i] if i < len(row_data) else ""
                                row_dict[col] = self._clean_cell(cell_value)
                            if any(row_dict.values()):
                                rows.append(row_dict)

                    if rows:
                        raw_tables.append({
                            "page": page_num + 1,  # Back to 1-indexed
                            "columns": headers,
                            "rows": rows,
                            "bbox": table.bbox,
                        })

        finally:
            doc.close()

        return raw_tables

    def _combine_tables(
        self,
        raw_tables: list[dict],
        start_page: int,
        end_page: int,
    ) -> ParsedTable:
        """Combine multiple raw tables into a single ParsedTable.

        Tables with matching columns are merged. If columns differ,
        only the first table's structure is used.

        Args:
            raw_tables: List of raw table dicts from _extract_tables_with_pymupdf.
            start_page: Source start page.
            end_page: Source end page.

        Returns:
            Combined ParsedTable.
        """
        if not raw_tables:
            return ParsedTable(
                columns=[],
                rows=[],
                source_pages=(start_page, end_page),
                raw_html="",
            )

        # Use first table's columns as canonical
        all_columns = raw_tables[0]["columns"]
        all_rows = []

        for table in raw_tables:
            # Check if columns match (allows merging multi-page tables)
            if table["columns"] == all_columns:
                all_rows.extend(table["rows"])
            else:
                # Different structure - try to map columns
                col_mapping = {}
                for col in table["columns"]:
                    if col in all_columns:
                        col_mapping[col] = col
                    else:
                        # Try fuzzy match
                        match = process.extractOne(col, all_columns, scorer=fuzz.ratio)
                        if match and match[1] >= 80:
                            col_mapping[col] = match[0]

                # Add rows with mapped columns
                for row in table["rows"]:
                    mapped_row = {}
                    for orig_col, value in row.items():
                        if orig_col in col_mapping:
                            mapped_row[col_mapping[orig_col]] = value
                    if mapped_row:
                        all_rows.append(mapped_row)

        return ParsedTable(
            columns=all_columns,
            rows=all_rows,
            source_pages=(start_page, end_page),
            raw_html="",
        )

    def _clean_cell(self, cell: str | None) -> str:
        """Clean cell content - normalize whitespace."""
        if cell is None:
            return ""
        # Normalize whitespace
        return " ".join(str(cell).split()).strip()
