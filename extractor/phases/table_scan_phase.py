"""Table scan phase - upfront scanning of all tables in the document.

Scans all pages using PyMuPDF's table detection to build a complete
inventory of tables before any extraction begins.

IMPORTANT: This phase only extracts RAW table structure. It does NOT:
- Classify tables into types (isin, fee, etc.)
- Infer what columns mean
- Map columns to schema fields

Classification and interpretation is the job of exploration/extraction phases.
"""

from dataclasses import dataclass, field

import fitz  # PyMuPDF

from extractor.phases.phase_base import PhaseRunner


@dataclass
class ScannedTable:
    """A table discovered during the table scan phase.

    Contains RAW structural data only. No interpretation or classification.

    Attributes:
        page: Page number where the table is (1-indexed).
        bbox: Bounding box (x0, y0, x1, y1) on the page.
        columns: Column headers as found (raw strings).
        rows: Data rows as list of dicts mapping column -> value.
        row_count: Number of data rows.
        col_count: Number of columns.
    """

    page: int
    bbox: tuple[float, float, float, float]
    columns: list[str]
    rows: list[dict[str, str]]
    row_count: int
    col_count: int


@dataclass
class TableScanResult:
    """Result from the table scan phase.

    Attributes:
        tables: All tables found in the document (raw data).
        pages_with_tables: Set of page numbers containing tables.
        pages_without_tables: Set of page numbers with no tables.
    """

    tables: list[ScannedTable]
    pages_with_tables: set[int]
    pages_without_tables: set[int]


class TableScanPhase(PhaseRunner[TableScanResult]):
    """Phase 0.5: Scan all pages for tables.

    Runs after skeleton phase but before exploration.
    Uses PyMuPDF's table detection to find all tables.

    Outputs RAW table data only - no classification or interpretation.
    """

    name = "TableScan"

    async def run(self) -> TableScanResult:
        """Scan all pages for tables.

        Returns:
            TableScanResult with all discovered tables (raw data).
        """
        pdf_path = str(self.context.pdf.path)
        total_pages = self.context.pdf.page_count

        self.start(total_pages, model="")

        tables: list[ScannedTable] = []
        pages_with_tables: set[int] = set()
        pages_without_tables: set[int] = set()

        doc = fitz.open(pdf_path)
        try:
            for page_num in range(total_pages):
                page = doc[page_num]
                page_tables = page.find_tables()

                page_1_indexed = page_num + 1

                if page_tables:
                    pages_with_tables.add(page_1_indexed)
                    for table in page_tables:
                        scanned = self._extract_scanned_table(table, page_1_indexed)
                        if scanned:
                            tables.append(scanned)
                else:
                    pages_without_tables.add(page_1_indexed)

                if page_num % 50 == 0:
                    self.progress(page_num + 1, total_pages, f"page {page_1_indexed}")

        finally:
            doc.close()

        # Store raw data in context state
        self.context.state.scanned_tables = tables
        self.context.state.pages_with_tables = pages_with_tables

        self.log(f"Found {len(tables)} tables on {len(pages_with_tables)} pages")
        self.end(f"{len(tables)} tables on {len(pages_with_tables)} pages")

        return TableScanResult(
            tables=tables,
            pages_with_tables=pages_with_tables,
            pages_without_tables=pages_without_tables,
        )

    def _extract_scanned_table(
        self,
        table,
        page_num: int,
    ) -> ScannedTable | None:
        """Extract a ScannedTable from a PyMuPDF table object.

        Args:
            table: PyMuPDF table object from find_tables().
            page_num: Page number (1-indexed).

        Returns:
            ScannedTable with raw data, or None if table is invalid.
        """
        extracted = table.extract()
        if not extracted or len(extracted) < 2:
            return None

        # First row is headers - store as-is
        headers = [self._clean_cell(c) for c in extracted[0] if c]
        if not headers:
            return None

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

        if not rows:
            return None

        return ScannedTable(
            page=page_num,
            bbox=table.bbox,
            columns=headers,
            rows=rows,
            row_count=len(rows),
            col_count=len(headers),
        )

    def _clean_cell(self, cell: str | None) -> str:
        """Normalize cell content."""
        if cell is None:
            return ""
        return " ".join(str(cell).split()).strip()
