"""External Reference Pre-Scan phase - detects external document references.

This lightweight phase runs BEFORE exploration to detect patterns like:
- "ISIN See applicable KIID"
- "Refer to the Annual Report"
- "Performance: see KIID"

By detecting these early, we prevent wasted LLM calls searching for data
that doesn't exist in the document.

No LLM calls - pure regex scanning for speed.
"""

import re
from dataclasses import dataclass, field

from extractor.phases.phase_base import PhaseRunner


@dataclass
class ExternalRefResult:
    """Result from external reference pre-scan."""

    total_refs_found: int
    fields_external: dict[str, str]  # field_name -> external_doc
    pages_scanned: int


# Patterns for detecting external references
# Each pattern: (regex, field_hint, external_doc)
EXTERNAL_REF_PATTERNS: list[tuple[str, str | None, str]] = [
    # ISIN patterns
    (r"ISIN[:\s]+[Ss]ee\s+(?:applicable\s+)?(?:the\s+)?KIID", "isin", "KIID"),
    (r"ISIN[:\s]+[Ss]ee\s+(?:applicable\s+)?(?:the\s+)?Key\s+Investor", "isin", "KIID"),
    (r"ISIN\s+codes?\s+(?:are\s+)?(?:available\s+)?in\s+(?:the\s+)?KIID", "isin", "KIID"),
    (r"(?:for\s+)?ISIN[s,\s]+(?:please\s+)?refer\s+to\s+(?:the\s+)?KIID", "isin", "KIID"),
    (r"ISIN[:\s]+(?:not\s+)?(?:available|disclosed)", "isin", "Not Disclosed"),

    # Performance patterns
    (r"[Pp]erformance[:\s]+[Ss]ee\s+(?:the\s+)?(?:applicable\s+)?KIID", "performance", "KIID"),
    (r"[Pp]erformance\s+(?:data\s+)?(?:is\s+)?(?:available\s+)?in\s+(?:the\s+)?Annual\s+Report", "performance", "Annual Report"),
    (r"[Pp]ast\s+performance[:\s]+[Ss]ee\s+KIID", "performance", "KIID"),

    # Risk patterns
    (r"[Rr]isk\s+(?:profile|indicator)[:\s]+[Ss]ee\s+(?:the\s+)?KIID", "risk_profile", "KIID"),
    (r"SRRI[:\s]+[Ss]ee\s+(?:the\s+)?KIID", "risk_profile", "KIID"),

    # Fee patterns
    (r"[Ff]ees?[:\s]+[Ss]ee\s+(?:the\s+)?KIID", "fee", "KIID"),
    (r"[Oo]ngoing\s+[Cc]harges?[:\s]+[Ss]ee\s+(?:the\s+)?KIID", "ongoing_charges", "KIID"),

    # General external doc references
    (r"[Rr]efer\s+to\s+(?:the\s+)?Annual\s+Report", None, "Annual Report"),
    (r"[Ss]ee\s+(?:the\s+)?(?:relevant\s+)?[Ss]upplement", None, "Supplement"),
    (r"[Ss]ee\s+(?:the\s+)?(?:applicable\s+)?KIID", None, "KIID"),
    (r"[Ss]ee\s+(?:the\s+)?Key\s+Investor\s+Information\s+Document", None, "KIID"),
    (r"[Aa]vailable\s+(?:on|at)\s+(?:the\s+)?(?:Manager's\s+)?[Ww]ebsite", None, "Website"),
    (r"[Aa]vailable\s+at\s+(?:www\.|https?://)", None, "Website"),
]


class ExternalRefScanPhase(PhaseRunner[ExternalRefResult]):
    """Pre-scan phase for external document references.

    This runs before exploration to detect external references early.
    Results are written to the knowledge graph so downstream phases
    know not to waste effort searching for externally-documented fields.
    """

    name = "ExternalRefScan"

    async def run(self) -> ExternalRefResult:
        """Scan document for external reference patterns.

        Returns:
            ExternalRefResult with detected external references.
        """
        total_pages = self.context.pdf.page_count
        self.log(f"Scanning {total_pages} pages for external references")

        # Track findings
        refs_found: list[tuple[str | None, str, int, str]] = []  # (field, doc, page, quote)
        fields_external: dict[str, str] = {}

        # Compile patterns
        compiled_patterns = [
            (re.compile(pattern, re.IGNORECASE), field_hint, external_doc)
            for pattern, field_hint, external_doc in EXTERNAL_REF_PATTERNS
        ]

        # Scan each page
        for page_num in range(1, total_pages + 1):
            try:
                page_text = self.context.pdf.read_pages(page_num, page_num)

                for regex, field_hint, external_doc in compiled_patterns:
                    for match in regex.finditer(page_text):
                        quote = match.group(0)
                        refs_found.append((field_hint, external_doc, page_num, quote))

                        if field_hint and field_hint not in fields_external:
                            fields_external[field_hint] = external_doc
                            self.log(f"Found: {field_hint} -> {external_doc} (page {page_num})")

            except Exception as e:
                self.log(f"Error scanning page {page_num}: {e}", "warning")

        # Record to knowledge graph
        knowledge = self.context.knowledge
        for field_hint, external_doc, page_num, quote in refs_found:
            if field_hint:
                # Check if already recorded (avoid duplicates)
                existing = knowledge.get_external_ref_for_field(field_hint)
                if not existing:
                    knowledge.record_external_reference(
                        field_name=field_hint,
                        external_doc=external_doc,
                        source_page=page_num,
                        source_quote=quote[:200],
                        source_agent="external_ref_scan",
                    )

        # Log summary
        if fields_external:
            self.log(f"Detected {len(fields_external)} external field(s): {list(fields_external.keys())}")
        else:
            self.log("No external references detected")

        self.logger.phase_result(
            self.name,
            f"{len(refs_found)} refs, {len(fields_external)} fields external",
        )

        return ExternalRefResult(
            total_refs_found=len(refs_found),
            fields_external=fields_external,
            pages_scanned=total_pages,
        )
