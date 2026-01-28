"""Critic prompts for extraction verification phase.

Critics verify that extracted data matches the source document,
catching hallucinations and suggesting corrections.
"""

from difflib import SequenceMatcher
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from extractor.core import ParsedTable


def format_table_for_critic(
    table_type: str,
    table: "ParsedTable",
    entity_name: str,
    fuzzy_threshold: float = 0.6,
) -> str:
    """Format a parsed table for critic context, filtering to entity-relevant rows.

    Args:
        table_type: Type of table (e.g., "isin", "fee").
        table: The parsed table with columns and rows.
        entity_name: Name of the entity to match rows against.
        fuzzy_threshold: Minimum similarity ratio for fuzzy matching.

    Returns:
        Formatted string showing relevant table rows, or empty string if no matches.
    """
    if not table or not table.rows:
        return ""

    # Find rows that match the entity name (fuzzy match)
    matching_rows = []
    name_columns = ["Share Class", "Class", "Name", "Fund", "Sub-Fund"]

    for row in table.rows:
        for col in name_columns:
            if col in row:
                cell_value = str(row[col]).strip()
                # Fuzzy match: check if entity_name is similar to cell value
                ratio = SequenceMatcher(None, entity_name.lower(), cell_value.lower()).ratio()
                if ratio >= fuzzy_threshold or entity_name.lower() in cell_value.lower():
                    matching_rows.append(row)
                    break

    if not matching_rows:
        return ""

    # Format the output
    lines = [f"## AUTHORITATIVE {table_type.upper()} TABLE (parsed from PDF)"]
    lines.append(f"Source pages: {table.source_pages[0]}-{table.source_pages[1]}")
    lines.append(f"Columns: {', '.join(table.columns)}")
    lines.append("")
    lines.append("Matching rows for this entity:")

    for row in matching_rows[:5]:  # Limit to 5 most relevant rows
        row_str = " | ".join(f"{k}: {v}" for k, v in row.items() if v)
        lines.append(f"  - {row_str}")

    return "\n".join(lines)


CRITIC_SYSTEM_PROMPT = """You are a verification specialist for fund prospectus extractions.

## Your Task: Correctness Verification

Given extracted data and the source document, verify each field VALUE is correct:
1. Search for supporting text in the source
2. Rate confidence: high/medium/low/not_found
3. Suggest corrections if the extraction is WRONG

## IMPORTANT: Correctness Only

- Your job is to verify VALUES are correct, not to find missing items
- Do NOT suggest adding new share classes or constraints not already extracted
- Completeness (finding missed items) is handled by a separate gleaning step
- You ONLY verify what was extracted is accurate

## Ground Truth Tables

When AUTHORITATIVE TABLE sections are provided, these are **parsed directly from the PDF**
(not LLM-interpreted). They are the ground truth for that data:

- If extracted value differs from table value, the TABLE IS CORRECT
- Use table data to provide corrections
- For ISINs/fees: if the table shows a value but extraction shows NOT_FOUND or different,
  the table value should be the correction
- Tables are pre-filtered to show rows relevant to the entity being verified

## Confidence Levels

- **high**: Exact text match found in source OR matches authoritative table
- **medium**: Similar text found, minor variations (e.g., extracted "1.5%" and source says "1.50%")
- **low**: No direct evidence but value seems plausible
- **not_found**: Value appears hallucinated or from wrong entity, OR contradicts authoritative table

## Handling NOT_FOUND Values

If the extracted value is "NOT_FOUND":
- First check authoritative tables - if value exists there, provide correction from table
- Check if the information actually exists in the source for THIS entity
- If it exists, confidence = "not_found" (extractor missed it) and provide correction
- If it truly doesn't exist, confidence = "high" (correctly reported as missing)
- Note: This is about verifying NOT_FOUND is correct, not about finding more items

## Output Format

Return JSON:
{
  "entity_type": "umbrella" | "subfund" | "shareclass",
  "entity_name": "<name of the entity>",
  "verifications": [
    {
      "field_name": "<field>",
      "extracted_value": "<what was extracted>",
      "source_text": "<exact text from source that supports/contradicts>",
      "confidence": "high" | "medium" | "low" | "not_found",
      "correction": "<correct value if extraction was wrong, else null>",
      "reasoning": "<brief explanation>"
    }
  ],
  "overall_confidence": <0.0 to 1.0>,
  "suggested_reread_pages": [<pages to re-read if confidence low>],
  "critical_errors": ["<errors that must be fixed>"]
}

## Priority Fields

Focus verification effort on HIGH priority fields:
1. Fees (management_fee, entry_fee, exit_fee, ongoing_charges)
2. ISINs (must be exactly 12 characters)
3. Investment restrictions (legally binding)
4. Currency and currency hedging

LOW priority (less strict verification):
- investment_objective, investment_policy (often paraphrased)
- generic descriptions

## Critical Rules

1. Quote EXACT source text that supports or contradicts
2. If authoritative tables are provided, they take precedence over LLM extraction
3. ISIN format: 2 letters + 10 alphanumeric = 12 chars total
4. Fees must include % symbol
5. Fund names must match EXACTLY
"""


def build_critic_prompt(
    entity_type: str,
    entity_name: str,
    extracted_data_json: str,
    source_text: str,
    parsed_tables: dict | None = None,
) -> str:
    """Build the user prompt for the critic.

    Args:
        entity_type: Type of entity (umbrella, subfund, shareclass).
        entity_name: Name of the entity being verified.
        extracted_data_json: JSON serialized extracted data.
        source_text: Source document text.
        parsed_tables: Optional dict of table_type -> ParsedTable for ground truth.

    Returns:
        Formatted user prompt.
    """
    # Build table context if available
    table_context = ""
    if parsed_tables:
        table_sections = []
        for table_type, table in parsed_tables.items():
            formatted = format_table_for_critic(table_type, table, entity_name)
            if formatted:
                table_sections.append(formatted)
        if table_sections:
            table_context = "\n\n" + "\n\n".join(table_sections) + "\n"

    return f"""Verify this {entity_type} extraction against the source document.

## Entity: {entity_name}

## Extracted Data:
{extracted_data_json}
{table_context}
## Source Document:
{source_text}

## Task:
For each field in the extracted data:
1. Check authoritative tables first (if provided) - table values are ground truth
2. Find supporting text in the source
3. Rate confidence
4. Suggest corrections if needed (use table values when available)

Focus on HIGH priority fields (fees, ISINs, restrictions) first."""
