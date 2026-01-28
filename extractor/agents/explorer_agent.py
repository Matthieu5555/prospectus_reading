"""Explorer agent for Phase 1: Document exploration.

Explorers traverse non-overlapping page ranges in parallel, creating structured
notes about what they find. Each explorer has fresh context - no accumulation.

Supports gleaning: multiple passes where subsequent passes focus on gaps
identified from prior passes (especially external references).
"""

from extractor.core.config import DEFAULT_MODELS
from extractor.core.cost_tracker import CostTracker
from extractor.core.llm_client import LLMClient
from extractor.pydantic_models.pipeline import ExplorationNotes
from extractor.prompts.explorer_prompt import EXPLORER_SYSTEM_PROMPT, build_explorer_prompt


# Gleaning prompt for second pass - focuses on external references
GLEANING_EXPLORER_PROMPT = """You are re-examining document pages to find EXTERNAL REFERENCES that may have been missed.

PREVIOUS FINDINGS for pages {page_start}-{page_end}:
{previous_summary}

CRITICAL: Many prospectuses point to external documents for key information:
- "ISIN See applicable KIID" → ISINs are NOT in this document
- "Performance: see KIID" → Performance data is external
- "Refer to Annual Report" → Data is in Annual Report
- "See Supplement" → Data is in a separate Supplement document

YOUR TASK:
1. Search the text for ANY reference to external documents (KIID, Key Investor Information, Annual Report, Supplement, Website)
2. Note which FIELDS are referenced externally (ISIN, fees, performance, risk profile, etc.)
3. Find any tables or fund sections that were missed in the first pass

Return JSON with ONLY NEW findings (don't repeat what was already found):
{{
  "additional_cross_references": [
    {{
      "text": "ISIN See applicable KIID",
      "source_page": <page number>,
      "is_external": true,
      "external_doc": "KIID",
      "field_hint": "isin"
    }}
  ],
  "additional_tables": [
    {{
      "table_type": "isin|fee|share_class|other",
      "page_start": <page>,
      "page_end": <page>,
      "columns": ["col1", "col2"],
      "has_fund_name_column": true/false
    }}
  ],
  "additional_observations": ["<anything else important that was missed>"]
}}

If you find nothing new, return empty arrays."""


async def run_explorer(
    pages_text: str,
    page_start: int,
    page_end: int,
    model: str = DEFAULT_MODELS["exploration"],
    cost_tracker: CostTracker | None = None,
    skeleton_context: str = "",
) -> ExplorationNotes:
    """Run an explorer on a page range.

    Args:
        pages_text: Pre-loaded text from the pages.
        page_start: First page number (1-indexed).
        page_end: Last page number (1-indexed).
        model: LLM model to use.
        cost_tracker: Optional tracker to record token usage.
        skeleton_context: Optional document structure context for cross-ref resolution.

    Returns:
        ExplorationNotes with structured findings.
    """
    client = LLMClient(cost_tracker=cost_tracker)

    try:
        notes = await client.complete_structured(
            system_prompt=EXPLORER_SYSTEM_PROMPT,
            user_prompt=build_explorer_prompt(pages_text, page_start, page_end, skeleton_context),
            model=model,
            response_model=ExplorationNotes,
            agent="explorer",
        )

        # Ensure page range is set correctly
        notes.page_start = page_start
        notes.page_end = page_end
        return notes

    except Exception as e:
        # Return minimal valid notes on any error
        return ExplorationNotes(
            page_start=page_start,
            page_end=page_end,
            observations=[f"Explorer error: {e}"],
        )


async def run_explorer_with_critic(
    pages_text: str,
    page_start: int,
    page_end: int,
    model: str = DEFAULT_MODELS["exploration"],
    critic_model: str = DEFAULT_MODELS["critic"],
    cost_tracker: CostTracker | None = None,
    skeleton_context: str = "",
) -> ExplorationNotes:
    """Run an explorer with critic verification.

    The critic checks that fund names are exact matches from the document.

    Args:
        pages_text: Pre-loaded text from the pages.
        page_start: First page number (1-indexed).
        page_end: Last page number (1-indexed).
        model: LLM model for explorer.
        critic_model: LLM model for critic.
        cost_tracker: Optional tracker to record token usage.
        skeleton_context: Optional document structure context for cross-ref resolution.

    Returns:
        Verified ExplorationNotes.
    """
    # First pass: explore
    notes = await run_explorer(pages_text, page_start, page_end, model, cost_tracker, skeleton_context)

    # Skip critic if no funds found
    if not notes.funds_mentioned:
        return notes

    # Critic pass: verify fund names exist in text
    critic_prompt = f"""You are verifying an explorer's findings against the source document.

EXPLORER FINDINGS:
{notes.model_dump_json(indent=2)}

SOURCE DOCUMENT (pages {page_start}-{page_end}):
{pages_text}

TASK:
1. For each fund in funds_mentioned, verify the name appears EXACTLY in the source
2. If a name is wrong or slightly different, provide the EXACT name from the source
3. Check if any funds were missed

Return JSON:
{{
  "verified_funds": [
    {{"original": "<explorer's name>", "verified": "<exact name from source or null if not found>", "correct": true/false}}
  ],
  "missed_funds": ["<any fund names the explorer missed>"],
  "observations": ["<any issues found>"]
}}
"""

    from extractor.pydantic_models.llm_responses import ExplorerCriticResult

    client = LLMClient(cost_tracker=cost_tracker)

    try:
        critic_data = await client.complete_structured(
            system_prompt="You verify document analysis results. Be precise about exact text matches.",
            user_prompt=critic_prompt,
            model=critic_model,
            response_model=ExplorerCriticResult,
            agent="explorer_critic",
        )

        # Apply corrections
        corrected_funds = []
        for fund in notes.funds_mentioned:
            # Find verification for this fund
            verification = next(
                (v for v in critic_data.verified_funds
                 if v.original == fund.name),
                None
            )
            if verification and verification.verified:
                # Use verified name
                fund.name = verification.verified
                corrected_funds.append(fund)
            elif verification and verification.correct:
                corrected_funds.append(fund)
            # Skip funds that couldn't be verified

        # Add missed funds
        for missed_name in critic_data.missed_funds:
            from extractor.pydantic_models.pipeline import FundMention
            corrected_funds.append(FundMention(
                name=missed_name,
                page=page_start,  # Don't know exact page
                has_dedicated_section=False,
            ))

        notes.funds_mentioned = corrected_funds
        notes.observations.extend(critic_data.observations)

    except Exception as e:
        # Critic failed - fund names were not verified against source text
        # Add prominent warning so planner/logs show this clearly
        warning = f"CRITIC_FAILED: {e}. {len(notes.funds_mentioned)} fund names are UNVERIFIED and may contain typos or hallucinations."
        notes.observations.insert(0, warning)  # Put at start so it's visible

    return notes


async def run_explorer_with_gleaning(
    pages_text: str,
    page_start: int,
    page_end: int,
    model: str = DEFAULT_MODELS["exploration"],
    cost_tracker: CostTracker | None = None,
    skeleton_context: str = "",
    gleaning_passes: int = 2,
) -> ExplorationNotes:
    """Run explorer with gleaning passes to catch missed discoveries.

    Gleaning focuses on external references - the most commonly missed
    and most impactful discoveries. If we miss "ISIN See KIID", we waste
    many LLM calls searching for ISINs that don't exist.

    Args:
        pages_text: Pre-loaded text from the pages.
        page_start: First page number (1-indexed).
        page_end: Last page number (1-indexed).
        model: LLM model to use.
        cost_tracker: Optional tracker to record token usage.
        skeleton_context: Optional document structure context.
        gleaning_passes: Total number of passes (1 = no gleaning, 2+ = gleaning).

    Returns:
        ExplorationNotes with findings from all passes merged.
    """
    # First pass: normal exploration
    notes = await run_explorer(
        pages_text, page_start, page_end, model, cost_tracker, skeleton_context
    )

    if gleaning_passes <= 1:
        return notes

    # Gleaning passes
    client = LLMClient(cost_tracker=cost_tracker)

    for pass_num in range(2, gleaning_passes + 1):
        # Summarize previous findings for context
        previous_summary = _summarize_notes_for_gleaning(notes)

        gleaning_prompt = GLEANING_EXPLORER_PROMPT.format(
            page_start=page_start,
            page_end=page_end,
            previous_summary=previous_summary,
        )

        try:
            from extractor.pydantic_models.llm_responses import ExplorerGleaningResult

            gleaning_data = await client.complete_structured(
                system_prompt="You are a document analyst focusing on finding external references.",
                user_prompt=f"{gleaning_prompt}\n\nDOCUMENT TEXT (pages {page_start}-{page_end}):\n{pages_text}",
                model=model,
                response_model=ExplorerGleaningResult,
                agent="explorer_gleaning",
            )

            # Merge additional cross references
            for xref in gleaning_data.additional_cross_references:
                if xref.text:
                    notes.cross_references.append(xref)

            # Merge additional tables
            existing_pages = set()
            for t in notes.tables:
                existing_pages.update(range(t.page_start, t.page_end + 1))
            for table in gleaning_data.additional_tables:
                if table.page_start not in existing_pages:
                    notes.tables.append(table)

            # Merge additional observations
            for obs in gleaning_data.additional_observations:
                if obs and obs not in notes.observations:
                    notes.observations.append(f"[gleaning] {obs}")

        except Exception as e:
            notes.observations.append(f"[gleaning pass {pass_num}] Error: {e}")

    return notes


def _summarize_notes_for_gleaning(notes: ExplorationNotes) -> str:
    """Create a summary of exploration notes for the gleaning prompt."""
    lines = []

    if notes.funds_mentioned:
        fund_names = [f.name for f in notes.funds_mentioned[:5]]
        lines.append(f"Funds found: {', '.join(fund_names)}")

    if notes.tables:
        table_types = [t.table_type for t in notes.tables]
        lines.append(f"Tables found: {', '.join(table_types)}")

    if notes.cross_references:
        xref_count = len(notes.cross_references)
        external_count = sum(1 for x in notes.cross_references if x.is_external)
        lines.append(f"Cross-references: {xref_count} total, {external_count} external")

    if notes.inventory:
        present = len(notes.inventory.fields_present)
        external = len(notes.inventory.fields_external)
        lines.append(f"Inventory: {present} fields present, {external} external")

    if not lines:
        return "No significant findings in first pass."

    return "\n".join(lines)
