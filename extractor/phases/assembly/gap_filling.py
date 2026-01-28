"""Gap filling logic for umbrella-level constraints.

When a field is NOT_FOUND across many funds (>80%), it's likely an
umbrella-level field that should be extracted once and applied to all funds.
"""

from extractor.core.config import ConfidenceThresholds, PageLimits, SearchLimits
from extractor.core.llm_client import LLMClient
from extractor.core import CONSTRAINT_FIELDS, is_not_found
from extractor.prompts import UMBRELLA_CONSTRAINT_PROMPT


async def fill_gaps(context, log_fn) -> int:
    """Fill gaps with umbrella-level constraints.

    Analyzes funds_data to find fields that are NOT_FOUND in >80% of funds,
    then searches for umbrella-level constraints that apply to all funds.

    Args:
        context: PhaseContext with funds_data and search capabilities
        log_fn: Logging function for progress messages

    Returns:
        Number of fields filled
    """
    if not context.funds_data:
        return 0

    # Count NOT_FOUND by field
    constraint_fields = CONSTRAINT_FIELDS
    not_found_counts = {f: 0 for f in constraint_fields}

    for fund in context.funds_data:
        for field in constraint_fields:
            if is_not_found(fund.get(field)):
                not_found_counts[field] += 1

    # Find umbrella-level fields (80%+ missing indicates umbrella-level data)
    total = len(context.funds_data)
    umbrella_fields = [
        f for f, c in not_found_counts.items()
        if c > total * ConfidenceThresholds.UMBRELLA_GAP
    ]

    if not umbrella_fields:
        log_fn("No common gaps detected")
        return 0

    log_fn(f"NOT_FOUND counts: {not_found_counts}")
    log_fn(f"Searching for umbrella-level: {umbrella_fields}")

    # Search for constraint pages
    search_context = context.create_search_context()
    all_hits = []
    for category in ["restriction", "leverage", "derivative"]:
        hits = search_context.search_patterns(category, max_results=SearchLimits.GENERAL_SEARCH)
        all_hits.extend(hits)

    if not all_hits:
        return 0

    # Read pages (limited to prevent token explosion)
    pages = sorted(set(hit["page"] for hit in all_hits))[:PageLimits.GAP_FILL_PAGES]
    pages_text = ""
    for page in pages:
        pages_text += context.pdf.read_pages(page, page) + "\n\n"

    # Extract umbrella constraints via LLM
    constraints = await extract_umbrella_constraints(
        pages_text,
        umbrella_fields,
        context.reader_model,
        context.cost_tracker,
        log_fn,
    )

    if not constraints:
        return 0

    # Apply to funds
    filled = 0
    for fund in context.funds_data:
        for field, value in constraints.items():
            if value and value != "NOT_FOUND":
                if is_not_found(fund.get(field)):
                    fund[field] = value
                    filled += 1

    log_fn(f"Filled {filled} fields from umbrella constraints")
    return filled


async def extract_umbrella_constraints(
    pages_text: str,
    fields: list[str],
    model: str,
    cost_tracker,
    log_fn,
) -> dict:
    """Extract umbrella-level constraints via LLM.

    Args:
        pages_text: Combined text from constraint pages
        fields: List of field names to extract
        model: LLM model to use
        cost_tracker: CostTracker for API usage
        log_fn: Logging function

    Returns:
        Dict mapping field names to extracted values
    """
    fields_desc = "\n".join([
        f"- {field}: {'investment limits' if 'restriction' in field else 'leverage limits' if 'leverage' in field else 'derivatives policy'}"
        for field in fields
    ])

    prompt = UMBRELLA_CONSTRAINT_PROMPT.format(fields_desc=fields_desc)

    client = LLMClient(cost_tracker=cost_tracker)

    try:
        response = await client.complete(
            system_prompt=prompt,
            user_prompt=f"Find umbrella-level constraints:\n\n{pages_text}",
            model=model,
            agent="assembly",
        )
        return response.content
    except Exception as e:
        log_fn(f"Umbrella constraint extraction failed: {e}", "error")
        return {}
