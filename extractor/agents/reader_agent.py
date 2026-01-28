"""Reader agent — the workhorse that reads PDF pages and pulls out structured field values.

The reader handles one fund at a time with fresh LLM context so it never
confuses data from different funds.  The planner tells it which pages to
read; it never sees irrelevant pages.

When a value is NOT_FOUND on the assigned pages the reader can search the
full document for it (search-enhanced extraction).

Knowledge-first extraction: before searching blindly, the reader checks
the knowledge graph to:
1. Skip fields known to live in external documents (KIID, Annual Report …)
2. Jump straight to known page locations instead of scanning everything
3. Use context from earlier phases to inform extraction
"""

import json

from extractor.core.config import DEFAULT_MODELS, SearchLimits, PageLimits
from extractor.core.cost_tracker import CostTracker
from extractor.core.llm_client import LLMClient
from extractor.core.pdf_reader import PDFReader
from extractor.core.value_helpers import is_not_found
from extractor.core.fund_names import fund_names_match, normalize_fund_name
from extractor.core.field_strategy import KnowledgeContext
from extractor.pydantic_models.constraints import (
    CONSTRAINT_FIELD_DESCRIPTIONS,
    CONSTRAINT_FIELDS,
)
from extractor.prompts.reader_prompt import (
    CONSTRAINT_EXTRACTION_PROMPT,
    DISCOVER_BONUS_FIELDS_PROMPT,
    FEE_EXTRACTION_PROMPT,
    GLEANING_FUND_PROMPT,
    GLEANING_SHARE_CLASS_PROMPT,
    ISIN_EXTRACTION_PROMPT,
    SHARE_CLASS_EXTRACTOR_PROMPT,
    SUBFUND_EXTRACTOR_PROMPT,
    UMBRELLA_EXTRACTOR_PROMPT,
)


def extract_raw_value(data) -> str | None:
    """Extract the actual value from provenance or legacy format.

    Handles:
    - "value" string (legacy format)
    - {"value": "...", ...} dict (provenance format)
    """
    if data is None:
        return None
    if isinstance(data, dict):
        return data.get("value")
    return data


# Core Extraction Functions


async def extract_umbrella(
    pages_text: str,
    model: str = DEFAULT_MODELS["reader"],
    cost_tracker: CostTracker | None = None,
) -> dict:
    """Extract umbrella-level information.

    Args:
        pages_text: Text from umbrella info pages.
        model: LLM model to use.
        cost_tracker: Optional tracker to record token usage.

    Returns:
        Dict with umbrella fields.
    """
    client = LLMClient(cost_tracker=cost_tracker)

    response = await client.complete(
        system_prompt=UMBRELLA_EXTRACTOR_PROMPT,
        user_prompt=f"Extract umbrella information from:\n\n{pages_text}",
        model=model,
        agent="extractor",
    )

    return response.content


async def extract_subfund(
    fund_name: str,
    pages_text: str,
    model: str = DEFAULT_MODELS["reader"],
    gleaning_passes: int = 1,
    cost_tracker: CostTracker | None = None,
) -> dict:
    """Extract sub-fund information with optional gleaning.

    Args:
        fund_name: Exact name of the fund to extract.
        pages_text: Text from fund's dedicated pages.
        model: LLM model to use.
        gleaning_passes: Number of extraction passes (1 = no gleaning, 2+ = gleaning).
        cost_tracker: Optional tracker to record token usage.

    Returns:
        Dict with sub-fund fields.
    """
    client = LLMClient(cost_tracker=cost_tracker)
    system_prompt = SUBFUND_EXTRACTOR_PROMPT.format(fund_name=fund_name)
    user_prompt = f'Extract information for "{fund_name}" from:\n\n{pages_text}'

    response = await client.complete(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        model=model,
        agent="extractor",
    )

    result = response.content
    # Ensure name is set with provenance format
    existing_name = result.get("name")
    if not isinstance(existing_name, dict) or "value" not in existing_name:
        result["name"] = {
            "value": fund_name,
            "source_page": None,
            "source_quote": None,
            "rationale": "Fund name from extraction task",
            "confidence": 1.0,
        }

    # Gleaning passes for constraints
    for pass_num in range(2, gleaning_passes + 1):
        # Check if we have NOT_FOUND constraints worth gleaning
        missing_constraints = [
            field for field in CONSTRAINT_FIELDS
            if is_not_found(result.get(field))
        ]
        if not missing_constraints:
            break  # All constraints found

        prev_extraction = json.dumps(
            {field: extract_raw_value(result.get(field)) for field in CONSTRAINT_FIELDS},
            indent=2,
        )

        gleaning_prompt = GLEANING_FUND_PROMPT.format(previous_extraction=prev_extraction)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": json.dumps(result)},
            {"role": "user", "content": gleaning_prompt},
        ]

        gleaning_response = await client.complete_with_history(
            messages=messages,
            model=model,
            agent="gleaning",
        )

        additions = gleaning_response.content.get("additions", {})

        if not additions:
            break  # No new data found

        # Merge additions into result (only for fields that were NOT_FOUND)
        for field, value in additions.items():
            if field in CONSTRAINT_FIELDS and is_not_found(result.get(field)):
                if not is_not_found(value):
                    result[field] = value

    return result


async def extract_share_classes(
    fund_name: str,
    pages_text: str,
    model: str = DEFAULT_MODELS["reader"],
    gleaning_passes: int = 1,
    cost_tracker: CostTracker | None = None,
) -> list[dict]:
    """Extract share classes for a fund with optional gleaning.

    Args:
        fund_name: Exact name of the fund.
        pages_text: Text from share class pages or tables.
        model: LLM model to use.
        gleaning_passes: Number of extraction passes (1 = no gleaning, 2+ = gleaning).
        cost_tracker: Optional tracker to record token usage.

    Returns:
        List of share class dicts.
    """
    client = LLMClient(cost_tracker=cost_tracker)
    system_prompt = SHARE_CLASS_EXTRACTOR_PROMPT.format(fund_name=fund_name)
    user_prompt = f'Extract share classes for "{fund_name}" from:\n\n{pages_text}'

    response = await client.complete(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        model=model,
        agent="extractor",
    )

    result = response.content
    share_classes = result.get("share_classes", [])

    # Gleaning passes - ask LLM to review and find missed items
    for pass_num in range(2, gleaning_passes + 1):
        if not share_classes:
            break  # Nothing to glean from

        prev_names = [share_class.get("name", "Unknown") for share_class in share_classes]
        prev_extraction = json.dumps({"extracted_names": prev_names, "count": len(share_classes)}, indent=2)

        gleaning_prompt = GLEANING_SHARE_CLASS_PROMPT.format(previous_extraction=prev_extraction)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": json.dumps(result)},
            {"role": "user", "content": gleaning_prompt},
        ]

        gleaning_response = await client.complete_with_history(
            messages=messages,
            model=model,
            agent="gleaning",
        )

        additional = gleaning_response.content.get("additional_share_classes", [])

        if not additional:
            break  # No more items found

        # Merge additional share classes (avoid duplicates by name)
        existing_names = {normalize_fund_name(share_class.get("name", "")) for share_class in share_classes}
        for share_class in additional:
            normalized = normalize_fund_name(share_class.get("name", ""))
            if normalized and normalized not in existing_names:
                share_classes.append(share_class)
                existing_names.add(normalized)

    return share_classes


def apply_broadcast_data(
    fund: dict,
    broadcast_data: dict[str, list[dict]],
    fund_name: str,
) -> dict:
    """Apply broadcast table data to a fund.

    Looks up the fund in broadcast tables and merges data.
    Preserves provenance format when copying fields.

    Args:
        fund: Fund dict to update.
        broadcast_data: Dict of table_type -> list of rows.
        fund_name: Fund name for lookup.

    Returns:
        Updated fund dict.
    """
    if "isin" in broadcast_data:
        for row in broadcast_data["isin"]:
            row_fund = row.get("fund_name", "")
            row_fund_val = extract_raw_value(row_fund)
            if fund_names_match(fund_name, row_fund_val):
                # Found matching fund - add to share classes
                share_classes = fund.get("share_classes", [])
                # Try to find matching share class
                class_name = row.get("share_class_name", "")
                class_name_val = extract_raw_value(class_name)
                for share_class in share_classes:
                    sc_name = share_class.get("name", "")
                    sc_name_val = extract_raw_value(sc_name)
                    if sc_name_val == class_name_val:
                        if row.get("isin") and not is_not_found(row.get("isin")):
                            share_class["isin"] = row["isin"]  # Preserve full provenance
                        break

    if "fee" in broadcast_data:
        for row in broadcast_data["fee"]:
            row_fund = row.get("fund_name", "")
            row_fund_val = extract_raw_value(row_fund)
            if fund_names_match(fund_name, row_fund_val):
                share_classes = fund.get("share_classes", [])
                class_name = row.get("share_class_name", "")
                class_name_val = extract_raw_value(class_name)
                for share_class in share_classes:
                    sc_name = share_class.get("name", "")
                    sc_name_val = extract_raw_value(sc_name)
                    if sc_name_val == class_name_val:
                        for field in ["management_fee", "entry_fee", "exit_fee", "ongoing_charges", "performance_fee"]:
                            if not is_not_found(row.get(field)):
                                share_class[field] = row[field]  # Preserve full provenance
                        break

    return fund


# Knowledge-First Helpers


def make_external_ref_value(field_name: str, external_doc: str, quote: str | None = None) -> dict:
    """Create a NOT_FOUND value with external reference information.

    Used when we know a field is documented in an external document
    (KIID, Annual Report, etc.) rather than the prospectus.
    """
    return {
        "value": "NOT_FOUND",
        "not_found_reason": "in_external_doc",
        "external_reference": external_doc,
        "source_page": None,
        "source_quote": quote[:100] if quote else None,
        "rationale": f"{field_name} is documented in {external_doc}, not in this prospectus",
        "confidence": 0.95,
    }


def mark_share_classes_external(
    share_classes: list[dict],
    field_name: str,
    external_doc: str,
    quote: str | None = None,
) -> list[dict]:
    """Mark a field as external across all share classes.

    Used when we know from exploration that a field (e.g., ISIN) is
    documented externally for all funds.
    """
    for sc in share_classes:
        if is_not_found(sc.get(field_name)):
            sc[field_name] = make_external_ref_value(field_name, external_doc, quote)
    return share_classes


# Search-Enhanced Extraction Functions


def _resolve_pages(
    field_name: str,
    fund_name: str,
    search_context: PDFReader,
    knowledge_ctx: KnowledgeContext | None,
    search_category: str,
    max_search: int,
    max_pages: int,
) -> tuple[str | None, list[int]]:
    """Resolve pages for a field via knowledge context or pattern search.

    Returns (external_doc, []) if the field is external,
    ("", pages) if pages were found, or ("", []) if nothing found.
    """
    if knowledge_ctx:
        strategy = knowledge_ctx.get_field_strategy(field_name, fund_name)
        if strategy.strategy == "external":
            return strategy.external_doc, []
        if strategy.pages:
            return "", strategy.pages[:max_pages]

    hits = search_context.search_patterns(search_category, max_results=max_search)
    if not hits:
        return "", []
    pages = sorted(set(hit["page"] for hit in hits))[:max_pages]
    return "", pages


def _read_pages(search_context: PDFReader, pages: list[int]) -> str:
    """Read and concatenate pages from search context."""
    return "\n\n".join(search_context.read_pages(p, p) for p in pages)


async def search_and_extract_isins(
    fund_name: str,
    search_context: PDFReader,
    existing_share_classes: list[dict],
    model: str = DEFAULT_MODELS["reader"],
    knowledge_ctx: KnowledgeContext | None = None,
    cost_tracker: CostTracker | None = None,
) -> list[dict]:
    """Search document for ISINs and update share classes."""
    # Check if external
    external_doc, pages = _resolve_pages(
        "isin", fund_name, search_context, knowledge_ctx,
        "isin", SearchLimits.ISIN_SEARCH, PageLimits.ISIN_PAGES,
    )
    if external_doc:
        strategy = knowledge_ctx.get_field_strategy("isin", fund_name)
        return mark_share_classes_external(
            existing_share_classes, "isin", external_doc, strategy.external_quote,
        )

    # Check if we need to search
    if not any(is_not_found(sc.get("isin")) for sc in existing_share_classes):
        return existing_share_classes

    if not pages:
        return existing_share_classes

    pages_text = _read_pages(search_context, pages)
    client = LLMClient(cost_tracker=cost_tracker)

    response = await client.complete(
        system_prompt=ISIN_EXTRACTION_PROMPT.format(fund_name=fund_name),
        user_prompt=f'Find ISINs for "{fund_name}" share classes in:\n\n{pages_text}',
        model=model,
        agent="extractor",
    )

    for found in response.content.get("share_class_isins", []):
        isin_data = found.get("isin", "")
        isin_value = extract_raw_value(isin_data)
        if isin_value:
            search_context.record_pattern("isin", isin_value)

        sc_name_to_match = extract_raw_value(found.get("share_class_name", ""))

        # Try exact match first
        for share_class in existing_share_classes:
            if extract_raw_value(share_class.get("name")) == sc_name_to_match:
                if isin_value and len(isin_value) == 12:
                    share_class["isin"] = isin_data
                break
        else:
            # Fuzzy match
            class_name_lower = (sc_name_to_match or "").lower()
            for share_class in existing_share_classes:
                sc_name_lower = (extract_raw_value(share_class.get("name")) or "").lower()
                if class_name_lower in sc_name_lower or sc_name_lower in class_name_lower:
                    if is_not_found(share_class.get("isin")) and isin_value and len(isin_value) == 12:
                        share_class["isin"] = isin_data
                        break

    return existing_share_classes


async def search_and_extract_fees(
    fund_name: str,
    search_context: PDFReader,
    existing_share_classes: list[dict],
    model: str = DEFAULT_MODELS["reader"],
    knowledge_ctx: KnowledgeContext | None = None,
    cost_tracker: CostTracker | None = None,
) -> list[dict]:
    """Search document for fee data and update share classes."""
    fee_fields = ["management_fee", "entry_fee", "exit_fee", "ongoing_charges"]

    # Check if external
    external_doc, pages = _resolve_pages(
        "fee", fund_name, search_context, knowledge_ctx,
        "fee", SearchLimits.FEE_SEARCH, PageLimits.FEE_PAGES,
    )
    if external_doc:
        strategy = knowledge_ctx.get_field_strategy("fee", fund_name)
        for field in fee_fields:
            existing_share_classes = mark_share_classes_external(
                existing_share_classes, field, external_doc, strategy.external_quote,
            )
        return existing_share_classes

    # Check if we need to search
    if not any(is_not_found(sc.get(f)) for sc in existing_share_classes for f in fee_fields):
        return existing_share_classes

    if not pages:
        return existing_share_classes

    pages_text = _read_pages(search_context, pages)
    client = LLMClient(cost_tracker=cost_tracker)

    response = await client.complete(
        system_prompt=FEE_EXTRACTION_PROMPT.format(fund_name=fund_name),
        user_prompt=f'Find fees for "{fund_name}" share classes in:\n\n{pages_text}',
        model=model,
        agent="extractor",
    )

    for found in response.content.get("share_class_fees", []):
        class_name_val = extract_raw_value(found.get("share_class_name", ""))
        for share_class in existing_share_classes:
            sc_name = extract_raw_value(share_class.get("name", ""))
            if sc_name == class_name_val or (class_name_val or "").lower() in (sc_name or "").lower():
                for field in fee_fields + ["performance_fee"]:
                    if not is_not_found(found.get(field)) and is_not_found(share_class.get(field)):
                        share_class[field] = found[field]
                break

    return existing_share_classes


async def search_and_extract_constraints(
    fund_name: str,
    search_context: PDFReader,
    existing_fund_data: dict,
    model: str = DEFAULT_MODELS["reader"],
    knowledge_ctx: KnowledgeContext | None = None,
    cost_tracker: CostTracker | None = None,
) -> dict:
    """Search document for binding constraints and update fund data."""
    missing_fields = [
        field for field in CONSTRAINT_FIELDS
        if is_not_found(existing_fund_data.get(field))
    ]
    if not missing_fields:
        return existing_fund_data

    # Check external refs per-field
    if knowledge_ctx:
        fields_to_skip = []
        for field in missing_fields:
            strategy = knowledge_ctx.get_field_strategy(field, fund_name)
            if strategy.strategy == "external":
                existing_fund_data[field] = make_external_ref_value(
                    field, strategy.external_doc, strategy.external_quote
                )
                fields_to_skip.append(field)
        missing_fields = [f for f in missing_fields if f not in fields_to_skip]
        if not missing_fields:
            return existing_fund_data

    # Search across constraint categories
    all_hits = []
    for category in ["restriction", "leverage", "derivative"]:
        all_hits.extend(search_context.search_patterns(category, max_results=SearchLimits.CONSTRAINT_SEARCH))
    if not all_hits:
        return existing_fund_data

    pages = sorted(set(hit["page"] for hit in all_hits))[:PageLimits.CONSTRAINT_PAGES]
    if not pages:
        return existing_fund_data

    pages_text = _read_pages(search_context, pages)
    fields_desc = "\n".join(
        f"- {field}: {CONSTRAINT_FIELD_DESCRIPTIONS.get(field, field)}"
        for field in missing_fields
    )
    client = LLMClient(cost_tracker=cost_tracker)

    response = await client.complete(
        system_prompt=CONSTRAINT_EXTRACTION_PROMPT.format(fund_name=fund_name, fields_to_find=fields_desc),
        user_prompt=f'Find binding constraints for "{fund_name}" in:\n\n{pages_text}',
        model=model,
        agent="extractor",
    )

    found = response.content
    for field in CONSTRAINT_FIELDS:
        if not is_not_found(found.get(field)) and is_not_found(existing_fund_data.get(field)):
            existing_fund_data[field] = found[field]

    return existing_fund_data


async def extract_with_search(
    fund_name: str,
    pages_text: str,
    search_context: PDFReader | None,
    model: str = DEFAULT_MODELS["reader"],
    gleaning_passes: int = 1,
    cost_tracker: CostTracker | None = None,
    knowledge_ctx: KnowledgeContext | None = None,
) -> tuple[dict, list[dict]]:
    """Extract fund and share classes with search fallback for NOT_FOUND fields.

    This is the main entry point for search-enhanced extraction. It:
    1. Extracts from assigned pages (with optional gleaning)
    2. If ISINs/fees are NOT_FOUND, searches the document and fills gaps
    3. If binding constraints are NOT_FOUND, searches for those too

    KNOWLEDGE-FIRST: When knowledge_ctx is provided, the function will:
    - Skip searching for fields known to be in external documents
    - Use known page locations instead of blind search
    - Mark external fields with appropriate NOT_FOUND reasons

    Args:
        fund_name: Fund name to extract.
        pages_text: Text from assigned pages.
        search_context: Optional search context for document-wide search.
        model: LLM model to use.
        gleaning_passes: Number of extraction passes (1 = no gleaning, 2+ = gleaning).
        cost_tracker: Optional tracker to record token usage.
        knowledge_ctx: Optional knowledge context for checking external refs and known locations.

    Returns:
        Tuple of (fund_data, share_classes).
    """
    import asyncio
    import time
    import logging

    logger = logging.getLogger(__name__)
    timings: dict[str, float] = {}

    t0 = time.perf_counter()
    fund_data = await extract_subfund(fund_name, pages_text, model, gleaning_passes, cost_tracker)
    timings["extract_subfund"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    share_classes = await extract_share_classes(fund_name, pages_text, model, gleaning_passes, cost_tracker)
    timings["extract_share_classes"] = time.perf_counter() - t0

    # If no search context, return as-is
    if not search_context:
        logger.debug(f"[{fund_name}] Extraction timings: {timings}")
        return fund_data, share_classes

    # Run constraint, ISIN, and fee searches IN PARALLEL (they're independent)
    t0 = time.perf_counter()

    async def search_constraints():
        return await search_and_extract_constraints(
            fund_name, search_context, fund_data, model, knowledge_ctx, cost_tracker
        )

    async def search_isins():
        return await search_and_extract_isins(
            fund_name, search_context, share_classes, model, knowledge_ctx, cost_tracker
        )

    async def search_fees():
        return await search_and_extract_fees(
            fund_name, search_context, share_classes, model, knowledge_ctx, cost_tracker
        )

    results = await asyncio.gather(
        search_constraints(),
        search_isins(),
        search_fees(),
        return_exceptions=True,
    )

    timings["parallel_searches"] = time.perf_counter() - t0

    # Unpack results (handle exceptions gracefully)
    if isinstance(results[0], Exception):
        logger.warning(f"[{fund_name}] Constraint search failed: {results[0]}")
    else:
        fund_data = results[0]

    if isinstance(results[1], Exception):
        logger.warning(f"[{fund_name}] ISIN search failed: {results[1]}")
    else:
        share_classes = results[1]

    if isinstance(results[2], Exception):
        logger.warning(f"[{fund_name}] Fee search failed: {results[2]}")
    else:
        # Fee search returns updated share_classes, merge with ISIN results
        fee_share_classes = results[2]
        # Merge fee data into share_classes (ISIN search result)
        if not isinstance(results[1], Exception):
            share_classes = _merge_share_class_data(share_classes, fee_share_classes)
        else:
            share_classes = fee_share_classes

    # Log timing summary
    total = sum(timings.values())
    logger.debug(
        f"[{fund_name}] Extraction timings: "
        f"subfund={timings['extract_subfund']:.1f}s, "
        f"share_classes={timings['extract_share_classes']:.1f}s, "
        f"parallel_searches={timings['parallel_searches']:.1f}s, "
        f"total={total:.1f}s"
    )

    return fund_data, share_classes


def _merge_share_class_data(base: list[dict], updates: list[dict]) -> list[dict]:
    """Merge share class data from parallel searches.

    The base list (from ISIN search) is updated with fee data from updates list.
    """
    if not updates:
        return base

    updates_by_name = {}
    for sc in updates:
        name = sc.get("name", {})
        if isinstance(name, dict):
            name = name.get("value", "")
        updates_by_name[name] = sc

    fee_fields = ["management_fee", "entry_fee", "exit_fee", "ongoing_charges", "performance_fee"]
    for sc in base:
        name = sc.get("name", {})
        if isinstance(name, dict):
            name = name.get("value", "")

        if name in updates_by_name:
            update_sc = updates_by_name[name]
            for field in fee_fields:
                if field in update_sc and field not in sc:
                    sc[field] = update_sc[field]
                elif field in update_sc and is_not_found(sc.get(field)) and not is_not_found(update_sc.get(field)):
                    sc[field] = update_sc[field]

    return base


# Schema Discovery Functions


async def discover_bonus_fields(
    entity_name: str,
    pages_text: str,
    extracted_data: dict,
    model: str = DEFAULT_MODELS["reader"],
    cost_tracker: CostTracker | None = None,
) -> tuple[list[dict], list[dict]]:
    """Discover PMS-relevant fields beyond the standard schema.

    Args:
        entity_name: Fund or umbrella name.
        pages_text: Text from the entity's pages.
        extracted_data: Already extracted standard fields.
        model: LLM model to use.
        cost_tracker: Optional tracker to record token usage.

    Returns:
        Tuple of (discovered_fields, schema_suggestions).
    """
    client = LLMClient(cost_tracker=cost_tracker)

    extracted_summary = json.dumps(list(extracted_data.keys()), indent=2)
    system_prompt = DISCOVER_BONUS_FIELDS_PROMPT.format(extracted_fields=extracted_summary)

    response = await client.complete(
        system_prompt=system_prompt,
        user_prompt=f'Find bonus fields for "{entity_name}" in:\n\n{pages_text[:15000]}',
        model=model,
        agent="discovery",
    )

    discovered = response.content.get("discovered_fields", [])
    suggestions = response.content.get("schema_suggestions", [])

    # Add entity name to discovered fields
    for field in discovered:
        field["entity_name"] = entity_name

    return discovered, suggestions
