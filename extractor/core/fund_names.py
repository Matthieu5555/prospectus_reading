"""Fund name normalization and deduplication utilities.

Provides:
1. Name normalization and matching functions
2. LLM-based deduplication (ask the model to identify duplicates)
3. Fallback heuristic deduplication
"""

from __future__ import annotations

from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import TYPE_CHECKING

from extractor.core.config import DEFAULT_MODELS
from extractor.core.llm_client import LLMClient


# Fund Name Utilities

def strip_umbrella_prefix(name: str) -> str:
    """Remove common umbrella/company prefixes from a fund name.

    Handles various dash styles (hyphen, en-dash, em-dash) used across
    different prospectus documents.
    """
    prefixes = [
        "JPMorgan Investment Funds - ",
        "JPMorgan Investment Funds – ",
        "JPMorgan Investment Funds— ",
        "JPMorgan Investment Funds-",
        "JPMorgan Funds - ",
        "BlackRock Global Funds - ",
        "Amundi Funds - ",
        "Schroder International Selection Fund - ",
    ]
    for prefix in prefixes:
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


def normalize_fund_name(name: str) -> str:
    """Normalize a fund name for comparison.

    Lowercases, replaces common separators with spaces, and collapses
    multiple whitespace to single spaces.
    """
    s = name.lower().strip()
    for sep in [" - ", " – ", " — ", "-"]:
        s = s.replace(sep, " ")
    return " ".join(s.split())


def fund_names_match(target: str, candidate: str, threshold: float = 0.85) -> bool:
    """Check if two fund names match.

    Checks in order:
    1. Exact match after normalization
    2. Suffix match (handles "Umbrella - Fund Name" vs "Fund Name")
    3. Fuzzy match using SequenceMatcher
    """
    if target == candidate:
        return True

    norm_target = normalize_fund_name(target)
    norm_candidate = normalize_fund_name(candidate)

    if norm_target == norm_candidate:
        return True

    if norm_target.endswith(norm_candidate) or norm_candidate.endswith(norm_target):
        return True

    ratio = SequenceMatcher(None, norm_target, norm_candidate).ratio()
    return ratio >= threshold

if TYPE_CHECKING:
    from extractor.core.cost_tracker import CostTracker


@dataclass
class FundNameMapping:
    """Result of fund name deduplication.

    Attributes:
        canonical_names: Deduplicated list of fund names.
        name_to_canonical: Maps any original name to its canonical form.
        duplicates_found: Number of duplicate names merged.
        merge_groups: For debugging - shows which names were merged.
    """
    canonical_names: list[str]
    name_to_canonical: dict[str, str]
    duplicates_found: int = 0
    merge_groups: dict[str, list[str]] = field(default_factory=dict)


DEDUP_SYSTEM_PROMPT = """You are analyzing fund names from a prospectus document.

Your task: Identify which names refer to the SAME fund and group them together.

Common patterns:
- Same fund, different formats: "Global Macro Fund" = "JPMorgan Investment Funds - Global Macro Fund" = "JPMORGAN INVESTMENT FUNDS - GLOBAL MACRO FUND"
- Different funds with similar names: "Global Macro Fund" ≠ "Global Macro Opportunities Fund" (these are DIFFERENT funds)

Rules:
1. Only group names that DEFINITELY refer to the same fund
2. When in doubt, keep them separate (false negatives are better than false positives)
3. The umbrella/company name alone (e.g., "JPMorgan Investment Funds") is NOT a fund - exclude it
4. For each group, the canonical name should be the SHORTEST variant

Return JSON:
{
  "fund_groups": [
    {
      "canonical": "Global Macro Fund",
      "variants": ["Global Macro Fund", "JPMorgan Investment Funds - Global Macro Fund", "JPMORGAN INVESTMENT FUNDS - GLOBAL MACRO FUND"]
    }
  ],
  "excluded": ["JPMorgan Investment Funds"]  // umbrella names, not actual funds
}"""


async def deduplicate_with_llm(
    names: list[str],
    model: str = DEFAULT_MODELS["reader"],
    cost_tracker: CostTracker | None = None,
) -> FundNameMapping:
    """Use LLM to deduplicate fund names.

    Args:
        names: List of fund names to deduplicate.
        model: LLM model to use.
        cost_tracker: Optional tracker to record token usage.

    Returns:
        FundNameMapping with canonical names and mappings.
    """
    names_list = "\n".join(f"- {name}" for name in sorted(set(names)))
    user_prompt = f"""Analyze these {len(set(names))} fund names and identify duplicates:

{names_list}

Return JSON with fund_groups and excluded lists."""

    client = LLMClient(cost_tracker=cost_tracker)

    try:
        response = await client.complete(
            system_prompt=DEDUP_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            model=model,
            agent="deduplication",
        )
        return _parse_llm_dedup_response(response.content, names)
    except Exception:
        # Fallback to heuristic
        return deduplicate_heuristic([(name, 0) for name in names])


def _is_umbrella_name(name: str) -> bool:
    """Check if a name is likely an umbrella/company name, not a fund."""
    norm = normalize_fund_name(name)
    # Common umbrella patterns - no specific fund identifier after the company name
    umbrella_patterns = [
        "jpmorgan investment funds",
        "jp morgan investment funds",
        "blackrock funds",
        "fidelity funds",
        "vanguard funds",
    ]
    # If the name is JUST the umbrella pattern (no fund-specific suffix), exclude it
    for pattern in umbrella_patterns:
        if norm == pattern or norm == pattern.replace(" ", ""):
            return True
    return False


def _parse_llm_dedup_response(data: dict, original_names: list[str]) -> FundNameMapping:
    """Parse LLM deduplication response into FundNameMapping."""
    canonical_names: list[str] = []
    name_to_canonical: dict[str, str] = {}
    merge_groups: dict[str, list[str]] = {}
    duplicates_found = 0

    # Collect excluded names from LLM response
    excluded = set(data.get("excluded", []))

    # Process fund groups
    for group in data.get("fund_groups", []):
        canonical = group.get("canonical", "")
        variants = group.get("variants", [])

        if not canonical or not variants:
            continue

        # Skip if canonical is an umbrella name
        if _is_umbrella_name(canonical):
            excluded.add(canonical)
            continue

        canonical_names.append(canonical)

        for variant in variants:
            name_to_canonical[variant] = canonical

        if len(variants) > 1:
            merge_groups[canonical] = variants
            duplicates_found += len(variants) - 1

    # Handle any names not in the response (shouldn't happen, but fallback)
    for name in original_names:
        if name not in name_to_canonical and name not in excluded:
            # Check if this is an umbrella name
            if _is_umbrella_name(name):
                continue

            # Check if this is a variant of an existing canonical name (LLM missed it)
            matched = False
            for canonical in canonical_names:
                if fund_names_match(name, canonical, threshold=0.85):
                    name_to_canonical[name] = canonical
                    if canonical in merge_groups:
                        merge_groups[canonical].append(name)
                    else:
                        merge_groups[canonical] = [canonical, name]
                    duplicates_found += 1
                    matched = True
                    break

            if not matched:
                canonical_names.append(name)
                name_to_canonical[name] = name

    canonical_names.sort()

    return FundNameMapping(
        canonical_names=canonical_names,
        name_to_canonical=name_to_canonical,
        duplicates_found=duplicates_found,
        merge_groups=merge_groups,
    )


def deduplicate_heuristic(
    names_with_pages: list[tuple[str, int]],
) -> FundNameMapping:
    """Fallback heuristic deduplication using suffix matching.

    Used when LLM deduplication fails.
    """
    if not names_with_pages:
        return FundNameMapping(canonical_names=[], name_to_canonical={})

    seen = set()
    unique_names = []
    for name, _ in names_with_pages:
        if name not in seen:
            seen.add(name)
            unique_names.append(name)

    groups: list[list[str]] = []

    for name in unique_names:
        matched_group = None
        for group_idx, group_members in enumerate(groups):
            for member in group_members:
                if fund_names_match(name, member):
                    matched_group = group_idx
                    break
            if matched_group is not None:
                break

        if matched_group is not None:
            groups[matched_group].append(name)
        else:
            groups.append([name])

    canonical_names: list[str] = []
    name_to_canonical: dict[str, str] = {}
    merge_groups: dict[str, list[str]] = {}
    duplicates_found = 0

    for group in groups:
        canonical = min(group, key=len)
        canonical_names.append(canonical)

        for name in group:
            name_to_canonical[name] = canonical

        if len(group) > 1:
            merge_groups[canonical] = group
            duplicates_found += len(group) - 1

    canonical_names.sort()

    return FundNameMapping(
        canonical_names=canonical_names,
        name_to_canonical=name_to_canonical,
        duplicates_found=duplicates_found,
        merge_groups=merge_groups,
    )


# Backwards compatibility aliases
deduplicate_fund_names = deduplicate_heuristic


async def deduplicate_from_exploration_notes(
    exploration_notes: list,
    use_llm: bool = True,
    model: str = DEFAULT_MODELS["reader"],
    cost_tracker: CostTracker | None = None,
) -> FundNameMapping:
    """Deduplicate fund names from exploration notes.

    Args:
        exploration_notes: List of ExplorationNotes objects.
        use_llm: Whether to use LLM for deduplication (recommended).
        model: LLM model to use if use_llm=True.
        cost_tracker: Optional tracker to record token usage.

    Returns:
        FundNameMapping with canonical names and mappings.
    """
    names_with_pages: list[tuple[str, int]] = []

    for note in exploration_notes:
        for fund in note.funds_mentioned:
            names_with_pages.append((fund.name, fund.page))

    names = [name for name, _ in names_with_pages]

    if use_llm and names:
        try:
            return await deduplicate_with_llm(names, model, cost_tracker)
        except Exception:
            pass  # Fall through to heuristic

    return deduplicate_heuristic(names_with_pages)
