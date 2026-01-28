"""Agent implementations for the extraction pipeline.

Each agent handles one phase of the extraction process with fresh LLM context,
so data from different funds or documents never leaks between calls.
"""

from extractor.agents.explorer_agent import (
    run_explorer,
    run_explorer_with_critic,
    run_explorer_with_gleaning,
)
from extractor.agents.planner_agent import (
    run_planner,
    compute_page_ranges,
    enrich_plan_with_pages,
    generate_recipes,
)
from extractor.agents.reader_agent import (
    extract_umbrella,
    extract_subfund,
    extract_share_classes,
    apply_broadcast_data,
    search_and_extract_isins,
    search_and_extract_fees,
    search_and_extract_constraints,
    extract_with_search,
    discover_bonus_fields,
    make_external_ref_value,
    mark_share_classes_external,
)
from extractor.agents.critic_agent import (
    run_critic,
    verify_and_correct,
    compute_confidence_score,
    verify_fund_extraction,
)

__all__ = [
    # Explorer
    "run_explorer",
    "run_explorer_with_critic",
    "run_explorer_with_gleaning",
    # Planner
    "run_planner",
    "compute_page_ranges",
    "enrich_plan_with_pages",
    "generate_recipes",
    # Extractor
    "extract_umbrella",
    "extract_subfund",
    "extract_share_classes",
    "apply_broadcast_data",
    "search_and_extract_isins",
    "search_and_extract_fees",
    "search_and_extract_constraints",
    "extract_with_search",
    "discover_bonus_fields",
    "make_external_ref_value",
    "mark_share_classes_external",
    # Critic
    "run_critic",
    "verify_and_correct",
    "compute_confidence_score",
    "verify_fund_extraction",
]
