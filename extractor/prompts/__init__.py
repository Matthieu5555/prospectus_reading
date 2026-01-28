"""Prompt templates for LLM agents.

Each module contains the system prompts and user prompt builders
for a specific agent type.
"""

from extractor.prompts.explorer_prompt import EXPLORER_SYSTEM_PROMPT, build_explorer_prompt
from extractor.prompts.planner_prompt import PLANNER_SYSTEM_PROMPT, build_planner_prompt
from extractor.prompts.reader_prompt import (
    UMBRELLA_EXTRACTOR_PROMPT,
    SUBFUND_EXTRACTOR_PROMPT,
    SHARE_CLASS_EXTRACTOR_PROMPT,
    BROADCAST_TABLE_PROMPT,
    ISIN_EXTRACTION_PROMPT,
    FEE_EXTRACTION_PROMPT,
    CONSTRAINT_EXTRACTION_PROMPT,
    UMBRELLA_CONSTRAINT_PROMPT,
    GLEANING_SHARE_CLASS_PROMPT,
    GLEANING_FUND_PROMPT,
    DISCOVER_BONUS_FIELDS_PROMPT,
)
from extractor.prompts.critic_prompt import CRITIC_SYSTEM_PROMPT, build_critic_prompt

__all__ = [
    # Explorer
    "EXPLORER_SYSTEM_PROMPT",
    "build_explorer_prompt",
    # Planner
    "PLANNER_SYSTEM_PROMPT",
    "build_planner_prompt",
    # Extractor
    "UMBRELLA_EXTRACTOR_PROMPT",
    "SUBFUND_EXTRACTOR_PROMPT",
    "SHARE_CLASS_EXTRACTOR_PROMPT",
    "BROADCAST_TABLE_PROMPT",
    "ISIN_EXTRACTION_PROMPT",
    "FEE_EXTRACTION_PROMPT",
    "CONSTRAINT_EXTRACTION_PROMPT",
    "UMBRELLA_CONSTRAINT_PROMPT",
    "GLEANING_SHARE_CLASS_PROMPT",
    "GLEANING_FUND_PROMPT",
    "DISCOVER_BONUS_FIELDS_PROMPT",
    # Critic
    "CRITIC_SYSTEM_PROMPT",
    "build_critic_prompt",
]
