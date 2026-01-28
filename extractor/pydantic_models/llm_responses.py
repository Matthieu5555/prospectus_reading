"""Pydantic models for ad-hoc LLM responses.

These models are used with Instructor's complete_structured() for responses
that don't have their own dedicated module (gleaning, explorer critic).
"""

from pydantic import BaseModel

from extractor.pydantic_models.exploration_models import CrossReference, TableDiscovery


class ExplorerGleaningResult(BaseModel):
    """Additional findings from a gleaning pass."""

    additional_cross_references: list[CrossReference] = []
    additional_tables: list[TableDiscovery] = []
    additional_observations: list[str] = []


class FundVerification(BaseModel):
    """Verification of a single fund name against source text."""

    original: str
    verified: str | None = None
    correct: bool = False


class ExplorerCriticResult(BaseModel):
    """Critic verification of explorer findings."""

    verified_funds: list[FundVerification] = []
    missed_funds: list[str] = []
    observations: list[str] = []
