"""Token and cost tracking for LLM calls.

Tracks usage across the pipeline and calculates costs based on model pricing.
"""

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# Fallback pricing per 1M tokens (USD) when litellm lookup fails.
# Covers common OpenRouter models. Updated Jan 2025.
_FALLBACK_PRICING: dict[str, tuple[float, float]] = {
    # (input_cost_per_1M, output_cost_per_1M)
    "openai/gpt-4o": (2.50, 10.00),
    "openai/gpt-4o-mini": (0.15, 0.60),
    "openai/gpt-4-turbo": (10.00, 30.00),
    "anthropic/claude-3.5-sonnet": (3.00, 15.00),
    "anthropic/claude-3-haiku": (0.25, 1.25),
    "google/gemini-pro-1.5": (1.25, 5.00),
}

_warned_models: set[str] = set()


def _normalize_model_name(model: str) -> str:
    """Strip provider routing prefixes for pricing lookup.

    LiteLLM's pricing database uses model names like 'gpt-4o-mini',
    but OpenRouter passes 'openrouter/openai/gpt-4o-mini'.
    """
    # Strip openrouter/ prefix
    if model.startswith("openrouter/"):
        model = model[len("openrouter/"):]
    return model


def _fallback_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Calculate cost from fallback pricing table."""
    normalized = _normalize_model_name(model)
    if normalized in _FALLBACK_PRICING:
        input_rate, output_rate = _FALLBACK_PRICING[normalized]
        return (prompt_tokens * input_rate + completion_tokens * output_rate) / 1_000_000
    return 0.0


@dataclass
class CallUsage:
    """Usage for a single LLM call."""

    model: str
    prompt_tokens: int
    completion_tokens: int
    agent: str = ""  # Which agent made the call (explorer, planner, extractor, critic)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def cost(self) -> float:
        """Calculate cost in USD using litellm's pricing database.

        Falls back to hardcoded pricing for common models when litellm
        doesn't recognize the model name (e.g., openrouter/ prefixed).
        """
        normalized = _normalize_model_name(self.model)
        try:
            from litellm import completion_cost
            return completion_cost(
                model=normalized,
                prompt_tokens=self.prompt_tokens,
                completion_tokens=self.completion_tokens,
            )
        except Exception:
            # Try fallback pricing
            fallback = _fallback_cost(self.model, self.prompt_tokens, self.completion_tokens)
            if fallback > 0:
                return fallback
            # No pricing available - warn once per model
            if self.model not in _warned_models:
                _warned_models.add(self.model)
                logger.warning(f"No pricing available for model '{self.model}', cost will show as $0")
            return 0.0


@dataclass
class CostTracker:
    """Accumulates token usage and costs across the pipeline."""

    calls: list[CallUsage] = field(default_factory=list)

    def record(self, model: str, usage: Any, agent: str = "") -> CallUsage:
        """Record usage from a LiteLLM response.

        Args:
            model: Model identifier (e.g., "openrouter/openai/gpt-4o-mini")
            usage: The usage object from response.usage
            agent: Which agent made the call

        Returns:
            The recorded CallUsage
        """
        if usage is None:
            return CallUsage(model=model, prompt_tokens=0, completion_tokens=0, agent=agent)

        call = CallUsage(
            model=model,
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
            agent=agent,
        )
        self.calls.append(call)
        return call

    @property
    def total_prompt_tokens(self) -> int:
        return sum(c.prompt_tokens for c in self.calls)

    @property
    def total_completion_tokens(self) -> int:
        return sum(c.completion_tokens for c in self.calls)

    @property
    def total_tokens(self) -> int:
        return self.total_prompt_tokens + self.total_completion_tokens

    @property
    def total_cost(self) -> float:
        """Total cost in USD."""
        return sum(c.cost for c in self.calls)

    @property
    def call_count(self) -> int:
        return len(self.calls)

    def by_agent(self) -> dict[str, dict[str, Any]]:
        """Get breakdown by agent."""
        breakdown: dict[str, dict[str, Any]] = {}
        for call in self.calls:
            agent = call.agent or "unknown"
            if agent not in breakdown:
                breakdown[agent] = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "cost": 0.0}
            breakdown[agent]["calls"] += 1
            breakdown[agent]["prompt_tokens"] += call.prompt_tokens
            breakdown[agent]["completion_tokens"] += call.completion_tokens
            breakdown[agent]["cost"] += call.cost
        return breakdown

    def summary(self) -> str:
        """Return a formatted summary of usage and costs."""
        lines = [
            "=" * 50,
            "COST SUMMARY",
            "=" * 50,
            f"Total API calls: {self.call_count}",
            f"Total tokens: {self.total_tokens:,}",
            f"  - Prompt: {self.total_prompt_tokens:,}",
            f"  - Completion: {self.total_completion_tokens:,}",
            f"Total cost: ${self.total_cost:.4f}",
            "",
            "By agent:",
        ]

        for agent, stats in sorted(self.by_agent().items()):
            lines.append(
                f"  {agent}: {stats['calls']} calls, "
                f"{stats['prompt_tokens'] + stats['completion_tokens']:,} tokens, "
                f"${stats['cost']:.4f}"
            )

        lines.append("=" * 50)
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Export as dict for JSON serialization."""
        return {
            "total_calls": self.call_count,
            "total_tokens": self.total_tokens,
            "prompt_tokens": self.total_prompt_tokens,
            "completion_tokens": self.total_completion_tokens,
            "total_cost_usd": round(self.total_cost, 6),
            "by_agent": self.by_agent(),
        }
