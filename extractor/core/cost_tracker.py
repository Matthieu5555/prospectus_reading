"""Token and cost tracking for LLM calls.

Tracks usage across the pipeline and calculates costs based on model pricing.
"""

from dataclasses import dataclass, field
from typing import Any


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
        """Calculate cost in USD using litellm's pricing database."""
        try:
            from litellm import completion_cost
            return completion_cost(
                model=self.model,
                prompt_tokens=self.prompt_tokens,
                completion_tokens=self.completion_tokens,
            )
        except Exception:
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
