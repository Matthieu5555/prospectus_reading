"""LLM client for the extraction pipeline.

Provides a unified interface for LLM calls that absorbs boilerplate:
- Message building
- Cost tracking integration
- JSON parsing with error handling
- Retry and fallback via litellm Router

Why does this exist? Before it did, every agent function had the same 8 lines
of setup code: build messages, call router, track cost, parse JSON. This was
copy-pasted across 15+ files. When we needed to add retry logic or change the
response format, we had to update 15 places. This client centralizes that
boilerplate. It's not clever—it's just DRY (Don't Repeat Yourself).

(This could have been a simple function instead of a class. We made it a class
so you can create one client and reuse it across multiple calls within an agent,
which reads cleaner than passing cost_tracker to every function call.)

The pattern this replaces:
    messages = [{"role": "system", ...}, {"role": "user", ...}]
    response = await router.acompletion(...)
    if cost_tracker: cost_tracker.record(...)
    return json.loads(response.choices[0].message.content)
"""

import json
import logging
from dataclasses import dataclass
from typing import Any, TypeVar

from extractor.core.config import LLMConfig
from extractor.core.cost_tracker import CostTracker
from extractor.core.llm_router import router

logger = logging.getLogger(__name__)

# Suppress LiteLLM debug noise (done once at module load)
logging.getLogger("litellm").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)

T = TypeVar("T")


@dataclass
class LLMResponse:
    """Parsed response from an LLM call.

    Attributes:
        content: Parsed JSON content as a dictionary.
        raw_content: Raw string content from the LLM.
        model: Model identifier used for the call.
    """

    content: dict[str, Any]
    raw_content: str
    model: str


class LLMClient:
    """Client for making LLM API calls.

    This class provides a deep interface that hides the complexity of:
    - Message formatting
    - Response parsing
    - Cost tracking
    - Error handling

    Retry and fallback are handled by the litellm Router (core/llm_router.py).

    Usage:
        client = LLMClient(cost_tracker=tracker)

        # Simple call
        response = await client.complete(
            system_prompt="You are an extraction assistant.",
            user_prompt="Extract the fund name from: ...",
            model="openrouter/openai/gpt-4o-mini",
            agent="extractor",
        )
        data = response.content  # Parsed JSON dict

        # Multi-turn call (for gleaning)
        response = await client.complete_with_history(
            messages=[
                {"role": "system", "content": "..."},
                {"role": "user", "content": "..."},
                {"role": "assistant", "content": "..."},
                {"role": "user", "content": "Did you miss anything?"},
            ],
            model="openrouter/openai/gpt-4o-mini",
            agent="gleaning",
        )

        # Structured call (returns validated pydantic model)
        from pydantic import BaseModel
        class FundInfo(BaseModel):
            name: str
            isin: str | None
        result = await client.complete_structured(
            system_prompt="Extract fund info.",
            user_prompt="...",
            model="openrouter/openai/gpt-4o-mini",
            response_model=FundInfo,
            agent="extractor",
        )
    """

    def __init__(self, cost_tracker: CostTracker | None = None) -> None:
        """Initialize the client.

        Args:
            cost_tracker: Optional tracker for recording API token usage.
                         If provided, all calls will be recorded.
        """
        self.cost_tracker = cost_tracker

    async def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        model: str,
        agent: str = "",
        temperature: float | None = None,
        response_format: dict[str, str] | None = None,
    ) -> LLMResponse:
        """Make an LLM completion call with system and user prompts.

        This is the primary method for single-turn extractions.

        Args:
            system_prompt: System message content (instructions/persona).
            user_prompt: User message content (the actual request).
            model: LLM model identifier (e.g., "openrouter/openai/gpt-4o-mini").
            agent: Agent name for cost tracking (e.g., "extractor", "explorer").
            temperature: Sampling temperature. Defaults to 0.0 for determinism.
            response_format: Response format dict. Defaults to JSON.

        Returns:
            LLMResponse with parsed JSON content.

        Raises:
            json.JSONDecodeError: If response is not valid JSON.
            litellm exceptions: For API errors (handled by Router retries).
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        return await self._call_llm(
            messages=messages,
            model=model,
            agent=agent,
            temperature=temperature,
            response_format=response_format,
        )

    async def complete_with_history(
        self,
        messages: list[dict[str, str]],
        model: str,
        agent: str = "",
        temperature: float | None = None,
        response_format: dict[str, str] | None = None,
    ) -> LLMResponse:
        """Make an LLM call with full conversation history.

        Use this for multi-turn conversations like gleaning, where the
        LLM needs to see its previous response to find missed items.

        Args:
            messages: Full message history as list of role/content dicts.
            model: LLM model identifier.
            agent: Agent name for cost tracking.
            temperature: Sampling temperature. Defaults to 0.0.
            response_format: Response format dict. Defaults to JSON.

        Returns:
            LLMResponse with parsed JSON content.

        Raises:
            json.JSONDecodeError: If response is not valid JSON.
        """
        return await self._call_llm(
            messages=messages,
            model=model,
            agent=agent,
            temperature=temperature,
            response_format=response_format,
        )

    async def complete_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        model: str,
        response_model: type[T],
        agent: str = "",
        temperature: float | None = None,
        max_retries: int = 2,
    ) -> T:
        """Make an LLM call that returns a validated pydantic model.

        Uses the Instructor library to:
        - Validate the response against the pydantic model
        - Automatically retry with validation errors in the prompt
          so the LLM can fix its own JSON mistakes

        Args:
            system_prompt: System message content.
            user_prompt: User message content.
            model: LLM model identifier.
            response_model: Pydantic model class to validate against.
            agent: Agent name for cost tracking.
            temperature: Sampling temperature. Defaults to 0.0.
            max_retries: Max validation retries (Instructor sends error back to LLM).

        Returns:
            Validated instance of response_model.
        """
        import instructor

        instructor_client = instructor.from_litellm(router.acompletion)

        result, raw_completion = (
            await instructor_client.chat.completions.create_with_completion(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_model=response_model,
                temperature=temperature
                if temperature is not None
                else LLMConfig.TEMPERATURE,
                max_retries=max_retries,
            )
        )

        if self.cost_tracker:
            self.cost_tracker.record(model, raw_completion.usage, agent=agent)

        return result

    async def _call_llm(
        self,
        messages: list[dict[str, str]],
        model: str,
        agent: str,
        temperature: float | None,
        response_format: dict[str, str] | None,
    ) -> LLMResponse:
        """Internal method that performs the actual LLM call via the Router."""
        response = await router.acompletion(
            model=model,
            messages=messages,
            response_format=response_format or LLMConfig.RESPONSE_FORMAT,
            temperature=temperature if temperature is not None else LLMConfig.TEMPERATURE,
        )

        if self.cost_tracker:
            self.cost_tracker.record(model, response.usage, agent=agent)

        raw_content = response.choices[0].message.content
        try:
            content = json.loads(raw_content)
        except json.JSONDecodeError:
            from json_repair import repair_json
            logger.warning("JSON parse failed, attempting repair")
            content = repair_json(raw_content, return_objects=True)

        return LLMResponse(
            content=content,
            raw_content=raw_content,
            model=model,
        )


# Convenience Functions

async def llm_complete(
    system_prompt: str,
    user_prompt: str,
    model: str,
    cost_tracker: CostTracker | None = None,
    agent: str = "",
) -> dict[str, Any]:
    """Convenience function for one-off LLM calls.

    Creates a temporary LLMClient and makes a single call. For agents
    that make multiple calls, prefer creating an LLMClient instance.

    Args:
        system_prompt: System message content.
        user_prompt: User message content.
        model: LLM model identifier.
        cost_tracker: Optional cost tracker.
        agent: Agent name for cost tracking.

    Returns:
        Parsed JSON content as a dictionary.
    """
    client = LLMClient(cost_tracker=cost_tracker)
    response = await client.complete(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        model=model,
        agent=agent,
    )
    return response.content
