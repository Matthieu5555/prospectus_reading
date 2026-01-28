"""Tests for extractor.core.llm_client module.

Tests the LLMClient class with mocked API calls:
- complete(): Single-turn completion
- complete_with_history(): Multi-turn conversation (gleaning)
- Cost tracking integration
"""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from extractor.core.llm_client import LLMClient, LLMResponse
from extractor.core.cost_tracker import CostTracker


# =============================================================================
# LLMResponse tests
# =============================================================================


class TestLLMResponse:
    """Tests for LLMResponse dataclass."""

    def test_response_stores_content(self):
        response = LLMResponse(
            content={"name": "Test Fund"},
            raw_content='{"name": "Test Fund"}',
            model="openrouter/openai/gpt-4o-mini",
        )
        assert response.content == {"name": "Test Fund"}

    def test_response_stores_raw_content(self):
        response = LLMResponse(
            content={"name": "Test Fund"},
            raw_content='{"name": "Test Fund"}',
            model="openrouter/openai/gpt-4o-mini",
        )
        assert response.raw_content == '{"name": "Test Fund"}'

    def test_response_stores_model(self):
        response = LLMResponse(
            content={},
            raw_content="{}",
            model="openrouter/openai/gpt-4o-mini",
        )
        assert response.model == "openrouter/openai/gpt-4o-mini"


# =============================================================================
# LLMClient tests
# =============================================================================


class TestLLMClient:
    """Tests for LLMClient class."""

    @pytest.fixture
    def mock_router(self):
        """Create a mock for the litellm Router's acompletion."""
        with patch("extractor.core.llm_client.router") as mock:
            mock.acompletion = AsyncMock()
            mock.acompletion.return_value = MagicMock(
                choices=[MagicMock(
                    message=MagicMock(content='{"name": "Test Fund", "isin": "LU0123456789"}')
                )],
                usage=MagicMock(prompt_tokens=100, completion_tokens=50)
            )
            yield mock

    @pytest.mark.asyncio
    async def test_complete_returns_parsed_json(self, mock_router):
        client = LLMClient()
        response = await client.complete(
            system_prompt="You are an extractor.",
            user_prompt="Extract fund info.",
            model="test-model",
            agent="test",
        )
        assert response.content == {"name": "Test Fund", "isin": "LU0123456789"}

    @pytest.mark.asyncio
    async def test_complete_calls_router_with_correct_args(self, mock_router):
        client = LLMClient()
        await client.complete(
            system_prompt="System prompt",
            user_prompt="User prompt",
            model="test-model",
            agent="test",
        )

        mock_router.acompletion.assert_called_once()
        call_kwargs = mock_router.acompletion.call_args.kwargs
        assert call_kwargs["model"] == "test-model"
        assert call_kwargs["response_format"] == {"type": "json_object"}
        assert call_kwargs["temperature"] == 0
        # No api_key param — Router owns keys
        assert "api_key" not in call_kwargs

        messages = call_kwargs["messages"]
        assert len(messages) == 2
        assert messages[0] == {"role": "system", "content": "System prompt"}
        assert messages[1] == {"role": "user", "content": "User prompt"}

    @pytest.mark.asyncio
    async def test_complete_tracks_costs(self, mock_router):
        tracker = CostTracker()
        client = LLMClient(cost_tracker=tracker)
        await client.complete(
            system_prompt="System",
            user_prompt="User",
            model="test-model",
            agent="extractor",
        )
        assert tracker.call_count == 1
        assert tracker.total_prompt_tokens == 100
        assert tracker.total_completion_tokens == 50

    @pytest.mark.asyncio
    async def test_complete_with_history_sends_full_conversation(self, mock_router):
        client = LLMClient()
        messages = [
            {"role": "system", "content": "You are an extractor."},
            {"role": "user", "content": "Extract info."},
            {"role": "assistant", "content": '{"partial": true}'},
            {"role": "user", "content": "What did you miss?"},
        ]
        await client.complete_with_history(
            messages=messages, model="test-model", agent="gleaning",
        )
        call_kwargs = mock_router.acompletion.call_args.kwargs
        assert call_kwargs["messages"] == messages

    @pytest.mark.asyncio
    async def test_invalid_json_raises_error(self, mock_router):
        mock_router.acompletion.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="not valid json"))],
            usage=MagicMock(prompt_tokens=100, completion_tokens=50)
        )
        client = LLMClient()
        with pytest.raises(json.JSONDecodeError):
            await client.complete(
                system_prompt="System",
                user_prompt="User",
                model="test-model",
                agent="test",
            )

    @pytest.mark.asyncio
    async def test_custom_temperature(self, mock_router):
        client = LLMClient()
        await client.complete(
            system_prompt="System",
            user_prompt="User",
            model="test-model",
            agent="test",
            temperature=0.5,
        )
        call_kwargs = mock_router.acompletion.call_args.kwargs
        assert call_kwargs["temperature"] == 0.5


# =============================================================================
# Cost tracking integration tests
# =============================================================================


class TestCostTrackerIntegration:

    @pytest.fixture
    def mock_router(self):
        with patch("extractor.core.llm_client.router") as mock:
            mock.acompletion = AsyncMock()
            mock.acompletion.return_value = MagicMock(
                choices=[MagicMock(message=MagicMock(content='{}'))],
                usage=MagicMock(prompt_tokens=100, completion_tokens=50)
            )
            yield mock

    @pytest.mark.asyncio
    async def test_multiple_calls_accumulate_costs(self, mock_router):
        tracker = CostTracker()
        client = LLMClient(cost_tracker=tracker)
        for _ in range(3):
            await client.complete(
                system_prompt="S", user_prompt="U", model="test-model", agent="test",
            )
        assert tracker.call_count == 3
        assert tracker.total_prompt_tokens == 300
        assert tracker.total_completion_tokens == 150

    @pytest.mark.asyncio
    async def test_agent_breakdown_tracked(self, mock_router):
        tracker = CostTracker()
        client = LLMClient(cost_tracker=tracker)
        await client.complete(system_prompt="S", user_prompt="U", model="m", agent="explorer")
        await client.complete(system_prompt="S", user_prompt="U", model="m", agent="explorer")
        await client.complete(system_prompt="S", user_prompt="U", model="m", agent="extractor")
        by_agent = tracker.by_agent()
        assert by_agent["explorer"]["calls"] == 2
        assert by_agent["extractor"]["calls"] == 1
