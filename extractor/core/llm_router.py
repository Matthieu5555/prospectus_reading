"""LiteLLM Router configuration for retry, fallback, and cooldown.

Replaces the Tenacity-based retry decorator with Router-level retries
that can fall back to alternative models on failure.

Supports multiple LLM providers:
- OpenRouter (default): Uses OPENROUTER_API_KEY
- Azure OpenAI: Uses AZURE_API_KEY, AZURE_API_BASE, AZURE_API_VERSION
"""

import os

from litellm import Router

from extractor.core.config import (
    LLM_PROVIDER,
    API_KEY_ENV_VAR,
    SMART_MODEL,
    FAST_MODEL,
)


def _build_openrouter_model_list() -> list[dict]:
    """Build model list for OpenRouter provider."""
    api_key_ref = f"os.environ/{API_KEY_ENV_VAR}"

    return [
        {
            "model_name": "openrouter/openai/gpt-4o-mini",
            "litellm_params": {
                "model": "openrouter/openai/gpt-4o-mini",
                "api_key": api_key_ref,
            },
        },
        {
            "model_name": "openrouter/openai/gpt-4o",
            "litellm_params": {
                "model": "openrouter/openai/gpt-4o",
                "api_key": api_key_ref,
            },
        },
        {
            "model_name": "openrouter/openai/gpt-4-turbo",
            "litellm_params": {
                "model": "openrouter/openai/gpt-4-turbo",
                "api_key": api_key_ref,
            },
        },
    ]


def _build_azure_model_list() -> list[dict]:
    """Build model list for Azure OpenAI provider.

    Azure requires:
    - AZURE_API_KEY: API key
    - AZURE_API_BASE: Endpoint URL (e.g., https://your-resource.openai.azure.com/)
    - AZURE_API_VERSION: API version (e.g., 2024-02-15-preview)

    Deployment names default to "gpt-4o" and "gpt-4o-mini" but can be
    overridden with AZURE_DEPLOYMENT_GPT_4O and AZURE_DEPLOYMENT_GPT_4O_MINI.
    """
    api_key = os.environ.get("AZURE_API_KEY", "")
    api_base = os.environ.get("AZURE_API_BASE", "")
    api_version = os.environ.get("AZURE_API_VERSION", "2024-02-15-preview")

    # Get deployment names (can be customized per Azure resource)
    smart_deployment = os.environ.get("AZURE_DEPLOYMENT_GPT_4O", "gpt-4o")
    fast_deployment = os.environ.get("AZURE_DEPLOYMENT_GPT_4O_MINI", "gpt-4o-mini")

    models = []

    # Fast model (gpt-4o-mini equivalent)
    models.append({
        "model_name": f"azure/{fast_deployment}",
        "litellm_params": {
            "model": f"azure/{fast_deployment}",
            "api_key": api_key,
            "api_base": api_base,
            "api_version": api_version,
        },
    })

    # Smart model (gpt-4o equivalent)
    models.append({
        "model_name": f"azure/{smart_deployment}",
        "litellm_params": {
            "model": f"azure/{smart_deployment}",
            "api_key": api_key,
            "api_base": api_base,
            "api_version": api_version,
        },
    })

    return models


def _build_fallbacks() -> list[dict]:
    """Build fallback configuration based on provider."""
    if LLM_PROVIDER == "azure":
        # Azure: fall back from fast to smart model
        return [{FAST_MODEL: [SMART_MODEL]}]
    else:
        # OpenRouter: fall back to different model
        return [{"openrouter/openai/gpt-4o-mini": ["openrouter/openai/gpt-4-turbo"]}]


def build_router() -> Router:
    """Build the LLM Router with retry and fallback configuration.

    The router handles:
    - Automatic retries with exponential backoff
    - Fallback from primary to secondary model on exhausted retries
    - Cooldown tracking for failed deployments
    - Rate-limit-aware routing

    Provider is determined by LLM_PROVIDER env var:
    - "openrouter" (default): Uses OpenRouter
    - "azure": Uses Azure OpenAI
    """
    if LLM_PROVIDER == "azure":
        model_list = _build_azure_model_list()
    else:
        model_list = _build_openrouter_model_list()

    return Router(
        model_list=model_list,
        num_retries=2,
        retry_after=4,
        cooldown_time=60,
        allowed_fails=2,
        fallbacks=_build_fallbacks(),
    )


router = build_router()
