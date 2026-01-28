"""CLI entrypoint for extraction pipeline."""

import argparse
import asyncio
import json
import logging
import os
import sys
import warnings
from pathlib import Path

from extractor.core.config import ChunkingConfig, DEFAULT_MODELS, SMART_MODEL, FAST_MODEL, LLM_PROVIDER, API_KEY_ENV_VAR

# Suppress LiteLLM's direct prints (must be before import)
os.environ["LITELLM_LOG"] = "ERROR"

# Suppress noisy warnings before any imports
warnings.filterwarnings("ignore", message="Pydantic serializer warnings")
warnings.filterwarnings("ignore", message="PydanticSerializationUnexpectedValue")
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
warnings.filterwarnings("ignore", message="Unclosed connection")
warnings.filterwarnings("ignore", message="Unclosed client session")
warnings.filterwarnings("ignore", category=ResourceWarning)

# Suppress noisy loggers (HTTP clients, LiteLLM internals)
for logger_name in ["httpx", "httpcore", "litellm", "LiteLLM",
                    "LiteLLM Proxy", "LiteLLM Router", "aiohttp", "asyncio"]:
    logging.getLogger(logger_name).setLevel(logging.ERROR)

from dotenv import load_dotenv  # noqa: E402 - must be after logging config

# Load environment variables
load_dotenv()

# Suppress LiteLLM's verbose output after import
try:
    import litellm
    litellm.suppress_debug_info = True
    litellm.set_verbose = False
except (ImportError, AttributeError):
    pass


async def extract(
    pdf_path: str,
    output_dir: str = "outputs",
    max_concurrent: int = 5,
    chunk_size: int = ChunkingConfig.CHUNK_SIZE,
    use_critic: bool = False,
    verbose: bool = False,
    max_funds: int | None = None,
    gleaning_passes: int = 1,
    discover_bonus: bool = False,
    smart_model: str | None = None,
) -> dict | None:
    """Run the extraction pipeline.

    Args:
        pdf_path: Path to the PDF file.
        output_dir: Directory for output files.
        max_concurrent: Max concurrent API calls.
        chunk_size: Pages per exploration chunk.
        use_critic: Whether to use critic verification.
        verbose: Verbose output.
        max_funds: Limit number of funds to extract (for testing).
        gleaning_passes: Number of extraction passes (1 = no gleaning, 2+ = gleaning).
        discover_bonus: Whether to discover PMS-relevant fields beyond schema.
        smart_model: Model to use for high-impact phases (exploration, planning).

    Returns:
        Extracted graph dict, or None on failure.
    """
    # Import here to avoid circular imports
    from extractor.orchestrator import Orchestrator

    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        print(f"Error: File not found: {pdf_path}")
        return None

    # Check API key (provider-dependent)
    if not os.environ.get(API_KEY_ENV_VAR):
        print(f"Error: {API_KEY_ENV_VAR} not set")
        if LLM_PROVIDER == "azure":
            print("For Azure, set: AZURE_API_KEY, AZURE_API_BASE, AZURE_API_VERSION")
        else:
            print("Set it in .env or export OPENROUTER_API_KEY=...")
        return None

    output_dir = Path(output_dir)
    json_dir = output_dir / "json"
    logs_dir = output_dir / "logs"
    json_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Extract model names for display (strip provider prefix)
    resolved_smart = smart_model or SMART_MODEL
    smart_display = resolved_smart.replace("openrouter/", "").replace("azure/", "")
    fast_display = DEFAULT_MODELS["reader"].replace("openrouter/", "").replace("azure/", "")

    print(f"\n{'='*50}")
    print(f"Extracting: {pdf_path.name}")
    print(f"{'='*50}")
    print(f"  Provider: {LLM_PROVIDER}")
    print(f"  Smart model: {smart_display}")
    print(f"  Fast model: {fast_display}")
    print(f"  Concurrency: {max_concurrent}")
    print(f"  Critic: {'ON' if use_critic else 'OFF'}")
    print(f"  Gleaning: {gleaning_passes} pass{'es' if gleaning_passes > 1 else ''}")
    if discover_bonus:
        print("  Schema discovery: ON")
    if max_funds:
        print(f"  Max funds: {max_funds}")
    print()

    try:
        orchestrator = Orchestrator(
            pdf_path=pdf_path,
            smart_model=smart_model,
            max_concurrent=max_concurrent,
            chunk_size=chunk_size,
            use_critic=use_critic,
            verbose=verbose,
            log_dir=logs_dir,
            max_funds=max_funds,
            gleaning_passes=gleaning_passes,
            discover_bonus=discover_bonus,
        )

        graph = await orchestrator.run()
        graph_dict = graph.model_dump()

        # Save output
        output_file = json_dir / f"{pdf_path.stem}.json"
        with open(output_file, "w") as f:
            json.dump(graph_dict, f, indent=2, ensure_ascii=False)
        print(f"\n[OUTPUT] {output_file}")

        # Print cost summary
        cost_tracker = orchestrator.context.cost_tracker
        if cost_tracker.call_count > 0:
            print(f"\n{cost_tracker.summary()}")

        return graph_dict

    except Exception as e:
        print(f"\n[ERROR] Pipeline failed: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Fund Prospectus Extraction Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run extract prospectuses/jpm_umbrella.pdf
  uv run extract --critic -g 2 prospectuses/jpm_umbrella.pdf  # quality mode
  uv run extract --max-funds 5 prospectuses/jpm_umbrella.pdf  # cheap test
        """,
    )
    parser.add_argument("pdf", help="Path to PDF file")
    parser.add_argument(
        "-o", "--output",
        default="outputs",
        help="Output directory (default: outputs)",
    )
    parser.add_argument(
        "-c", "--concurrent",
        type=int,
        default=5,
        help="Max concurrent API calls (default: 5)",
    )
    parser.add_argument(
        "-s", "--chunk-size",
        type=int,
        default=ChunkingConfig.CHUNK_SIZE,
        help=f"Pages per exploration chunk (default: {ChunkingConfig.CHUNK_SIZE})",
    )
    parser.add_argument(
        "--critic",
        action="store_true",
        help="Enable critic verification (slower but catches errors)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output with DEBUG level logging",
    )
    parser.add_argument(
        "--max-funds",
        type=int,
        default=None,
        help="Limit number of funds to extract (for testing)",
    )
    parser.add_argument(
        "-g", "--gleaning",
        type=int,
        default=1,
        metavar="N",
        help="Number of extraction passes (1=no gleaning, 2+=gleaning). Default: 1",
    )
    parser.add_argument(
        "--discover-bonus",
        action="store_true",
        help="Discover PMS-relevant fields beyond the standard schema",
    )
    parser.add_argument(
        "--smart-model",
        type=str,
        default=None,
        help=f"Model for high-impact phases (exploration, planning). Default: {SMART_MODEL}",
    )

    args = parser.parse_args()

    result = asyncio.run(extract(
        pdf_path=args.pdf,
        output_dir=args.output,
        max_concurrent=args.concurrent,
        chunk_size=args.chunk_size,
        use_critic=args.critic,
        verbose=args.verbose,
        max_funds=args.max_funds,
        gleaning_passes=args.gleaning,
        discover_bonus=args.discover_bonus,
        smart_model=args.smart_model,
    ))

    sys.exit(0 if result else 1)


if __name__ == "__main__":
    main()
