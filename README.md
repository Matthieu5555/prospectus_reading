# Prospectus Entity Extraction

This project reads UCITS fund prospectuses (those dense, 200-400 page legal PDFs that describe everything about a fund) and turns them into structured, machine-readable data. It does this using a team of AI agents that work together, much like a group of analysts would: one scans the document for structure, another plans what to extract, a third reads and extracts values, and an optional fourth double-checks the work.

The output is a hierarchical JSON graph: an **umbrella fund** contains **sub-funds**, which contain **share classes**, each annotated with provenance (which page, which quote, how confident the system is).

Think of it like this: imagine handing a 300-page legal document to a new analyst and asking them to fill out a spreadsheet with every fund's ISIN, fees, investment constraints, and objectives. This system automates that process.

## Quick start

```bash
# 1. Install dependencies
uv sync

# 2. Set up your API key
cp .env.example .env
# Edit .env and paste your OpenRouter key (https://openrouter.ai/keys)

# 3. Drop a prospectus PDF into inputs/ and run
uv run python -m extractor.cli inputs/jpm_umbrella.pdf
```

Output lands in `outputs/json/`. That's it.

For best quality, enable critic verification and multiple extraction passes:

```bash
uv run python -m extractor.cli --critic -g 3 inputs/jpm_umbrella.pdf
```

A few other useful flags:

```bash
# Cheap test run: only extract the first 5 funds
uv run python -m extractor.cli --max-funds 5 inputs/jpm_umbrella.pdf

# Use a different smart model for exploration/planning (default: gpt-4o)
uv run python -m extractor.cli --smart-model openrouter/openai/gpt-4-turbo inputs/jpm_umbrella.pdf

# Discover fields beyond the standard schema
uv run python -m extractor.cli --discover-bonus inputs/jpm_umbrella.pdf
```

Run `uv run python -m extractor.cli --help` for the full option list.

## Reviewing extraction quality

After an extraction completes, you'll have a JSON file in `outputs/json/`. To actually make sense of it, the project ships with a standalone HTML report generator:

```bash
uv run visualize outputs/json/jpm_umbrella.json
```

This opens an interactive dashboard in your browser (dark-themed, searchable, with every extracted value showing its source page, confidence score, and the exact quote it was pulled from on hover). It also surfaces cost breakdowns, unresolved questions, and schema suggestions. It's the primary tool for judging whether an extraction run was any good.

This is deliberately separate from the main pipeline. Extraction and review are different activities: one is automated and expensive (API calls), the other is manual and free (reading HTML). You'll often re-review old outputs without re-running extraction.

## How it works

The system is a 9-phase pipeline. Each phase is a self-contained class that reads from shared state, does its job, and writes results back. If you want to understand the architecture, **start by reading `extractor/orchestrator.py`**, it's the most important file in the codebase and contains an ASCII diagram of the full phase graph.

Here's what happens when you run an extraction:

**Phase 1-3: Understanding the document.** The pipeline first builds a mental model of the PDF.

- **Skeleton** detects the table of contents and document structure. It uses PyMuPDF's native TOC extraction when available, falling back to regex-based heading detection. The output is a hierarchical section tree with page ranges.

- **TableScan** finds data tables (fee tables, ISIN listings). It scans every page for HTML table structures using PyMuPDF's table detection, extracts column headers and row data, then classifies tables by content (ISIN table if it contains ISIN patterns, fee table if it contains fee keywords). These pre-parsed tables become "ground truth" for later extraction.

- **ExternalRefScan** identifies references to documents that aren't the prospectus itself (KIIDs, annual reports). Currently this is a simple regex scan looking for patterns like "ISIN: See KIID" or "refer to Annual Report". This matters because the system needs to know when a value simply doesn't exist in this document. *Note: this phase is basic regex matching and could be improved with LLM-based detection for more nuanced references.*

**Phase 4-5: Figuring out what's in there.**

- **Exploration** sends LLM agents to scan the document in parallel chunks (30 pages at a time by default). Each agent reads its chunk and outputs structured notes: which funds are mentioned, on which pages, what relationships exist between them (e.g., "Fund X has share classes A, B, C"), and any notable patterns. The agents don't extract values yet, they just map the territory.

- **EntityResolution** deduplicates fund mentions across chunks. It uses fuzzy string matching (SequenceMatcher) to identify when "JPMorgan Global Bond Fund" and "Global Bond Fund" refer to the same entity. The output is a canonical fund list with aliases, preventing duplicate extraction work.

**Phase 6-7: Planning the work.**

- **Logic** aggregates patterns without calling any LLM (pure Python). It analyzes exploration notes to determine: which funds share pages (suggesting umbrella-level content), which tables are "broadcast" tables that apply to all funds (vs. fund-specific tables), and what the document's structural logic is. This is rule-based pattern detection, not AI.

- **Planning** takes the Logic output and creates an extraction recipe for each fund. The LLM planner decides: which pages to read for each fund, which fields to extract from text vs. look up in tables, and what the extraction priority order should be. The output is a per-fund work plan.

**Phase 8-9: Doing the extraction.**

- **Extraction** executes the plan. For each fund, it reads the assigned pages and calls reader agents to pull actual values (ISINs, fees, constraints, objectives). Table fields are looked up directly from pre-parsed tables (no LLM needed). Text fields go through LLM extraction with provenance tracking (source page, exact quote, confidence score). If gleaning is enabled (`-g 2+`), the agent re-reads pages looking for missed values.

- **Assembly** ties everything into the final hierarchical graph. It converts raw extraction dicts into typed Pydantic models (Umbrella → SubFund → ShareClass), runs gap-filling for umbrella-level fields that apply to all funds, inherits values where appropriate, and computes quality metrics (confidence distribution, coverage stats, unresolved questions).

There's also an optional **FailureRecovery** phase that retries failed extractions with broader search strategies (more pages, different search patterns), and the **Critic** agent (enabled with `--critic`) that independently verifies extracted values and assigns confidence scores. The critic receives parsed tables as authoritative ground truth, so if LLM extraction differs from table data, the table wins.

If that sounds like a lot, the key insight is simple: the pipeline mimics how a human analyst would work. You'd skim the table of contents, note which pages cover which funds, plan your reading, then systematically extract data. The system does the same thing, just in parallel and with structured outputs.

## Project structure

```
inputs/                  # Place prospectus PDFs here
outputs/                 # Extraction results land here (json/, logs/)
visualizer/              # HTML report generator (see "Reviewing extraction quality")
extractor/
  orchestrator.py        # Pipeline orchestrator, read this first
  cli.py                 # CLI entry point
  agents/                # The four agent types (explorer, planner, reader, critic)
  phases/                # One class per pipeline phase
  core/                  # PDF reader, search, config, logging, error handling
  prompts/               # LLM prompt templates
  pydantic_models/       # Output schemas (Umbrella → SubFund → ShareClass)
  tests/                 # pytest suite
```

## Using as a library

```python
import asyncio
from extractor import Orchestrator

async def main():
    orchestrator = Orchestrator(
        pdf_path="prospectus.pdf",
        smart_model="openrouter/openai/gpt-4o",  # For exploration/planning
        max_concurrent=5,
        use_critic=True,
        gleaning_passes=2,
    )
    graph = await orchestrator.run()
    # graph is a ProspectusGraph, Pydantic model with .model_dump()

asyncio.run(main())
```

The main types you'll interact with are `ProspectusGraph`, `Umbrella`, `SubFund`, `ShareClass`, and `ExtractedValue`. Every field that comes from the document is wrapped in `ExtractedValue`, which carries the raw value plus provenance (source page, quote, confidence, rationale).

## Running tests

```bash
uv sync --group dev
pytest
```

The test suite covers phase logic, assembly, constraint parsing, provenance tracking, the knowledge graph, and end-to-end integration. Tests mock the PDF reader and LLM responses, so no API key is needed.

## Configuration

All tunable constants live in `extractor/core/config.py`, heavily documented with explanations of what each value controls and why it was chosen. The defaults work well for typical prospectuses. The most common things to tweak are:

- **Smart model** (`--smart-model`): Model for high-impact phases (exploration, planning, umbrella extraction). Default `gpt-4o`. These phases require better reasoning since their output affects all downstream extraction.
- **Concurrency** (`-c`): How many API calls run in parallel. Default 5. Higher is faster but may hit rate limits.
- **Gleaning** (`-g`): Number of extraction passes. 1 means single-pass, 2+ means the system re-reads pages looking for values it missed. More passes = better recall, higher cost. 3 is a good maximum; beyond that you hit diminishing returns.
- **Critic** (`--critic`): Adds an independent verification step. Slower and more expensive, but catches hallucinations and assigns calibrated confidence scores. The critic receives parsed tables as "ground truth", so if there's a discrepancy between LLM extraction and table parsing, the table wins.

### Model tiers

The pipeline uses a two-tier model approach:

- **Smart model** (`gpt-4o` by default): Used for exploration, planning, and umbrella extraction, phases where reasoning quality significantly impacts downstream results. Override with `--smart-model`.
- **Fast model** (`gpt-4o-mini` by default): Used for per-fund extraction and critic verification, high-volume operations where cost matters more than reasoning depth.

This balances quality and cost: smart reasoning where it counts, fast/cheap execution for the bulk of the work.

### LLM Providers

The system supports multiple LLM providers via [LiteLLM](https://docs.litellm.ai/). Set the `LLM_PROVIDER` environment variable to switch.

#### OpenRouter (default)

```bash
# .env
LLM_PROVIDER=openrouter
OPENROUTER_API_KEY=sk-or-...
```

OpenRouter is an API gateway that provides access to multiple model providers (OpenAI, Anthropic, etc.) through a single API.

#### Azure OpenAI

For enterprise deployments on Azure:

```bash
# .env
LLM_PROVIDER=azure
AZURE_API_KEY=your-azure-api-key
AZURE_API_BASE=https://your-resource.openai.azure.com/
AZURE_API_VERSION=2024-02-15-preview

# Optional: customize deployment names (defaults to gpt-4o and gpt-4o-mini)
AZURE_DEPLOYMENT_GPT_4O=my-gpt4o-deployment
AZURE_DEPLOYMENT_GPT_4O_MINI=my-gpt4o-mini-deployment
```

The model names automatically adapt to the provider. When using Azure, the `--smart-model` flag accepts Azure deployment names:

```bash
# Azure example
uv run python -m extractor.cli --smart-model azure/my-custom-gpt4 inputs/prospectus.pdf
```

### Programmatic configuration

```python
orchestrator = Orchestrator(
    pdf_path="prospectus.pdf",
    smart_model="openrouter/openai/gpt-4o",       # High-impact phases (default)
    reader_model="openrouter/openai/gpt-4o-mini", # Per-fund extraction (default)
)
```
