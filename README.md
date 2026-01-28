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

After an extraction completes, you'll have a JSON file in `outputs/json/`. The natural question is: did it actually work? The project ships with a visualizer that answers this:

```bash
uv run visualize outputs/json/jpm_umbrella.json
```

This opens a searchable dashboard where you can click any extracted value and see exactly where it came from: the page number, the confidence score, and the literal quote from the PDF. It's how you spot-check whether the system found real data or hallucinated something plausible-looking. The dashboard also shows cost breakdowns (how much you spent on API calls), unresolved questions (fields the system couldn't find), and schema suggestions (data patterns it noticed but couldn't categorize).

Why is this separate from the main pipeline? Because extraction and review are different activities. Extraction is automated and costs money (every LLM call has a price). Review is manual and free (you're just reading HTML). You'll often re-review old outputs days later without wanting to re-run extraction. Keeping them separate means you can iterate on the visualizer without touching the extraction code, and vice versa.

## How it works

The system is a 9-phase pipeline. Each phase is a self-contained class that reads from shared state, does its job, and writes results back. If you want to understand the architecture, **start by reading `extractor/orchestrator.py`**—it's the most important file in the codebase and contains an ASCII diagram of the full phase graph.

### Why phases and shared state?

You might wonder: why not just have one big function that does everything? Two reasons. First, testability—each phase can be tested in isolation with mocked inputs. Second, memory—a 400-page prospectus generates a lot of intermediate data during exploration. Shared state lets each phase drop its working data after handing off results, keeping memory bounded. (This could have been done with generators or streaming, but explicit phases are easier to debug. You can inspect the state between any two phases without reconstructing call chains.)

There's also a distinction between **phases** and **agents**. Phases are orchestration units that run in sequence and manage the pipeline's state. Agents are the LLM-calling workers that phases spawn—they do the actual reading and extraction. This split means you can test an agent (e.g., "does the reader correctly extract ISINs from this text?") without running the whole pipeline.

Here's what happens when you run an extraction:

**Phase 1-3: Understanding the document.** The pipeline first builds a mental model of the PDF.

- **Skeleton** detects the table of contents and document structure. It uses PyMuPDF's native TOC extraction when available, falling back to regex-based heading detection. The output is a hierarchical section tree with page ranges. *Limitation: relies on PDF having embedded TOC metadata. If the PDF was created without bookmarks (common with scanned documents or poorly-generated PDFs), the fallback regex detection is much less reliable and may miss sections entirely.*

- **TableScan** finds data tables (fee tables, ISIN listings). It scans every page for HTML table structures using PyMuPDF's table detection, extracts column headers and row data, then classifies tables by content (ISIN table if it contains ISIN patterns, fee table if it contains fee keywords). These pre-parsed tables become "ground truth" for later extraction. *Limitation: depends on tables being actual HTML/PDF table structures. Tables created as plain text with spacing, or tables embedded as images, will not be detected.*

- **ExternalRefScan** identifies when field values aren't in the prospectus but live in external documents (KIIDs, annual reports). It uses two detection methods: (1) table cell detection, which checks pre-scanned tables for cells containing "See KIID" instead of actual values (very accurate since table structure is unambiguous), and (2) proximity-based text matching, which finds sentences where a field keyword, referral phrase, and external document name appear together (e.g., "ISIN codes can be found in the KIID"). This matters because the system needs to know when a value simply doesn't exist in this document, preventing wasted LLM calls. *Limitation: only catches explicit references with known keywords. Subtle references like "ISIN codes are published separately" without naming the document may be missed.*

**Phase 4-5: Figuring out what's in there.**

- **Exploration** sends LLM agents to scan the document in parallel chunks (30 pages at a time by default). Each agent reads its chunk and outputs structured notes: which funds are mentioned, on which pages, what relationships exist between them (e.g., "Fund X has share classes A, B, C"), and any notable patterns. The agents don't extract values yet, they just map the territory. *Limitation: if a fund is mentioned only once in passing or with an unusual name variant, it may be missed. Chunk boundaries can also split fund descriptions awkwardly.*

- **EntityResolution** deduplicates fund mentions across chunks. It uses fuzzy string matching (SequenceMatcher) to identify when "JPMorgan Global Bond Fund" and "Global Bond Fund" refer to the same entity. The output is a canonical fund list with aliases, preventing duplicate extraction work. *Limitation: fuzzy matching can both under-merge (miss that "Global Bond" and "JPM Global Bond Fund" are the same) and over-merge (incorrectly merge "Global Bond Fund" and "Global Equity Fund" if threshold is too loose).*

**Phase 6-7: Planning the work.**

- **Logic** aggregates patterns without calling any LLM (pure Python). It analyzes exploration notes to determine: which funds share pages (suggesting umbrella-level content), which tables are "broadcast" tables that apply to all funds (vs. fund-specific tables), and what the document's structural logic is. This is rule-based pattern detection, not AI. *Limitation: heuristics assume common prospectus layouts. Unusual document structures (e.g., funds organized by share class rather than by fund) may confuse the logic.*

- **Planning** takes the Logic output and creates an extraction recipe for each fund. The LLM planner decides: which pages to read for each fund, which fields to extract from text vs. look up in tables, and what the extraction priority order should be. The output is a per-fund work plan. *Limitation: planner quality depends heavily on exploration quality. If exploration missed pages or misidentified fund boundaries, the plan will be wrong.*

**Phase 8-9: Doing the extraction.**

- **Extraction** executes the plan. For each fund, it reads the assigned pages and calls reader agents to pull actual values (ISINs, fees, constraints, objectives). Table fields are looked up directly from pre-parsed tables (no LLM needed). Text fields go through LLM extraction with provenance tracking (source page, exact quote, confidence score). If gleaning is enabled (`-g 2+`), the agent re-reads pages looking for missed values. *Limitation: LLM extraction can hallucinate values that look plausible but don't exist in the source. Table lookups require exact column header matching, which fails if headers are non-standard.*

- **Assembly** ties everything into the final hierarchical graph. It converts raw extraction dicts into typed Pydantic models (Umbrella → SubFund → ShareClass), runs gap-filling for umbrella-level fields that apply to all funds, inherits values where appropriate, and computes quality metrics (confidence distribution, coverage stats, unresolved questions). *Limitation: inheritance logic assumes standard umbrella/subfund/shareclass hierarchy. Non-standard structures (e.g., multiple umbrellas in one prospectus) may cause incorrect inheritance.*

There's also an optional **FailureRecovery** phase that retries failed extractions with broader search strategies (more pages, different search patterns), and the **Critic** agent (enabled with `--critic`) that independently verifies extracted values and assigns confidence scores. The critic receives parsed tables as authoritative ground truth, so if LLM extraction differs from table data, the table wins.

If that sounds like a lot, the key insight is simple: the pipeline mimics how a human analyst would work. You'd skim the table of contents, note which pages cover which funds, plan your reading, then systematically extract data. The system does the same thing, just in parallel and with structured outputs.

## When things go wrong

Extraction failed or produced garbage? Here's how to debug common problems:

**Zero funds extracted**: Check whether the TOC was detected. Run with `--verbose` and look for the Skeleton phase output. If it says "No native TOC found" and falls back to regex detection, your PDF probably lacks bookmarks. Try a different PDF or check if the PDF was created by scanning (scanned PDFs often lack structural metadata).

**Funds detected but values are wrong**: Open the visualizer and click on a suspicious value. Check the "source quote"—this is the literal text the LLM saw. If the quote is correct but the extraction is wrong, the LLM hallucinated. If the quote is garbage (garbled text, wrong section), the PDF text extraction failed. Multi-column layouts and unusual fonts often cause this.

**"External reference" for fields that exist in the document**: The ExternalRefScan phase might have false-positived. Check the logs for "Text proximity: field_name -> KIID" messages. If it's matching something like "ISIN codes are assigned by..." as an external reference, the keyword detection is too aggressive. You can temporarily disable external ref detection by editing `external_ref_scan_phase.py`.

**Token overflow errors**: Your PDF is too dense for the chunk size. Run with a smaller chunk size: `--chunk-size 15`. Or check if specific pages have unusually dense content (tables with hundreds of rows)—the adaptive chunking should handle this, but edge cases exist.

**Extraction is slow**: Check your concurrency setting. Default is 5 parallel API calls. If you have headroom on rate limits, try `-c 10`. Also check if the critic is enabled (`--critic`)—it doubles the LLM calls. For test runs, disable it.

**Costs are higher than expected**: Run the visualizer and check the cost breakdown. Exploration and planning use the smart model by default when `--smart-model` is set. Per-fund extraction uses the fast model. If you're extracting 100 funds, most cost comes from volume, not model choice.

## Known limitations

The pipeline works well on standard UCITS prospectuses but has known weaknesses:

**PDF quality dependencies:**
- **TOC metadata required**: Skeleton phase relies on embedded PDF bookmarks. PDFs without TOC metadata (scanned documents, poorly-generated exports) fall back to regex heading detection, which is unreliable.
- **Real table structures required**: TableScan only detects actual PDF table objects. Tables created as formatted text, or tables embedded as images, will not be parsed.
- **Text extraction quality**: PyMuPDF text extraction can struggle with multi-column layouts, unusual fonts, or PDFs with complex formatting.

**Structural assumptions:**
- **Standard hierarchy expected**: The pipeline assumes umbrella → subfund → share class structure. Non-standard structures (multiple umbrellas, funds without share classes) may cause incorrect data inheritance.
- **One prospectus per PDF**: Multiple prospectuses in a single PDF are not supported.
- **European/UCITS focus**: Field schemas and search patterns are tuned for UCITS prospectuses. US mutual fund prospectuses have different terminology and structure.

**Detection gaps:**
- **Subtle external references**: ExternalRefScan catches explicit "See KIID" patterns but misses subtle references like "published separately" without naming the document.
- **Fund name variations**: EntityResolution fuzzy matching can miss unusual name variants or incorrectly merge similarly-named but distinct funds.
- **LLM hallucinations**: Despite critic verification, LLMs can confidently fabricate plausible-looking values. Always spot-check high-stakes extractions.

**Performance considerations:**
- **Large documents**: Documents over 500 pages may hit token limits or memory constraints.
- **API rate limits**: High concurrency with cheap models may trigger rate limiting.

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

The separation between `agents/` and `phases/` might seem redundant—both "do work," right? But they serve different purposes. Phases manage sequencing and state: "first we explore, then we plan, then we extract." Agents do the actual LLM work: "read these pages and find the ISINs." This separation lets you unit-test agents with fake PDF text without running the whole pipeline, and it lets you add new phases (e.g., a "summarize" phase at the end) without touching agent code.

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

- **Smart model** (`gpt-4o` by default): Used for exploration, planning, and umbrella extraction—phases where reasoning quality significantly impacts downstream results. Override with `--smart-model`.
- **Fast model** (`gpt-4o-mini` by default): Used for per-fund extraction and critic verification—high-volume operations where cost matters more than reasoning depth.

Why not just use the smart model everywhere? Because errors in early phases cascade. If exploration misses a fund, planning won't include it, and extraction will never look for it. Those early phases justify the higher cost. But per-fund extraction is repetitive work—reading pages, finding ISINs, parsing fees—that benefits more from parallelism than deep reasoning. Running 50 parallel fast-model calls beats 5 sequential smart-model calls for this workload. (This could be made configurable per-phase, but the two-tier split covers 90% of use cases without complexity.)

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
