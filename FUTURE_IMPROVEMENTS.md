# Future Improvements

Known issues and optimizations that would make the pipeline faster, cheaper, or more accurate.

## Conflict detection compares provenance dicts instead of values

**The problem**: The knowledge consolidator compares entire provenance dicts rather than extracting values first. This causes two issues:

1. **Identical values trigger conflicts**: When two extractions find `management_fee: 0.60%`, but with different source pages or quotes, the system sees them as different and triggers conflict resolution:
   ```
   Conflict resolved for share_class:C.management_fee: '0.60%' vs '0.60%' → kept '0.60%' (llm_verified)
   ```
   This wastes an LLM call to "resolve" a non-conflict.

2. **Unreadable logs**: When values ARE different, the logs dump full provenance dicts:
   ```
   Conflict resolved for share_class:I2.exit_fee: '{'value': '0.40%', 'source_page': 20, ...}' vs '{'value': '0.40%', ...}'
   ```
   Instead of the readable: `'0.40%' vs '0.40%'`

**Why it matters**:
1. **Wasted API cost**: Resolving `'0.60%' vs '0.60%'` burns an LLM call for nothing.
2. **Log noise**: Hard to spot real conflicts when identical values flood the output.
3. **Wrong resolution strategy**: "Recency" is used for identical values, which is meaningless.

**The fix**: In `knowledge_consolidator.py`, extract raw values before comparison:
```python
# Before
if new_value != existing_value:
    # trigger conflict resolution

# After
new_raw = get_raw_value(new_value)
existing_raw = get_raw_value(existing_value)
if new_raw != existing_raw:
    # trigger conflict resolution
```

Also fix the logging to show just values, not full dicts.

**Estimated scope**: ~1 hour. Localized to `knowledge_consolidator.py`.

## Per-fund table count is always zero despite tables existing

**The problem**: Fund extraction logs show `(T:0)` for every fund:
```
[1/24] Global Core Equity Fund (T:0) (68.7s)
[2/24] Europe Select Equity Fund (T:0) (375.4s)
```

But TableScan found 85 tables, and broadcast table parsing works (`Using pre-scanned fee: 243 rows`). So tables exist and get used, but the per-fund count is wrong.

**Why it matters**:
1. **Misleading diagnostics**: When debugging extraction issues, `T:0` suggests no tables were found for a fund, but that's not true.
2. **Possible missed tables**: If this count is supposed to reflect fund-specific tables (not broadcast), maybe fund-specific table detection is actually broken.

**Investigation needed**: Check what `T:0` is supposed to measure:
- If it's "tables in this fund's dedicated pages" - why is it always 0 when fee tables appear on fund pages (p6, p8, etc.)?
- If it's "fund-specific tables (not broadcast)" - is that distinction being made correctly?
- Is it a display bug where the count just isn't being populated?

Look at `extraction_phase.py` where the `(T:X)` logging happens and trace where that count comes from.

**Estimated scope**: Unknown until investigated. Could be a 5-minute logging fix or reveal a deeper table association bug.

## LLM JSON parsing failures aren't handled gracefully

**The problem**: When the LLM returns malformed JSON, extraction fails completely for that fund:
```
ERROR: [Extraction] [Global Income ESG Fund] Recipe extraction failed: Unterminated string starting at: line 1804 column 22 (char 64944)
```

This happens when:
1. Response hits max tokens and gets truncated mid-string
2. LLM generates unescaped quotes or newlines in string values
3. LLM "forgets" to close a bracket deep in a nested structure

At 64k+ characters and 1800 lines, this response was massive - likely the LLM tried to extract too much at once.

**Why it matters**:
1. **Total fund loss**: One malformed response = zero data for that fund. No partial recovery.
2. **Silent failures**: The pipeline continues, so you might not notice a fund is missing until reviewing output.
3. **Wasted cost**: You paid for the LLM call but got nothing usable.

**Possible fixes** (in order of complexity):
1. **JSON repair**: Try common fixes (close open brackets, escape quotes) before giving up
2. **Retry with smaller scope**: If response was truncated, retry asking for fewer fields
3. **Chunked extraction**: For funds with many share classes, extract in batches instead of all at once
4. **Streaming validation**: Parse JSON as it streams, catch errors earlier

**Estimated scope**: 2-4 hours depending on approach. JSON repair is quick; chunked extraction is more involved.

## Cost tracking shows $0.0000 despite millions of tokens

**The problem**: The cost summary reports zero cost even with substantial token usage:
```
Total tokens: 2,607,331
  - Prompt: 2,151,041
  - Completion: 456,290
Total cost: $0.0000

By agent:
  conflict_resolver: 147 calls, 210,626 tokens, $0.0000
  extractor: 71 calls, 639,000 tokens, $0.0000
  ...
```

2.6 million tokens at GPT-4o-mini rates (~$0.15/1M input, ~$0.60/1M output) should be roughly $0.60. At GPT-4o rates it would be significantly more.

**Why it matters**:
1. **No cost visibility**: Can't evaluate whether optimizations (fewer conflict resolution calls, smaller chunks) actually save money.
2. **Budget planning impossible**: Can't estimate cost for batch processing multiple prospectuses.
3. **Model comparison blocked**: Can't compare cost/quality tradeoffs between models.

**Investigation needed**: Check `cost_tracker.py` for:
- Whether pricing tables are populated for the models being used
- Whether OpenRouter returns cost info differently than direct OpenAI
- Whether the cost calculation is happening but not being summed correctly

**Estimated scope**: ~1-2 hours. Likely a missing price lookup or OpenRouter-specific response parsing issue.

## Share class entity keys need fund scoping

**The problem**: Share class entity keys are currently just `share_class:{class_name}`, like `share_class:C` or `share_class:A`. But share class names are generic letters that repeat across every fund. When Fund 1 extracts "Class C with currency USD" and Fund 2 extracts "Class C with currency EUR", the knowledge consolidator sees these as conflicting facts about the same entity and triggers LLM-based conflict resolution.

You can see this in extraction logs:
```
Conflict resolved for share_class:C.currency: 'USD' vs 'EUR' → kept 'USD' (llm_verified)
Conflict resolved for share_class:A.currency: 'EUR' vs 'USD' → kept 'EUR' (llm_verified)
```

These aren't conflicts at all - they're different share classes that happen to share a letter. The system is burning LLM calls verifying "conflicts" that don't exist.

**Why it matters**:
1. **Wasted API cost**: Each "conflict" triggers an LLM verification call. A 24-fund prospectus with 7 share classes each could generate 100+ spurious conflict checks.
2. **Potential for wrong resolutions**: If the LLM picks the wrong value during conflict resolution, you get silent data corruption. Class C of Fund X gets Fund Y's currency.
3. **Noisy logs**: Hard to spot real conflicts when they're buried in false positives.

**The fix**: Scope entity keys to their parent fund. Change `share_class:{class_name}` to `share_class:{fund_name}:{class_name}`.

The relevant code is in `extraction_phase.py` around the `_record_extraction_facts` method and anywhere else that constructs share class entity keys. Grep for `share_class:` to find all the places.

This is a straightforward change but touches multiple files, so it needs careful testing. The entity key format is also used in graph queries and the visualizer, so those need updating too.

**Estimated scope**: ~2 hours. Mechanical find-and-replace plus test updates.

## Constraint extraction could use the skeleton more aggressively

**The problem**: The two-pass umbrella extraction finds constraint sections via TOC pattern matching (looking for "Investment Restrictions", "Risk Management", etc. in section titles). But it only uses this for the umbrella. Per-fund constraint extraction still falls back to keyword search across the whole document.

Many prospectuses have a common structure: general constraints in a dedicated chapter (pages 59-70), then fund-specific overrides in each fund's section. The current approach reads the general chapter for umbrella constraints but doesn't use that knowledge to skip re-searching those same pages for each fund.

**Why it matters**:
1. **Redundant searches**: If 24 funds each search for "leverage limit" and all find the same umbrella-level constraint on page 65, that's 24 searches finding the same thing.
2. **Missing inheritance**: Fund-specific constraints should override umbrella defaults, but the current approach doesn't explicitly model this hierarchy.

**The fix**: After umbrella constraint extraction, mark those constraints as "umbrella-level defaults" in the knowledge graph. During per-fund extraction, check if a constraint was already found at umbrella level before searching. Only search if the fund's dedicated pages mention constraint overrides.

This requires changes to:
- `extraction_phase.py`: Record umbrella constraints with an "applies_to: all_funds" marker
- `reader_agent.py`: Check knowledge graph before constraint search
- Possibly the assembly phase to handle inheritance explicitly

**Estimated scope**: ~4-6 hours. Requires careful thought about inheritance semantics.

## Critic verification could batch share classes

**The problem**: The critic currently verifies each fund's extraction as a unit. But share class verification is repetitive - checking that ISINs are 12 characters, fees are percentages, currencies are ISO codes. These checks don't need the full LLM reasoning that fund-level verification needs.

**Why it matters**: Critic mode roughly doubles extraction time. Much of that is spent on mechanical share class checks that could be rule-based.

**The fix**: Split critic into two tiers:
1. **Rule-based validation** for share classes: regex for ISINs, percentage parsing for fees, currency code lookup. No LLM needed.
2. **LLM verification** for fund-level fields: investment objectives, constraints, benchmark descriptions. These need reasoning.

Only escalate to LLM verification when rule-based checks find anomalies (e.g., ISIN that doesn't match the fund's domicile pattern).

**Estimated scope**: ~3-4 hours. Need to define the rule set and figure out which fields are "mechanical" vs "semantic".

## Table parsing could be smarter about multi-page tables

**The problem**: When a fee table spans pages 232-236, the current approach concatenates rows from each page. But if column headers only appear on page 232, the rows from pages 233-236 might get misaligned if the table structure isn't perfectly consistent.

This hasn't caused visible bugs yet, but it's fragile. Some prospectuses have tables where later pages drop columns or add footnote rows.

**Why it matters**: Table data is treated as ground truth for the critic. If table parsing is wrong, the critic will "correct" accurate LLM extractions to match incorrect table values.

**The fix**: After concatenating multi-page tables, run a consistency check:
1. Verify all rows have the same column count
2. Check for header-like rows in the middle (indicating a repeated header)
3. Flag tables with inconsistent structure for manual review rather than using them as ground truth

**Estimated scope**: ~2-3 hours. Mostly defensive checks in `table_extraction.py`.

## Exploration could detect "this fund has no dedicated section"

**The problem**: Some prospectuses list funds in a summary table but don't give each fund its own section. The current pipeline assumes every fund has dedicated pages, leading to empty or wrong page assignments when this assumption fails.

**Why it matters**: If Fund X has no dedicated section, the planner assigns it pages from an adjacent fund. Extraction then pulls data for the wrong fund. This is a silent failure - you get plausible-looking but incorrect data.

**The fix**: During exploration, explicitly detect funds that appear only in summary tables vs. funds with dedicated narrative sections. Mark "summary-only" funds in the knowledge graph. During planning, handle these differently - maybe extract only from tables, or flag for manual review.

**Estimated scope**: ~4 hours. Requires changes to explorer prompts and planning logic.

## Text extraction could use pymupdf-layout for cleaner LLM input

**The problem**: The pipeline extracts text with basic `page.get_text()` in `pdf_reader.py`. This returns raw text blobs with no structure, no filtering, and garbled reading order for multi-column layouts. Every page includes repetitive headers ("UCITS Prospectus - Page 42 of 250"), footers, and legal disclaimers that waste tokens and add noise to LLM prompts. Financial prospectuses are particularly affected because they commonly have multi-column fee tables where text extraction jumbles columns together, repetitive page headers and footers on every page, and complex nested section structures that become flat text.

**Why it matters**: Headers and footers repeat on every page. A 300-page document might waste 10-15% of tokens on "Page X of Y" and boilerplate. When exploration agents receive garbled multi-column text, they may misidentify fund boundaries or miss table structures. LLMs also parse structured markdown (with headings, proper lists, preserved tables) better than raw text blobs.

**The fix**: The `pymupdf-layout` package is already in dependencies but unused. It provides automatic header/footer detection and filtering, proper reading order for multi-column layouts, markdown output via `pymupdf4llm` with heading detection and table preservation, and OCR fallback for scanned pages. To enable it, import `pymupdf.layout` before using layout features in `pdf_reader.py`, then add a markdown extraction method that calls `pymupdf4llm.to_markdown()`. Update `exploration_phase.py` and `reader_agent.py` to use the markdown version. The trade-off is processing overhead from layout analysis per page, and `pymupdf-layout` pulls in extra dependencies (networkx, onnxruntime, numpy). There may also be edge cases with unusual layouts that need handling.

**Estimated scope**: ~3-4 hours. Changes to `pdf_reader.py`, `exploration_phase.py`, `reader_agent.py`. Should A/B test on a few prospectuses to measure actual improvement before full rollout.
