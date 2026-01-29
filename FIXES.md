# Fixes Log

Bugs found and fixed, with before/after context.

---

## 1. Planning prompt exceeds model context window on large documents

**Date**: 2026-01-28

**Symptom**: On a 757-page BlackRock prospectus, the PLANNING phase failed with:
```
OpenrouterException - This endpoint's maximum context length is 128000 tokens.
However, you requested about 175410 tokens (174162 of text input, 1248 of tool input).
```
The error fired twice (LiteLLM retried once), then the planner fell through to its `except` block and produced a minimal fallback plan. The pipeline continued but the planner contributed nothing -- all page assignments came from `enrich_plan_with_pages()` instead.

**Root cause**: `run_planner()` in `planner_agent.py` serialized all 76 exploration notes as full JSON with `json.dumps([note.model_dump() for note in exploration_notes], indent=2)`. Each `ExplorationNotes` object contains:
- `page_index`: 10 `PageContent` entries per explorer, each with `content_type`, `fund_name`, and a `description` string
- `observations`: free-form text notes
- `cross_references`: full text of every cross-reference found
- `inventory`: nested object with field presence data

For 76 explorers covering a 757-page document, this produced ~175k tokens of JSON. gpt-4o's context limit is 128k.

**Before** (`planner_agent.py:85`):
```python
notes_json = json.dumps([note.model_dump() for note in exploration_notes], indent=2)
```

**After**: Added `_compress_exploration_notes()` that:
1. Strips `description` from page index entries (planner only needs `page`, `type`, `fund`)
2. Drops external cross-references (planner only needs internal refs with target pages)
3. Drops `observations` (free text, not structured data the planner uses)
4. Drops `inventory` (redundant with page_index)
5. Uses compact JSON separators (`separators=(",",":")`) instead of `indent=2`
6. Caps table column lists at 5 entries

The planner still receives all the data it actually uses: fund names with pages, table locations with types, page-level content type index, and internal cross-reference targets. Estimated reduction: ~60-70% fewer tokens.

**Files changed**: `extractor/agents/planner_agent.py`

---

## 2. Orphaned facts from share class entity key mismatch

**Date**: 2026-01-28

**Symptom**: After fund extraction, reconciliation reported:
```
RECONCILIATION ISSUE: 70 orphaned facts (entity not found)
```
The knowledge graph had 90 recorded facts but 70 of them referenced entity keys that didn't exist in the entities dict.

**Root cause**: Entity registration and fact recording used different key formats for share classes.

In `_record_extraction_facts` (`extraction_phase.py`), share class entities were registered via:
```python
consolidator.resolve_entity(
    entity_type="share_class",
    name=sc_name,           # e.g. "A (acc) EUR"
    ...
)
```
This created an entity with key `share_class:A (acc) EUR` (because `Entity.key()` returns `f"{entity_type.value}:{self.id}"` and `id` is set to `name`).

But facts were recorded with a fund-scoped key:
```python
await consolidator.add_fact(
    entity_key=f"share_class:{fund_name}:{sc_name}",  # e.g. "share_class:Asia Pacific Bond Fund:A (acc) EUR"
    ...
)
```

The entity dict had `share_class:A (acc) EUR` but the fact referenced `share_class:Asia Pacific Bond Fund:A (acc) EUR`. No match = orphaned fact.

This also meant that table lookup facts (ISIN, fee) recorded at lines 535-536 and 568-569 with `f"share_class:{recipe.fund_name}:{sc_name}"` were always orphaned, since no entity existed with that scoped key.

**Before** (`extraction_phase.py:869-874`):
```python
# Register entity so facts aren't orphaned
consolidator.resolve_entity(
    entity_type="share_class",
    name=sc_name,
    confidence=0.8,
    properties={"fund": fund_name},
)
```

**After**:
```python
# Register entity with fund-scoped key so facts aren't orphaned.
# Facts use entity_key="share_class:{fund_name}:{sc_name}",
# so the entity id must match: "{fund_name}:{sc_name}".
scoped_name = f"{fund_name}:{sc_name}"
consolidator.resolve_entity(
    entity_type="share_class",
    name=scoped_name,
    confidence=0.8,
    properties={"fund": fund_name, "class_name": sc_name},
)
```

Now the entity key is `share_class:{fund_name}:{sc_name}`, matching the fact keys exactly. This also prevents the cross-fund conflict issue described in FUTURE_IMPROVEMENTS.md (share class "A" from Fund X no longer collides with share class "A" from Fund Y).

**Files changed**: `extractor/phases/extraction_phase.py`
