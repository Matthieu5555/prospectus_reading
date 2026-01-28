"""Render functions for prospectus extraction visualization.

Converts extracted data structures (ExtractedValue, Constraint, ShareClass, etc.)
into HTML fragments for the dashboard.
"""

from html import escape
from typing import Any

from extractor.core.config import ConfidenceThresholds


def is_extracted_value(obj: Any) -> bool:
    """Check if object looks like an ExtractedValue."""
    if not isinstance(obj, dict):
        return False
    return "value" in obj and any(k in obj for k in ["source_page", "source_quote", "confidence", "rationale"])


def render_extracted_value(extracted_value: dict, field_name: str = "") -> str:
    """Render an ExtractedValue with provenance tooltip."""
    value = extracted_value.get("value")
    confidence = extracted_value.get("confidence", 1.0)
    source_page = extracted_value.get("source_page")
    source_quote = extracted_value.get("source_quote", "")
    rationale = extracted_value.get("rationale", "")
    not_found = extracted_value.get("not_found_reason")
    external_ref = extracted_value.get("external_reference")
    is_discovered = extracted_value.get("is_discovered", False)
    extraction_pass = extracted_value.get("extraction_pass", 1)
    source_type = extracted_value.get("source_type")

    # Color-code by type so users can quickly scan for specific data types
    # (red for missing, green for strings, orange for numbers, etc.)
    if value is None or not_found:
        display = '<span class="not-found">NOT FOUND</span>'
        if not_found:
            display += f' <span class="not-found-reason">({not_found})</span>'
    elif isinstance(value, bool):
        display = f'<span class="bool-{"true" if value else "false"}">{value}</span>'
    elif isinstance(value, (int, float)):
        display = f'<span class="number">{value}</span>'
    elif isinstance(value, list):
        if len(value) == 0:
            display = '<span class="empty">[]</span>'
        else:
            items = [escape(str(v)) for v in value[:5]]
            if len(value) > 5:
                items.append(f"... +{len(value)-5} more")
            display = f'<span class="list">[{", ".join(items)}]</span>'
    else:
        display = f'<span class="string">{escape(str(value))}</span>'

    # Tooltips show provenance on hover - users need to verify extractions
    # against source document, so we show page, quote, and confidence
    tooltip_lines = []
    if is_discovered:
        tooltip_lines.append('<div class="tip-row"><span class="tip-label tip-discovered">⚡ DISCOVERED FIELD</span></div>')
    if source_page is not None:
        tooltip_lines.append(f'<div class="tip-row"><span class="tip-label">Page:</span> {source_page}</div>')
    if confidence < 1.0:
        tooltip_lines.append(f'<div class="tip-row"><span class="tip-label">Confidence:</span> {confidence:.0%}</div>')
    if rationale:
        tooltip_lines.append(f'<div class="tip-row"><span class="tip-label">Rationale:</span> {escape(rationale)}</div>')
    if source_type:
        tooltip_lines.append(f'<div class="tip-row"><span class="tip-label">Source:</span> {escape(source_type)}</div>')
    if extraction_pass and extraction_pass > 1:
        tooltip_lines.append(f'<div class="tip-row"><span class="tip-label">Pass:</span> {extraction_pass}</div>')
    if source_quote:
        quote_preview = source_quote[:300] + "..." if len(source_quote) > 300 else source_quote
        tooltip_lines.append(f'<div class="tip-row tip-quote"><span class="tip-label">Quote:</span> "{escape(quote_preview)}"</div>')
    if external_ref:
        tooltip_lines.append(f'<div class="tip-row"><span class="tip-label">External:</span> {escape(external_ref)}</div>')

    confidence_class = (
        "high" if confidence >= ConfidenceThresholds.HIGH
        else "medium" if confidence >= ConfidenceThresholds.LOW
        else "low"
    )

    if tooltip_lines:
        tooltip_content = "".join(tooltip_lines)
        return f'''<span class="ev ev-{confidence_class}">
            {display}
            <span class="tooltip">{tooltip_content}</span>
        </span>'''
    return display


def render_value(obj: Any) -> str:
    """Recursively render any value."""
    if is_extracted_value(obj):
        return render_extracted_value(obj)

    if obj is None:
        return '<span class="null">null</span>'
    if isinstance(obj, bool):
        return f'<span class="bool-{"true" if obj else "false"}">{obj}</span>'
    if isinstance(obj, (int, float)):
        return f'<span class="number">{obj}</span>'
    if isinstance(obj, str):
        if len(obj) > 100:
            return f'<span class="string long" title="{escape(obj)}">{escape(obj[:100])}...</span>'
        return f'<span class="string">{escape(obj)}</span>'
    if isinstance(obj, list):
        if len(obj) == 0:
            return '<span class="empty">[]</span>'
        return f'<span class="list-count">[{len(obj)} items]</span>'
    if isinstance(obj, dict):
        return '<span class="dict-count">{' + str(len(obj)) + ' fields}</span>'
    return escape(str(obj))


def render_constraint(c: dict, idx: int) -> str:
    """Render a single constraint."""
    constraint_type = c.get("constraint_type", "Unknown")
    binding = c.get("binding_status", "")
    is_predefined = c.get("is_predefined", True)  # Default True for old data

    binding_class = binding.lower() if binding else "unknown"

    rows = []
    for key, val in c.items():
        if key in ("constraint_type", "binding_status", "is_predefined"):
            continue
        rows.append(f'''
            <tr>
                <td class="field-name">{escape(key)}</td>
                <td class="field-value">{render_value(val)}</td>
            </tr>
        ''')

    # Show discovered badge if not predefined
    discovered_badge = '' if is_predefined else '<span class="discovered-badge">DISCOVERED</span>'

    return f'''
    <div class="constraint constraint-{binding_class}{"" if is_predefined else " constraint-discovered"}">
        <div class="constraint-header">
            <span class="constraint-type">{escape(constraint_type)}</span>
            {discovered_badge}
            {f'<span class="binding-badge binding-{binding_class}">{binding}</span>' if binding else ''}
        </div>
        <table class="constraint-details">
            {''.join(rows)}
        </table>
    </div>
    '''


def render_share_class(sc: dict, idx: int) -> str:
    """Render a share class."""
    # Handle both "name" and "class_name" for compatibility
    name = sc.get("name") or sc.get("class_name", {})
    if is_extracted_value(name):
        name_display = name.get("value", f"Share Class {idx+1}")
    else:
        name_display = name or f"Share Class {idx+1}"

    # Core fields to show prominently
    core_fields = ["isin", "currency", "distribution_policy", "minimum_investment",
                   "management_fee", "ongoing_charges", "performance_fee"]

    core_rows = []
    other_rows = []

    for key, val in sc.items():
        if key in ("name", "class_name"):
            continue
        row = f'''
            <tr>
                <td class="field-name">{escape(key.replace('_', ' ').title())}</td>
                <td class="field-value">{render_value(val)}</td>
            </tr>
        '''
        if key in core_fields:
            core_rows.append(row)
        else:
            other_rows.append(row)

    other_section = ""
    if other_rows:
        other_section = f'''
        <details class="other-fields">
            <summary>Other fields ({len(other_rows)})</summary>
            <table class="field-table">
                {''.join(other_rows)}
            </table>
        </details>
        '''

    return f'''
    <div class="share-class">
        <div class="share-class-name">{escape(str(name_display))}</div>
        <table class="field-table core-fields">
            {''.join(core_rows)}
        </table>
        {other_section}
    </div>
    '''


def render_fund(fund: dict, idx: int) -> str:
    """Render a single sub-fund."""
    # Handle both "name" and "fund_name" keys
    name = fund.get("name") or fund.get("fund_name", {})
    if is_extracted_value(name):
        name_display = name.get("value", f"Fund {idx+1}")
    else:
        name_display = name or f"Fund {idx+1}"

    # Extract share classes and constraints
    share_classes = fund.get("share_classes", [])
    constraints = fund.get("constraints", [])

    # Core fund fields
    skip_fields = {"name", "fund_name", "share_classes", "constraints"}
    fund_fields = []
    for key, val in fund.items():
        if key not in skip_fields:
            fund_fields.append(f'''
                <tr>
                    <td class="field-name">{escape(key.replace('_', ' ').title())}</td>
                    <td class="field-value">{render_value(val)}</td>
                </tr>
            ''')

    # Render share classes
    share_class_html = ""
    if share_classes:
        sc_items = [render_share_class(sc, i) for i, sc in enumerate(share_classes)]
        share_class_html = f'''
        <details class="share-classes-section" open>
            <summary>Share Classes ({len(share_classes)})</summary>
            <div class="share-classes-grid">
                {''.join(sc_items)}
            </div>
        </details>
        '''

    # Render constraints
    constraints_html = ""
    if constraints:
        c_items = [render_constraint(c, i) for i, c in enumerate(constraints)]
        constraints_html = f'''
        <details class="constraints-section">
            <summary>Constraints ({len(constraints)})</summary>
            <div class="constraints-list">
                {''.join(c_items)}
            </div>
        </details>
        '''

    return f'''
    <details class="fund" data-name="{escape(str(name_display).lower())}">
        <summary class="fund-header">
            <span class="fund-name">{escape(str(name_display))}</span>
            <span class="fund-meta">
                {f'<span class="badge">{len(share_classes)} classes</span>' if share_classes else ''}
                {f'<span class="badge">{len(constraints)} constraints</span>' if constraints else ''}
            </span>
        </summary>
        <div class="fund-content">
            <table class="field-table fund-fields">
                {''.join(fund_fields)}
            </table>
            {share_class_html}
            {constraints_html}
        </div>
    </details>
    '''


def render_umbrella(data: dict) -> str:
    """Render umbrella information."""
    umbrella = data.get("umbrella", {})
    if not umbrella:
        return ""

    rows = []
    for key, val in umbrella.items():
        if key == "sub_funds":
            continue
        rows.append(f'''
            <tr>
                <td class="field-name">{escape(key.replace('_', ' ').title())}</td>
                <td class="field-value">{render_value(val)}</td>
            </tr>
        ''')

    return f'''
    <div class="umbrella-info">
        <h2>Umbrella Fund</h2>
        <table class="field-table">
            {''.join(rows)}
        </table>
    </div>
    '''


def render_stats(data: dict) -> str:
    """Render quick stats bar."""
    umbrella = data.get("umbrella", {})
    sub_funds = data.get("sub_funds") or umbrella.get("sub_funds", [])
    metadata = data.get("metadata", {})

    total_funds = len(sub_funds)
    total_share_classes = sum(len(f.get("share_classes", [])) for f in sub_funds)
    total_constraints = sum(len(f.get("constraints", [])) for f in sub_funds)

    # Count NOT_FOUND fields
    not_found_count = metadata.get("not_found_count", 0)
    total_pages = metadata.get("total_pages", "?")

    stats = [
        ("Total Funds", total_funds, ""),
        ("Share Classes", total_share_classes, ""),
        ("Constraints", total_constraints, ""),
        ("Pages", total_pages, ""),
    ]

    if not_found_count:
        stats.append(("NOT FOUND", not_found_count, "stat-warning"))

    stat_html = ""
    for label, value, cls in stats:
        stat_html += f'''
        <div class="stat {cls}">
            <span class="stat-value">{value}</span>
            <span class="stat-label">{label}</span>
        </div>
        '''

    return f'<div class="stats-bar">{stat_html}</div>'


def render_exploration(data: dict) -> str:
    """Render exploration summary section."""
    exploration = data.get("exploration", {})
    if not exploration:
        return ""

    tables = exploration.get("tables_discovered", [])
    cross_refs = exploration.get("cross_references", [])
    observations = exploration.get("observations", [])
    pages_explored = exploration.get("pages_explored", [])

    sections = []

    # Tables discovered
    if tables:
        table_items = []
        for t in tables:
            table_type = t.get("table_type", "unknown")
            pages = f"p.{t.get('page_start', '?')}-{t.get('page_end', '?')}"
            cols = ", ".join(t.get("columns", [])[:5])
            notes = t.get("notes", "")
            table_items.append(f'<li><span class="table-type">{escape(table_type)}</span> {pages}'
                             f'{f" - Columns: {escape(cols)}" if cols else ""}'
                             f'{f" <span class=\"muted\">({escape(notes[:50])})</span>" if notes else ""}</li>')
        sections.append(f'''
        <div class="exploration-subsection">
            <h3>Tables Discovered ({len(tables)})</h3>
            <ul class="exploration-list">{''.join(table_items)}</ul>
        </div>
        ''')

    # Cross references
    if cross_refs:
        ref_items = [f'<li>{escape(ref)}</li>' for ref in cross_refs[:10]]
        sections.append(f'''
        <div class="exploration-subsection">
            <h3>Cross References</h3>
            <ul class="exploration-list">{''.join(ref_items)}</ul>
        </div>
        ''')

    # Observations
    if observations:
        obs_items = [f'<li>{escape(obs)}</li>' for obs in observations[:10]]
        sections.append(f'''
        <div class="exploration-subsection">
            <h3>Explorer Observations</h3>
            <ul class="exploration-list">{''.join(obs_items)}</ul>
        </div>
        ''')

    # Pages explored
    if pages_explored:
        ranges = [f"{s}-{e}" for s, e in pages_explored]
        sections.append(f'''
        <div class="exploration-subsection">
            <h3>Pages Explored</h3>
            <p class="muted">{", ".join(ranges)}</p>
        </div>
        ''')

    if not sections:
        return ""

    return f'''
    <details class="exploration-section">
        <summary><h2>Document Exploration</h2></summary>
        <div class="exploration-content">
            {''.join(sections)}
        </div>
    </details>
    '''


def render_metadata(data: dict) -> str:
    """Render metadata section."""
    metadata = data.get("metadata", {})
    if not metadata:
        return ""

    rows = []
    for key, val in metadata.items():
        if key in ("cost_summary", "agent_notes"):
            continue
        rows.append(f'''
            <tr>
                <td class="field-name">{escape(key.replace('_', ' ').title())}</td>
                <td class="field-value">{render_value(val)}</td>
            </tr>
        ''')

    notes = metadata.get("agent_notes", [])
    notes_html = ""
    if notes:
        note_items = [f'<li>{escape(n)}</li>' for n in notes[:10]]
        notes_html = f'''
        <div class="exploration-subsection">
            <h3>Agent Notes</h3>
            <ul class="exploration-list">{''.join(note_items)}</ul>
        </div>
        '''

    return f'''
    <details class="metadata-section">
        <summary><h2>Extraction Metadata</h2></summary>
        <div class="exploration-content">
            <table class="field-table">
                {''.join(rows)}
            </table>
            {notes_html}
        </div>
    </details>
    '''


def render_external_refs(data: dict) -> str:
    """Render external references section."""
    refs = data.get("external_references", [])
    if not refs:
        return ""

    items = []
    for ref in refs:
        doc = ref.get("document_name", "Unknown")
        field = ref.get("field_name", "")
        entity = ref.get("entity_name", "")
        page = ref.get("source_page", "?")
        quote = ref.get("source_quote", "")

        items.append(f'''
        <div class="external-ref">
            <span class="ref-doc">{escape(doc)}</span>
            <span class="ref-field">{escape(field)}</span>
            {f'<span class="ref-entity">({escape(entity)})</span>' if entity else ''}
            <span class="ref-page">p.{page}</span>
            {f'<div class="ref-quote">"{escape(quote[:150])}"</div>' if quote else ''}
        </div>
        ''')

    return f'''
    <details class="external-refs-section">
        <summary><h2>External Document References ({len(refs)})</h2></summary>
        <div class="external-refs-list">
            {''.join(items)}
        </div>
    </details>
    '''


def render_cost_summary(data: dict) -> str:
    """Render cost/token summary section."""
    metadata = data.get("metadata", {})
    cost = metadata.get("cost_summary")
    if not cost:
        return ""

    total_cost = cost.get("total_cost_usd", 0)
    total_tokens = cost.get("total_tokens", 0)
    total_calls = cost.get("total_calls", 0)
    prompt_tokens = cost.get("prompt_tokens", 0)
    completion_tokens = cost.get("completion_tokens", 0)
    by_agent = cost.get("by_agent", {})

    # Overview stats
    overview = f'''
    <div class="cost-overview">
        <div class="cost-stat">
            <span class="cost-value">${total_cost:.4f}</span>
            <span class="cost-label">Total Cost</span>
        </div>
        <div class="cost-stat">
            <span class="cost-value">{total_tokens:,}</span>
            <span class="cost-label">Total Tokens</span>
        </div>
        <div class="cost-stat">
            <span class="cost-value">{total_calls}</span>
            <span class="cost-label">API Calls</span>
        </div>
        <div class="cost-stat">
            <span class="cost-value">{prompt_tokens:,}</span>
            <span class="cost-label">Prompt Tokens</span>
        </div>
        <div class="cost-stat">
            <span class="cost-value">{completion_tokens:,}</span>
            <span class="cost-label">Completion Tokens</span>
        </div>
    </div>
    '''

    # By-agent breakdown
    agent_rows = ""
    if by_agent:
        for agent, stats in by_agent.items():
            calls = stats.get("calls", 0)
            p_tokens = stats.get("prompt_tokens", 0)
            c_tokens = stats.get("completion_tokens", 0)
            agent_cost = stats.get("cost_usd", 0)
            agent_rows += f'''
            <tr>
                <td>{escape(agent)}</td>
                <td>{calls}</td>
                <td>{p_tokens:,}</td>
                <td>{c_tokens:,}</td>
                <td>${agent_cost:.4f}</td>
            </tr>
            '''

    breakdown = ""
    if agent_rows:
        breakdown = f'''
        <table class="cost-breakdown">
            <thead>
                <tr>
                    <th>Agent</th>
                    <th>Calls</th>
                    <th>Prompt</th>
                    <th>Completion</th>
                    <th>Cost</th>
                </tr>
            </thead>
            <tbody>
                {agent_rows}
            </tbody>
        </table>
        '''

    return f'''
    <details class="cost-section">
        <summary><h2>Cost Summary</h2></summary>
        <div class="cost-content">
            {overview}
            {breakdown}
        </div>
    </details>
    '''


def render_unresolved_questions(data: dict) -> str:
    """Render unresolved questions section."""
    questions = data.get("unresolved_questions", [])
    if not questions:
        return ""

    items = []
    for q in questions:
        priority = q.get("priority", "low")
        field = q.get("field_name", "")
        entity = q.get("entity_name", "")
        text = q.get("question", "")
        agent = q.get("source_agent", "")
        pages = q.get("pages_searched", [])

        pages_str = f"Searched: {', '.join(map(str, pages[:5]))}" if pages else ""

        items.append(f'''
        <div class="question-item priority-{priority}">
            <div class="question-header">
                <span class="question-field">{escape(field)}</span>
                <span class="question-priority">{priority.upper()}</span>
            </div>
            <div class="question-text">{escape(text)}</div>
            <div class="question-meta">
                <span>Entity: {escape(entity)}</span>
                <span>Agent: {escape(agent)}</span>
                {f'<span>{pages_str}</span>' if pages_str else ''}
            </div>
        </div>
        ''')

    return f'''
    <details class="questions-section">
        <summary><h2>Unresolved Questions ({len(questions)})</h2></summary>
        <div class="questions-list">
            {''.join(items)}
        </div>
    </details>
    '''


def render_extraction_plan(data: dict) -> str:
    """Render extraction plan section."""
    plan = data.get("extraction_plan")
    if not plan:
        return ""

    umbrella_pages = plan.get("umbrella_pages", [])
    fund_assignments = plan.get("fund_assignments", [])
    broadcast_tables = plan.get("broadcast_tables", [])
    parallel_safe = plan.get("parallel_safe", True)

    # Stats
    stats = f'''
    <div class="plan-stats">
        <div class="plan-stat">
            <strong>Umbrella Pages:</strong> {', '.join(map(str, umbrella_pages)) if umbrella_pages else 'None'}
        </div>
        <div class="plan-stat">
            <strong>Funds Planned:</strong> {len(fund_assignments)}
        </div>
        <div class="plan-stat">
            <strong>Broadcast Tables:</strong> {len(broadcast_tables)}
        </div>
        <div class="plan-stat">
            <strong>Parallel Safe:</strong> {'Yes' if parallel_safe else 'No'}
        </div>
    </div>
    '''

    # Fund assignments table
    fund_rows = ""
    for fa in fund_assignments[:20]:  # Limit to 20
        name = fa.get("fund_name", "Unknown")
        dedicated = fa.get("dedicated_pages", [])
        isin_pages = fa.get("isin_lookup_pages", [])
        fee_pages = fa.get("fee_lookup_pages", [])
        fund_rows += f'''
        <tr>
            <td>{escape(name[:50])}</td>
            <td>{', '.join(map(str, dedicated[:5])) or '-'}</td>
            <td>{', '.join(map(str, isin_pages[:3])) or '-'}</td>
            <td>{', '.join(map(str, fee_pages[:3])) or '-'}</td>
        </tr>
        '''

    assignments_html = ""
    if fund_rows:
        assignments_html = f'''
        <div class="plan-subsection">
            <h3>Fund Page Assignments</h3>
            <table class="plan-table">
                <thead>
                    <tr>
                        <th>Fund Name</th>
                        <th>Dedicated Pages</th>
                        <th>ISIN Lookup</th>
                        <th>Fee Lookup</th>
                    </tr>
                </thead>
                <tbody>
                    {fund_rows}
                </tbody>
            </table>
            {f'<p class="muted">Showing first 20 of {len(fund_assignments)} funds</p>' if len(fund_assignments) > 20 else ''}
        </div>
        '''

    # Broadcast tables
    broadcast_html = ""
    if broadcast_tables:
        bt_items = []
        for bt in broadcast_tables:
            table_type = bt.get("table_type", "unknown")
            pages = bt.get("pages", [])
            bt_items.append(f'<li><span class="table-type">{escape(table_type)}</span> pages: {", ".join(map(str, pages))}</li>')
        broadcast_html = f'''
        <div class="plan-subsection">
            <h3>Broadcast Tables</h3>
            <ul class="exploration-list">{''.join(bt_items)}</ul>
        </div>
        '''

    return f'''
    <details class="plan-section">
        <summary><h2>Extraction Plan</h2></summary>
        <div class="plan-content">
            {stats}
            {broadcast_html}
            {assignments_html}
        </div>
    </details>
    '''


def render_discovered_fields(data: dict) -> str:
    """Render discovered fields section.

    These are PMS-relevant fields the LLM found beyond the standard schema.
    Grouped by category for easier review.
    """
    fields = data.get("discovered_fields", [])
    if not fields:
        return ""

    # Group by category
    by_category: dict[str, list] = {}
    for f in fields:
        cat = f.get("category", "other")
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append(f)

    category_sections = []
    for category, category_fields in sorted(by_category.items()):
        items = []
        for f in category_fields:
            field_name = f.get("field_name", "unknown")
            value = f.get("value", "")
            entity = f.get("entity_name", "umbrella-level")
            page = f.get("source_page", "?")
            quote = f.get("source_quote", "")
            rationale = f.get("rationale", "")

            items.append(f'''
            <div class="discovered-field">
                <div class="discovered-header">
                    <span class="discovered-name">{escape(field_name)}</span>
                    <span class="discovered-entity">{escape(str(entity) if entity else "umbrella-level")}</span>
                    <span class="discovered-page">p.{page}</span>
                </div>
                <div class="discovered-value">{escape(str(value)[:200])}</div>
                {f'<div class="discovered-rationale">{escape(rationale)}</div>' if rationale else ''}
                {f'<div class="discovered-quote">"{escape(quote[:150])}"</div>' if quote else ''}
            </div>
            ''')

        category_sections.append(f'''
        <div class="discovered-category">
            <h3 class="category-header">{escape(category.replace('_', ' ').title())} ({len(category_fields)})</h3>
            <div class="discovered-list">
                {''.join(items)}
            </div>
        </div>
        ''')

    return f'''
    <details class="discovered-section" open>
        <summary><h2>Discovered Fields ({len(fields)})</h2></summary>
        <p class="section-description">PMS-relevant fields found beyond the standard schema</p>
        <div class="discovered-content">
            {''.join(category_sections)}
        </div>
    </details>
    '''


def render_schema_suggestions(data: dict) -> str:
    """Render schema suggestions section.

    These are suggestions from the LLM for fields to add to the standard schema,
    based on fields that appeared frequently across funds.
    """
    suggestions = data.get("schema_suggestions", [])
    if not suggestions:
        return ""

    rows = []
    for s in suggestions:
        field = s.get("suggested_field", "unknown")
        location = s.get("suggested_location", "?")
        rationale = s.get("rationale", "")
        count = s.get("occurrence_count", 1)

        rows.append(f'''
        <tr>
            <td class="suggestion-field">{escape(field)}</td>
            <td class="suggestion-location"><span class="location-badge location-{location.lower()}">{escape(location)}</span></td>
            <td class="suggestion-count">{count}</td>
            <td class="suggestion-rationale">{escape(rationale)}</td>
        </tr>
        ''')

    return f'''
    <details class="suggestions-section">
        <summary><h2>Schema Suggestions ({len(suggestions)})</h2></summary>
        <p class="section-description">Suggestions for new fields to add to the standard schema</p>
        <table class="suggestions-table">
            <thead>
                <tr>
                    <th>Field Name</th>
                    <th>Location</th>
                    <th>Count</th>
                    <th>Rationale</th>
                </tr>
            </thead>
            <tbody>
                {''.join(rows)}
            </tbody>
        </table>
    </details>
    '''
