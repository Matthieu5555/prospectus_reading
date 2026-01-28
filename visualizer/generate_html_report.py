#!/usr/bin/env python3
"""Generate interactive HTML dashboard for prospectus extraction results.

Main entry point for visualization. Combines template and render functions
to produce a hierarchical, searchable dashboard with provenance details.

Usage:
    visualize extraction_result.json
"""

import json
import sys
from pathlib import Path
from html import escape

from visualizer.define_html_template import CSS_STYLES, JS_SCRIPT
from visualizer.render_extracted_values import (
    render_fund,
    render_umbrella,
    render_stats,
    render_exploration,
    render_metadata,
    render_external_refs,
    render_cost_summary,
    render_unresolved_questions,
    render_extraction_plan,
    render_discovered_fields,
    render_schema_suggestions,
)


def load_json(path: str) -> dict:
    """Load JSON data from file."""
    with open(path) as f:
        return json.load(f)


def generate_html(data: dict, title: str) -> str:
    """Generate the full HTML dashboard."""
    umbrella = data.get("umbrella", {})
    # sub_funds can be top-level OR nested under umbrella
    sub_funds = data.get("sub_funds") or umbrella.get("sub_funds", [])

    funds_html = "\n".join(render_fund(f, i) for i, f in enumerate(sub_funds))

    return f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{escape(title)}</title>
    <style>
{CSS_STYLES}
    </style>
</head>
<body>
    <div class="header">
        <h1>{escape(title)}</h1>
        <div class="controls">
            <input type="text" class="search-box" placeholder="Search funds..." id="search">
            <button class="btn" onclick="expandAll()">Expand All</button>
            <button class="btn" onclick="collapseAll()">Collapse All</button>
        </div>
    </div>

    {render_stats(data)}
    {render_umbrella(data)}

    <h2 style="color: var(--text); margin: 20px 0 10px;">Sub-Funds</h2>
    <div id="funds-container">
        {funds_html}
    </div>

    <h2 style="color: var(--text); margin: 20px 0 10px;">Discovered Information</h2>
    {render_discovered_fields(data)}
    {render_schema_suggestions(data)}

    <h2 style="color: var(--text); margin: 20px 0 10px;">Extraction Data</h2>
    {render_cost_summary(data)}
    {render_metadata(data)}
    {render_exploration(data)}
    {render_extraction_plan(data)}
    {render_unresolved_questions(data)}
    {render_external_refs(data)}

    <div id="no-results" class="no-results" style="display: none;">
        No funds match your search.
    </div>

    <script>
{JS_SCRIPT}
    </script>
</body>
</html>
'''


def main():
    """CLI entry point - generate HTML and open in browser."""
    import tempfile
    import webbrowser

    if len(sys.argv) < 2:
        print("Usage: visualize <json_file>")
        sys.exit(1)

    json_path = Path(sys.argv[1])

    if not json_path.exists():
        print(f"Error: {json_path} not found")
        sys.exit(1)

    print(f"Loading {json_path}...")
    data = load_json(json_path)

    title = json_path.stem.replace("_", " ").title()
    html = generate_html(data, title)

    # Write to temp file and open
    with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
        f.write(html)
        temp_path = f.name

    print("Opening in browser...")
    webbrowser.open(f"file://{temp_path}")


if __name__ == "__main__":
    main()
