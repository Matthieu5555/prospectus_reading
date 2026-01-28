"""Assembly sub-module for building the final ProspectusGraph.

This module is split into focused components:
- converters: Transform LLM output to ExtractedValue models
- statistics: Compute provenance statistics and summaries
- graph_builder: Build the complete ProspectusGraph
- gap_filling: Fill gaps with umbrella-level constraints
"""

from extractor.phases.assembly.converters import (
    not_found_reason_from_string,
    to_name_extracted_value,
    to_extracted_value,
)
from extractor.phases.assembly.statistics import (
    count_provenance,
    build_cost_summary,
    build_unresolved_questions,
    build_extraction_plan_summary,
)
from extractor.phases.assembly.graph_builder import (
    build_umbrella,
    build_share_class,
    build_subfund,
    build_exploration_summary,
    collect_external_references,
    collect_discovered_fields,
    collect_schema_suggestions,
    build_graph,
)
from extractor.phases.assembly.gap_filling import (
    fill_gaps,
    extract_umbrella_constraints,
)

__all__ = [
    # Converters
    "not_found_reason_from_string",
    "to_name_extracted_value",
    "to_extracted_value",
    # Statistics
    "count_provenance",
    "build_cost_summary",
    "build_unresolved_questions",
    "build_extraction_plan_summary",
    # Graph builder
    "build_umbrella",
    "build_share_class",
    "build_subfund",
    "build_exploration_summary",
    "collect_external_references",
    "collect_discovered_fields",
    "collect_schema_suggestions",
    "build_graph",
    # Gap filling
    "fill_gaps",
    "extract_umbrella_constraints",
]
