"""Statistics and summary builders for the assembly phase.

Provides functions to compute provenance statistics, cost summaries,
and other metadata from the extracted data.
"""

from extractor.core.config import ConfidenceThresholds
from extractor.core.cost_tracker import CostTracker
from extractor.core.document_knowledge import DocumentKnowledge
from extractor.pydantic_models import (
    ProspectusGraph,
    ExtractedValue,
    CostSummary,
    AgentCostBreakdown,
    UnresolvedQuestion,
    FundPageAssignment,
    ExtractionPlanSummary,
)


def count_provenance(graph: ProspectusGraph) -> dict:
    """Count provenance statistics including NOT_FOUND breakdown by reason.

    Traverses the entire graph (umbrella, subfunds, share classes) and counts:
    - Total extracted values
    - High/low confidence values
    - NOT_FOUND values broken down by reason
    - Actionable NOT_FOUND (extraction_failed) that need investigation

    Args:
        graph: The complete ProspectusGraph to analyze

    Returns:
        Dictionary with statistics including total, high_confidence,
        low_confidence, not_found, not_found_by_reason, actionable_not_found
    """
    stats = {
        "total": 0,
        "high_confidence": 0,
        "low_confidence": 0,
        "not_found": 0,
        # NOT_FOUND breakdown by reason
        "not_found_by_reason": {
            "not_in_document": 0,
            "not_applicable": 0,
            "in_external_doc": 0,
            "inherited": 0,
            "extraction_failed": 0,
        },
        # Actionable = extraction_failed (needs investigation)
        "actionable_not_found": 0,
    }

    def count_model(model):
        """Recursively count ExtractedValue instances in a Pydantic model."""
        for _, value in model:
            if isinstance(value, ExtractedValue):
                stats["total"] += 1
                if value.is_not_found:
                    stats["not_found"] += 1
                    # Count by reason
                    if value.not_found_reason:
                        reason_key = str(value.not_found_reason)
                        if reason_key in stats["not_found_by_reason"]:
                            stats["not_found_by_reason"][reason_key] += 1
                        if value.is_actionable_not_found:
                            stats["actionable_not_found"] += 1
                    else:
                        # No reason = legacy, count as extraction_failed
                        stats["not_found_by_reason"]["extraction_failed"] += 1
                        stats["actionable_not_found"] += 1
                elif value.confidence >= ConfidenceThresholds.HIGH:
                    stats["high_confidence"] += 1
                elif value.confidence < ConfidenceThresholds.LOW:
                    stats["low_confidence"] += 1
            elif isinstance(value, list):
                for item in value:
                    if hasattr(item, "__iter__") and not isinstance(item, str):
                        count_model(item)

    count_model(graph.umbrella)
    for subfund in graph.sub_funds:
        count_model(subfund)
        for sc in subfund.share_classes:
            count_model(sc)

    return stats


def build_cost_summary(tracker: CostTracker) -> CostSummary | None:
    """Build cost summary from cost tracker.

    Converts CostTracker data into the output-compatible CostSummary model.

    Args:
        tracker: CostTracker instance with recorded API calls

    Returns:
        CostSummary with total and per-agent breakdowns, or None if no calls
    """
    if not tracker.calls:
        return None

    # Build per-agent breakdown
    by_agent_raw = tracker.by_agent()
    by_agent = {
        agent: AgentCostBreakdown(
            calls=stats["calls"],
            prompt_tokens=stats["prompt_tokens"],
            completion_tokens=stats["completion_tokens"],
            cost_usd=round(stats["cost"], 6),
        )
        for agent, stats in by_agent_raw.items()
    }

    return CostSummary(
        total_calls=tracker.call_count,
        total_tokens=tracker.total_tokens,
        prompt_tokens=tracker.total_prompt_tokens,
        completion_tokens=tracker.total_completion_tokens,
        total_cost_usd=round(tracker.total_cost, 6),
        by_agent=by_agent,
    )


def build_unresolved_questions(knowledge: DocumentKnowledge) -> list[UnresolvedQuestion]:
    """Build unresolved questions list from knowledge graph.

    Converts DocumentKnowledge questions into output-compatible models.
    Only includes unresolved questions.

    Args:
        knowledge: DocumentKnowledge instance with recorded questions

    Returns:
        List of UnresolvedQuestion models
    """
    questions = knowledge.get_unresolved_questions()
    return [
        UnresolvedQuestion(
            question=q.question,
            field_name=q.field_name,
            entity_name=q.entity_name,
            priority=q.priority.value,  # Convert enum to string
            pages_searched=q.pages_searched,
            source_agent=q.source_agent,
        )
        for q in questions
    ]


def build_extraction_plan_summary(plan) -> ExtractionPlanSummary | None:
    """Build extraction plan summary from planner output.

    Summarizes which pages were assigned to each fund for debugging.

    Args:
        plan: PlannerOutput from the planning phase

    Returns:
        ExtractionPlanSummary with fund assignments and table locations,
        or None if no plan available
    """
    if not plan:
        return None

    # Build fund assignments
    fund_assignments = []
    for task in plan.fund_tasks:
        assignment = FundPageAssignment(
            fund_name=task.fund_name,
            dedicated_pages=task.dedicated_pages,
            isin_lookup_pages=task.isin_lookup.table_pages if task.isin_lookup else [],
            fee_lookup_pages=task.fee_lookup.table_pages if task.fee_lookup else [],
        )
        fund_assignments.append(assignment)

    # Build broadcast tables summary
    broadcast_tables = [
        {
            "table_type": bt.table_type,
            "pages": bt.pages,
            "notes": bt.notes,
        }
        for bt in plan.broadcast_tables
    ]

    return ExtractionPlanSummary(
        umbrella_pages=plan.umbrella_pages,
        fund_assignments=fund_assignments,
        broadcast_tables=broadcast_tables,
        parallel_safe=plan.parallel_safe,
    )
