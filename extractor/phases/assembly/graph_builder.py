"""Graph builder for constructing the final ProspectusGraph.

Handles the conversion of raw extracted data into typed Pydantic models
with full provenance tracking.
"""

from datetime import date

from extractor.core import parse_fund_constraints, parse_share_class_constraints
from extractor.pydantic_models import (
    Umbrella,
    SubFund,
    ShareClass,
    ProspectusGraph,
    ExtractionMetadata,
    ExternalDocReference,
    DiscoveredField,
    SchemaSuggestion,
    DocumentStructure,
    TableLocation,
    ExplorationSummary,
)
from extractor.phases.assembly.converters import to_extracted_value, to_name_extracted_value
from extractor.phases.assembly.statistics import (
    build_cost_summary,
    build_unresolved_questions,
    build_extraction_plan_summary,
)


def build_umbrella(umbrella_data: dict, default_name: str) -> Umbrella:
    """Build the Umbrella model from extracted data.

    Args:
        umbrella_data: Raw extracted umbrella data from extraction phase
        default_name: Fallback name from planner if extraction failed

    Returns:
        Typed Umbrella model with all fields as ExtractedValue
    """
    umbrella_name = to_name_extracted_value(
        umbrella_data.get("name"),
        default=default_name
    )

    return Umbrella(
        name=umbrella_name,
        legal_form=to_extracted_value(umbrella_data.get("legal_form"), "legal_form"),
        product_type=to_extracted_value(umbrella_data.get("product_type"), "product_type"),
        domicile=to_extracted_value(umbrella_data.get("domicile"), "domicile"),
        inception_date=to_extracted_value(umbrella_data.get("inception_date"), "inception_date"),
        management_company=to_extracted_value(umbrella_data.get("management_company"), "management_company"),
        depositary=to_extracted_value(umbrella_data.get("depositary"), "depositary"),
        auditor=to_extracted_value(umbrella_data.get("auditor"), "auditor"),
        regulator=to_extracted_value(umbrella_data.get("regulator"), "regulator"),
        registered_office=to_extracted_value(umbrella_data.get("registered_office"), "registered_office"),
    )


def build_share_class(sc_data: dict) -> ShareClass:
    """Build a ShareClass model from extracted data.

    Args:
        sc_data: Raw extracted share class data

    Returns:
        Typed ShareClass model with constraints parsed
    """
    # Convert flat fields to ExtractedValue for constraint parsing
    currency_hedged_ev = to_extracted_value(sc_data.get("currency_hedged"), "currency_hedged")
    hedging_details_ev = to_extracted_value(sc_data.get("hedging_details"), "hedging_details")
    investor_restrictions_ev = to_extracted_value(sc_data.get("investor_restrictions"), "investor_restrictions")
    minimum_investment_ev = to_extracted_value(sc_data.get("minimum_investment"), "minimum_investment")
    minimum_holding_period_ev = to_extracted_value(sc_data.get("minimum_holding_period"), "minimum_holding_period")

    # Parse share class constraints from text
    sc_constraints = parse_share_class_constraints(
        currency_hedged=currency_hedged_ev,
        hedging_details=hedging_details_ev,
        investor_restrictions=investor_restrictions_ev,
        minimum_investment=minimum_investment_ev,
        minimum_holding_period=minimum_holding_period_ev,
    )

    return ShareClass(
        name=to_name_extracted_value(sc_data.get("name"), "Unknown"),
        isin=to_extracted_value(sc_data.get("isin"), "isin"),
        currency=to_extracted_value(sc_data.get("currency"), "currency"),
        currency_hedged=currency_hedged_ev,
        hedging_details=hedging_details_ev,
        distribution_policy=to_extracted_value(sc_data.get("distribution_policy"), "distribution_policy"),
        dividend_frequency=to_extracted_value(sc_data.get("dividend_frequency"), "dividend_frequency"),
        dividend_dates=to_extracted_value(sc_data.get("dividend_dates"), "dividend_dates"),
        investor_type=to_extracted_value(sc_data.get("investor_type"), "investor_type"),
        investor_restrictions=investor_restrictions_ev,
        minimum_investment=minimum_investment_ev,
        minimum_subsequent=to_extracted_value(sc_data.get("minimum_subsequent"), "minimum_subsequent"),
        minimum_holding_period=minimum_holding_period_ev,
        management_fee=to_extracted_value(sc_data.get("management_fee"), "management_fee"),
        performance_fee=to_extracted_value(sc_data.get("performance_fee"), "performance_fee"),
        entry_fee=to_extracted_value(sc_data.get("entry_fee"), "entry_fee"),
        exit_fee=to_extracted_value(sc_data.get("exit_fee"), "exit_fee"),
        ongoing_charges=to_extracted_value(sc_data.get("ongoing_charges"), "ongoing_charges"),
        dealing_frequency=to_extracted_value(sc_data.get("dealing_frequency"), "dealing_frequency"),
        dealing_cutoff=to_extracted_value(sc_data.get("dealing_cutoff"), "dealing_cutoff"),
        valuation_point=to_extracted_value(sc_data.get("valuation_point"), "valuation_point"),
        settlement_period=to_extracted_value(sc_data.get("settlement_period"), "settlement_period"),
        listing=to_extracted_value(sc_data.get("listing"), "listing"),
        launch_date=to_extracted_value(sc_data.get("launch_date"), "launch_date"),
        constraints=sc_constraints,
    )


def build_subfund(fund_data: dict) -> SubFund:
    """Build a SubFund model from extracted data.

    Args:
        fund_data: Raw extracted fund data with share_classes list

    Returns:
        Typed SubFund model with share classes and constraints
    """
    # Build share classes
    share_classes = [
        build_share_class(sc)
        for sc in fund_data.get("share_classes", [])
    ]

    # Convert flat fields to ExtractedValue for constraint parsing
    investment_restrictions_ev = to_extracted_value(
        fund_data.get("investment_restrictions"), "investment_restrictions"
    )
    leverage_policy_ev = to_extracted_value(
        fund_data.get("leverage_policy"), "leverage_policy"
    )
    derivatives_usage_ev = to_extracted_value(
        fund_data.get("derivatives_usage"), "derivatives_usage"
    )

    # Parse fund constraints from text
    fund_constraints = parse_fund_constraints(
        investment_restrictions=investment_restrictions_ev,
        leverage_policy=leverage_policy_ev,
        derivatives_usage=derivatives_usage_ev,
    )

    return SubFund(
        name=to_name_extracted_value(fund_data.get("name"), "Unknown"),
        investment_objective=to_extracted_value(fund_data.get("investment_objective"), "investment_objective"),
        investment_policy=to_extracted_value(fund_data.get("investment_policy"), "investment_policy"),
        benchmark=to_extracted_value(fund_data.get("benchmark"), "benchmark"),
        benchmark_usage=to_extracted_value(fund_data.get("benchmark_usage"), "benchmark_usage"),
        asset_class=to_extracted_value(fund_data.get("asset_class"), "asset_class"),
        geographic_focus=to_extracted_value(fund_data.get("geographic_focus"), "geographic_focus"),
        sector_focus=to_extracted_value(fund_data.get("sector_focus"), "sector_focus"),
        currency_base=to_extracted_value(fund_data.get("currency_base"), "currency_base"),
        risk_profile=to_extracted_value(fund_data.get("risk_profile"), "risk_profile"),
        # Global constraint extractions (kept for auditability, parsed into constraints list)
        global_leverage_policy=leverage_policy_ev,
        global_derivatives_usage=derivatives_usage_ev,
        global_investment_restrictions=investment_restrictions_ev,
        inception_date=to_extracted_value(fund_data.get("inception_date"), "inception_date"),
        share_classes=share_classes,
        constraints=fund_constraints,
    )


def build_exploration_summary(exploration_notes: list) -> ExplorationSummary:
    """Build exploration summary from exploration notes.

    Aggregates information from all explorers into a single summary.

    Args:
        exploration_notes: List of ExplorationNotes from exploration phase

    Returns:
        ExplorationSummary with aggregated document structure info
    """
    all_toc_pages: set[int] = set()
    all_umbrella_pages: set[int] = set()
    fund_sections: dict[str, list[int]] = {}
    all_tables: list[TableLocation] = []
    all_cross_refs: list[str] = []
    all_observations: list[str] = []
    page_ranges: list[tuple[int, int]] = []

    for notes in exploration_notes:
        # Page coverage
        page_ranges.append((notes.page_start, notes.page_end))

        # Document structure
        all_toc_pages.update(notes.toc_pages)
        all_umbrella_pages.update(notes.umbrella_info_pages)

        # Fund mentions with dedicated sections
        for fund in notes.funds_mentioned:
            if fund.has_dedicated_section:
                if fund.name not in fund_sections:
                    fund_sections[fund.name] = []
                fund_sections[fund.name].append(fund.page)

        # Tables
        for table in notes.tables:
            all_tables.append(TableLocation(
                table_type=table.table_type,
                page_start=table.page_start,
                page_end=table.page_end,
                columns=table.columns,
                notes=table.notes,
            ))

        # Cross-references
        for ref in notes.cross_references:
            ref_text = f"Page {ref.source_page}: {ref.text}"
            if ref.target_description:
                ref_text += f" ({ref.target_description})"
            all_cross_refs.append(ref_text)

        # Observations
        all_observations.extend(notes.observations)

    return ExplorationSummary(
        document_structure=DocumentStructure(
            toc_pages=sorted(all_toc_pages),
            umbrella_info_pages=sorted(all_umbrella_pages),
            fund_dedicated_sections=fund_sections,
        ),
        tables_discovered=all_tables,
        cross_references=all_cross_refs,
        observations=all_observations,
        pages_explored=sorted(page_ranges),
    )


def collect_external_references(knowledge, exploration_notes: list) -> list[ExternalDocReference]:
    """Collect external references from multiple sources.

    Gathers external document references from:
    1. Knowledge graph external_refs
    2. Document inventory fields_external
    3. Classified cross-references in exploration notes

    Args:
        knowledge: DocumentKnowledge instance
        exploration_notes: List of ExplorationNotes

    Returns:
        Deduplicated list of ExternalDocReference models
    """
    external_refs = []
    seen_refs: set[tuple[str, str]] = set()  # (field_name, doc_name) to avoid duplicates

    # Invalid field names to skip (LLM sometimes outputs "null" string)
    invalid_field_names = {"null", "none", "unknown", ""}

    def is_valid_field_name(name: str | None) -> bool:
        """Check if field_name is valid (not null/none/unknown/empty)."""
        return name is not None and name.lower() not in invalid_field_names

    # Source 1: From knowledge graph external_refs
    for ref in knowledge.external_refs:
        if not is_valid_field_name(ref.field_name):
            continue  # Skip invalid field names
        key = (ref.field_name, ref.external_doc)
        if key not in seen_refs:
            seen_refs.add(key)
            external_refs.append(ExternalDocReference(
                document_name=ref.external_doc,
                field_name=ref.field_name,
                entity_name=ref.entity_name,
                source_page=ref.source_page,
                source_quote=ref.source_quote,
            ))

    # Source 2: From document inventory fields_external
    if knowledge.document_inventory:
        for ext_ref in knowledge.document_inventory.fields_external:
            if not is_valid_field_name(ext_ref.field_name):
                continue  # Skip invalid field names
            key = (ext_ref.field_name, ext_ref.external_doc)
            if key not in seen_refs:
                seen_refs.add(key)
                external_refs.append(ExternalDocReference(
                    document_name=ext_ref.external_doc,
                    field_name=ext_ref.field_name,
                    entity_name=None if ext_ref.applies_to == "all" else ext_ref.applies_to,
                    source_page=ext_ref.source_page,
                    source_quote=ext_ref.source_quote,
                ))

    # Source 3: From classified cross-references in exploration notes
    for notes in exploration_notes:
        for xref in notes.cross_references:
            is_external = getattr(xref, 'is_external', False)
            field_hint = getattr(xref, 'field_hint', None)
            external_doc = getattr(xref, 'external_doc', None)

            if is_external and is_valid_field_name(field_hint) and external_doc:
                key = (field_hint, external_doc)
                if key not in seen_refs:
                    seen_refs.add(key)
                    external_refs.append(ExternalDocReference(
                        document_name=external_doc,
                        field_name=field_hint,
                        entity_name=None,
                        source_page=xref.source_page,
                        source_quote=xref.text[:200] if xref.text else "",
                    ))

    return external_refs


def collect_discovered_fields(discovered_fields: list) -> list[DiscoveredField]:
    """Convert raw discovered field dicts to typed models.

    Args:
        discovered_fields: List of dicts with discovered field data

    Returns:
        List of DiscoveredField models
    """
    return [
        DiscoveredField(
            field_name=df.get("field_name", "unknown"),
            value=df.get("value", ""),
            category=df.get("category", "other"),
            entity_name=df.get("entity_name"),
            source_page=df.get("source_page"),
            source_quote=df.get("source_quote"),
            rationale=df.get("rationale"),
            pms_use_case=df.get("pms_use_case"),
        )
        for df in discovered_fields
    ]


def collect_schema_suggestions(schema_suggestions: list) -> list[SchemaSuggestion]:
    """Aggregate and count schema suggestions.

    Only returns suggestions that appear in 2+ entities.

    Args:
        schema_suggestions: List of raw suggestion dicts

    Returns:
        List of SchemaSuggestion models with occurrence counts
    """
    suggestion_counts: dict[str, dict] = {}
    for sugg in schema_suggestions:
        key = sugg.get("suggested_field", "")
        if key in suggestion_counts:
            suggestion_counts[key]["occurrence_count"] += 1
        else:
            suggestion_counts[key] = {
                "suggested_field": sugg.get("suggested_field", ""),
                "suggested_location": sugg.get("suggested_location", "SubFund"),
                "rationale": sugg.get("rationale", ""),
                "occurrence_count": 1,
            }

    return [
        SchemaSuggestion(
            suggested_field=v["suggested_field"],
            suggested_location=v["suggested_location"],
            rationale=v["rationale"],
            occurrence_count=v["occurrence_count"],
        )
        for v in suggestion_counts.values()
        if v["occurrence_count"] >= 2  # Only suggest if found in 2+ entities
    ]


def build_graph(context) -> ProspectusGraph:
    """Build the complete ProspectusGraph from pipeline context.

    This is the main entry point for graph construction. It orchestrates
    all the sub-builders to create the final typed output.

    Args:
        context: PhaseContext with all extracted data

    Returns:
        Complete ProspectusGraph with all entities and metadata
    """
    plan = context.plan

    # Get umbrella name with provenance preserved
    default_name = plan.umbrella_name if plan else "Unknown"
    umbrella = build_umbrella(context.umbrella_data, default_name)

    # Build SubFunds
    subfunds = [build_subfund(fund_data) for fund_data in context.funds_data]

    # Build metadata
    metadata = ExtractionMetadata(
        source_file=context.pdf.filename,
        total_pages=context.pdf.page_count,
        extraction_date=date.today().isoformat(),
        extraction_version="1.0",
        agent_notes=[
            f"Explorers: {len(context.exploration_notes)}",
            f"Funds extracted: {len(subfunds)}",
            f"Critic verifications: {len(context.critic_results)}",
        ],
    )

    # Build auxiliary collections
    external_refs = collect_external_references(context.knowledge, context.exploration_notes)
    discovered = collect_discovered_fields(context.discovered_fields)
    schema_suggestions = collect_schema_suggestions(context.schema_suggestions)
    exploration_summary = build_exploration_summary(context.exploration_notes)

    # Build cost summary from cost tracker
    cost_summary = build_cost_summary(context.cost_tracker)
    metadata.cost_summary = cost_summary

    # Build unresolved questions from knowledge graph
    unresolved_questions = build_unresolved_questions(context.knowledge)

    # Build extraction plan summary
    extraction_plan = build_extraction_plan_summary(plan)

    return ProspectusGraph(
        umbrella=umbrella,
        sub_funds=subfunds,
        metadata=metadata,
        exploration=exploration_summary,
        external_references=external_refs,
        discovered_fields=discovered,
        schema_suggestions=schema_suggestions,
        unresolved_questions=unresolved_questions,
        extraction_plan=extraction_plan,
    )
