"""Resolver phase - resolves EXTRACTION_FAILED fields with specialized resolvers.

This phase runs after extraction to find values that extractors couldn't locate.
It uses the knowledge graph and specialized field resolvers to search the document
more thoroughly.
"""

import asyncio
from dataclasses import dataclass, field

from extractor.phases.phase_base import PhaseRunner
from extractor.core.field_searchers import get_resolver, ResolverResult
from extractor.core import ExtractionError, ErrorCategory, ErrorSeverity
from extractor.core.value_helpers import get_raw_value
from extractor.pydantic_models.provenance import NotFoundReason


@dataclass
class ResolverPhaseResult:
    """Result from the resolver phase."""

    total_actionable: int           # Fields that needed resolution
    resolved_count: int             # Successfully resolved
    still_failed: int               # Still EXTRACTION_FAILED
    upgraded_to_typed: int          # Changed to proper NOT_FOUND reason
    results: list[ResolverResult] = field(default_factory=list)


class FailureRecoveryPhase(PhaseRunner[ResolverPhaseResult]):
    """Phase that retries EXTRACTION_FAILED fields with specialized resolvers.

    Pipeline position: Explore → Plan → Extract → **FailureRecovery** → Assemble

    For each EXTRACTION_FAILED value:
    1. Check if we have a specialized resolver for that field
    2. Use knowledge graph to find relevant pages
    3. Run resolver to attempt extraction
    4. Update the value with result or upgraded NOT_FOUND reason
    """

    name = "FailureRecovery"

    async def run(self) -> ResolverPhaseResult:
        """Execute resolver phase.

        Returns:
            ResolverPhaseResult with resolution statistics.
        """
        # Collect all actionable NOT_FOUND values
        actionable = self._collect_actionable_fields()

        if not actionable:
            self.log("No actionable NOT_FOUND fields to resolve")
            return ResolverPhaseResult(
                total_actionable=0,
                resolved_count=0,
                still_failed=0,
                upgraded_to_typed=0,
            )

        self.start(len(actionable))
        self.log(f"Found {len(actionable)} fields needing resolution")

        # Group by field type for efficient resolution
        by_field: dict[str, list[dict]] = {}
        for item in actionable:
            field_name = item["field_name"]
            if field_name not in by_field:
                by_field[field_name] = []
            by_field[field_name].append(item)

        # Resolve each field type
        results = []
        resolved = 0
        upgraded = 0

        search_context = self.context.create_search_context()

        # Build list of resolver tasks
        async def resolve_item(field_name: str, item: dict, resolver) -> tuple[str, dict, any, Exception | None]:
            """Resolve a single item, returns (field_name, item, result, error)."""
            async with self.context.semaphore:
                try:
                    result = await resolver.resolve(
                        item["entity_name"],
                        self.context.knowledge,
                        self.context.pdf,
                        search_context,
                    )
                    return (field_name, item, result, None)
                except Exception as e:
                    return (field_name, item, None, e)

        tasks = []
        items_without_resolver = []

        for field_name, items in by_field.items():
            resolver = get_resolver(field_name)

            if not resolver:
                # No resolver - handle these synchronously
                items_without_resolver.extend((field_name, item) for item in items)
                continue

            self.log(f"Queuing {len(items)} {field_name} fields for resolution")
            for item in items:
                tasks.append(resolve_item(field_name, item, resolver))

        # Handle items without resolvers (no async needed)
        # Use inventory to suggest appropriate NOT_FOUND reason
        for field_name, item in items_without_resolver:
            # Use knowledge graph's suggest_not_found_reason which checks inventory first
            suggested_reason = self.context.knowledge.suggest_not_found_reason(field_name)

            external_ref = None
            if suggested_reason == "in_external_doc":
                # Get the external doc name from inventory or external_refs
                if self.context.knowledge.document_inventory:
                    inv_ref = self.context.knowledge.document_inventory.get_external_ref(field_name)
                    if inv_ref:
                        external_ref = inv_ref.external_doc
                if not external_ref:
                    ext_ref = self.context.knowledge.get_external_ref_for_field(field_name)
                    if ext_ref:
                        external_ref = ext_ref.external_doc

            self._update_value(item, suggested_reason, external_ref)
            upgraded += 1

        # Run all resolver tasks in parallel
        if tasks:
            self.log(f"Running {len(tasks)} resolver tasks in parallel")
            task_results = await asyncio.gather(*tasks, return_exceptions=True)

            for i, result in enumerate(task_results):
                if isinstance(result, Exception):
                    self.log(f"Resolver task failed: {result}", "error")
                    self.context.errors.add(ExtractionError(
                        category=ErrorCategory.LLM_API,
                        severity=ErrorSeverity.WARNING,
                        message=f"Resolver task exception: {result}",
                        phase=self.name,
                    ))
                    continue

                field_name, item, resolver_result, error = result

                if error:
                    self.log(f"Resolver failed for {field_name}/{item['entity_name']}: {error}", "error")
                    self.context.errors.add(ExtractionError(
                        category=ErrorCategory.LLM_API,
                        severity=ErrorSeverity.WARNING,
                        message=str(error),
                        phase=self.name,
                        entity_name=item["entity_name"],
                    ))
                    continue

                # Apply the result
                self._apply_result(item, resolver_result)

                if resolver_result.value != "NOT_FOUND":
                    resolved += 1
                    results.append(ResolverResult(
                        field_name=field_name,
                        entity_name=item["entity_name"],
                        value=resolver_result,
                        resolver_used="resolver",
                        pages_searched=[]
                    ))
                elif resolver_result.not_found_reason != NotFoundReason.EXTRACTION_FAILED:
                    upgraded += 1

                self.progress(i + 1, len(tasks), "resolving")

        still_failed = len(actionable) - resolved - upgraded

        # Log phase result with metrics
        self.logger.phase_result(
            "FailureRecovery",
            f"{resolved} resolved",
            upgraded=upgraded,
            still_failed=still_failed,
        )

        return ResolverPhaseResult(
            total_actionable=len(actionable),
            resolved_count=resolved,
            still_failed=still_failed,
            upgraded_to_typed=upgraded,
            results=results,
        )

    def _collect_actionable_fields(self) -> list[dict]:
        """Collect all EXTRACTION_FAILED values from funds data."""
        actionable = []

        for fund_data in self.context.funds_data:
            fund_name = get_raw_value(fund_data.get("name"), "Unknown")

            # Check fund-level fields
            for field_name in ["investment_restrictions", "leverage_policy", "derivatives_usage",
                              "benchmark", "currency_base"]:
                value = fund_data.get(field_name)
                if self._is_actionable(value):
                    actionable.append({
                        "type": "fund",
                        "fund_data": fund_data,
                        "field_name": field_name,
                        "entity_name": fund_name,
                        "value_ref": value,
                    })

            # Check share class fields
            for sc in fund_data.get("share_classes", []):
                sc_name = get_raw_value(sc.get("name"), "Unknown")
                full_name = f"{fund_name} - {sc_name}"

                for field_name in ["isin", "management_fee", "ongoing_charges", "entry_fee",
                                  "exit_fee", "dividend_dates", "dividend_frequency",
                                  "valuation_point", "dealing_cutoff", "settlement_period"]:
                    value = sc.get(field_name)
                    if self._is_actionable(value):
                        actionable.append({
                            "type": "share_class",
                            "share_class": sc,
                            "field_name": field_name,
                            "entity_name": full_name,
                            "value_ref": value,
                        })

        return actionable

    def _is_actionable(self, value) -> bool:
        """Check if a value is EXTRACTION_FAILED (actionable)."""
        if value is None:
            return False

        # String NOT_FOUND (legacy)
        if value == "NOT_FOUND":
            return True

        # Dict format
        if isinstance(value, dict):
            if value.get("value") != "NOT_FOUND":
                return False
            # Check if it's specifically EXTRACTION_FAILED or has no reason (legacy)
            reason = value.get("not_found_reason")
            if reason is None:
                return True
            # Handle both string and enum formats
            reason_str = str(reason).lower() if reason else ""
            return reason_str in ("extraction_failed", "notfoundreason.extraction_failed")

        return False

    def _update_value(self, item: dict, reason: str, external_ref: str | None):
        """Update a value with a new NOT_FOUND reason."""
        if item["type"] == "fund":
            target = item["fund_data"]
        else:
            target = item["share_class"]

        field_name = item["field_name"]
        current = target.get(field_name)

        if isinstance(current, dict):
            current["not_found_reason"] = reason
            if external_ref:
                current["external_reference"] = external_ref
                current["rationale"] = f"Value is in {external_ref}"
        else:
            target[field_name] = {
                "value": "NOT_FOUND",
                "not_found_reason": reason,
                "external_reference": external_ref,
                "rationale": f"Value is in {external_ref}" if external_ref else "Confirmed not in document",
                "confidence": 1.0
            }

    def _apply_result(self, item: dict, result):
        """Apply resolver result to the original data."""
        if item["type"] == "fund":
            target = item["fund_data"]
        else:
            target = item["share_class"]

        field_name = item["field_name"]

        # Convert ExtractedValue to dict format
        target[field_name] = {
            "value": result.value,
            "source_page": result.source_page,
            "source_quote": result.source_quote,
            "rationale": result.rationale,
            "confidence": result.confidence,
            "not_found_reason": str(result.not_found_reason) if result.not_found_reason else None,
            "external_reference": result.external_reference,
        }
