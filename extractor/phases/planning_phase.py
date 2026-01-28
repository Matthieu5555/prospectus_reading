"""Planning phase - synthesis of exploration notes into extraction plan."""

from dataclasses import dataclass

from extractor.phases.phase_base import PhaseRunner
from extractor.pydantic_models import PlannerOutput
from extractor.agents import run_planner, enrich_plan_with_pages
from extractor.core import ExtractionError, ErrorCategory, ErrorSeverity


@dataclass
class PlanningResult:
    """Result from the planning phase."""

    plan: PlannerOutput
    tasks_with_pages: int
    tasks_without_pages: int


class PlanningPhase(PhaseRunner[PlanningResult]):
    """Phase 2: Extraction planning.

    Synthesizes exploration notes into a concrete extraction plan with:
    - Fund list from EntityResolution (authoritative source)
    - Page assignments for each fund
    - Broadcast table locations
    """

    name = "Planning"

    async def run(self) -> PlanningResult:
        """Execute planning phase.

        Returns:
            PlanningResult with extraction plan.
        """
        self.start(1, model=self.context.planner_model)

        if not self.context.exploration_notes:
            self.log("No exploration notes available, creating minimal plan", "warning")
            plan = PlannerOutput(
                umbrella_name="Unknown",
                total_funds=0,
                fund_names=[],
                umbrella_pages=[1, 2, 3],
                fund_tasks=[],
            )
            self.context.plan = plan
            self.end("empty plan created")
            return PlanningResult(plan=plan, tasks_with_pages=0, tasks_without_pages=0)

        self.log(f"Synthesizing {len(self.context.exploration_notes)} exploration reports...")

        # Get canonical funds from EntityResolution (authoritative source)
        # fund_name_variants maps all name variants to their canonical form
        name_variants = self.context.fund_name_variants
        canonical_funds = None
        if name_variants:
            canonical_funds = list(set(name_variants.values()))
            self.log(f"Using {len(canonical_funds)} canonical funds from EntityResolution")

        try:
            plan = await run_planner(
                self.context.exploration_notes,
                self.context.pdf.page_count,
                model=self.context.planner_model,
                cost_tracker=self.context.cost_tracker,
                canonical_funds=canonical_funds,
                fund_name_variants=name_variants,
            )

            self.log(f"Plan created for {plan.total_funds} funds")
            if plan.broadcast_tables:
                self.log(f"Broadcast tables: {[t.table_type for t in plan.broadcast_tables]}")

            # Enrich with page ranges - skeleton (TOC) is primary, explorer data is fallback
            plan = enrich_plan_with_pages(
                plan,
                self.context.exploration_notes,
                self.context.pdf.page_count,
                skeleton=self.context.skeleton,
            )

        except Exception as e:
            self.log(f"Planner failed: {e}", "error")
            self.context.errors.add(ExtractionError(
                category=ErrorCategory.LLM_API,
                severity=ErrorSeverity.ERROR,
                message=str(e),
                phase=self.name,
            ))
            # Create minimal plan using canonical funds if available
            fund_list = canonical_funds if canonical_funds else []
            if not fund_list:
                for note in self.context.exploration_notes:
                    for fund in note.funds_mentioned:
                        if fund.name not in fund_list:
                            fund_list.append(fund.name)

            plan = PlannerOutput(
                umbrella_name="Unknown",
                total_funds=len(fund_list),
                fund_names=fund_list,
                umbrella_pages=[1, 2, 3],
                fund_tasks=[],
                observations=[f"Planner error: {e}"],
            )

        # Store in context
        self.context.plan = plan

        # Normalize fund names in knowledge graph
        if name_variants:
            renamed = self.context.store.normalize_fund_names(name_variants)
            if renamed > 0:
                self.log(f"Updated {renamed} fund entities to canonical names in knowledge graph")

        # Count tasks with/without pages
        tasks_with_pages = sum(1 for t in plan.fund_tasks if t.dedicated_pages)
        tasks_without_pages = len(plan.fund_tasks) - tasks_with_pages

        # Log phase result with key metrics
        self.logger.phase_result(
            "Planning",
            f"{plan.umbrella_name}",
            funds=plan.total_funds,
            with_pages=tasks_with_pages,
            broadcast_tables=len(plan.broadcast_tables),
        )

        # Log fund assignments at debug level
        for i, task in enumerate(plan.fund_tasks[:10]):
            pages = f"p{task.dedicated_pages[0]}-{task.dedicated_pages[-1]}" if task.dedicated_pages else "NO PAGES"
            self.log(f"Fund {i+1}: {task.fund_name[:40]} ({pages})")
        if len(plan.fund_tasks) > 10:
            self.log(f"... and {len(plan.fund_tasks) - 10} more funds")

        return PlanningResult(
            plan=plan,
            tasks_with_pages=tasks_with_pages,
            tasks_without_pages=tasks_without_pages,
        )
