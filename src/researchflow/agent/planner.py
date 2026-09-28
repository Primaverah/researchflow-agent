"""Deterministic research task planning."""

from uuid import uuid4

from researchflow.domain import PlanStep, ResearchPlan


class RulePlanner:
    """Create the fixed four-step plan used by the offline agent."""

    def create_plan(self, query: str) -> ResearchPlan:
        """Return a stable search, read, summarize, and save plan."""
        if not query.strip():
            raise ValueError("query cannot be blank")
        return ResearchPlan(
            plan_id=str(uuid4()),
            goal=query,
            steps=[
                PlanStep(
                    step_id="search",
                    description="Search local documents",
                    tool_name="search_documents",
                ),
                PlanStep(
                    step_id="read",
                    description="Read relevant documents",
                    tool_name="read_document",
                ),
                PlanStep(
                    step_id="summarize",
                    description="Extract and summarize relevant content",
                ),
                PlanStep(
                    step_id="save",
                    description="Save the research report",
                    tool_name="save_note",
                ),
            ],
        )
