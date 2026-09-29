"""Bounded rule planning and selection for optional web sources."""

from researchflow.agent.models import AgentAction, AgentActionType
from researchflow.agent.planner import RulePlanner
from researchflow.agent.selector import StateSelector
from researchflow.domain import AgentState, PlanStep, ResearchPlan


class WebRulePlanner(RulePlanner):
    def create_plan(self, query: str) -> ResearchPlan:
        plan = super().create_plan(query)
        plan.steps.insert(
            2,
            PlanStep(
                step_id="web_search",
                description="Search public web sources",
                tool_name="web_search",
            ),
        )
        plan.steps.insert(
            3,
            PlanStep(
                step_id="fetch_url",
                description="Read safe public web sources",
                tool_name="fetch_url",
            ),
        )
        return plan


class WebStateSelector(StateSelector):
    def select(self, state: AgentState) -> AgentAction:
        action = super().select(state)
        if action.action_type is not AgentActionType.SUMMARIZE:
            return action
        search_result = self._latest_result(state, "web_search")
        if search_result is None:
            return AgentAction(
                AgentActionType.WEB_SEARCH,
                tool_name="web_search",
                arguments={"query": state.query, "limit": self.search_limit},
            )
        if search_result.success:
            attempted = {
                call.arguments.get("url")
                for call in state.tool_calls
                if call.tool_name == "fetch_url"
            }
            for item in (search_result.output or {}).get("results", []):
                url = item.get("url") if isinstance(item, dict) else None
                if isinstance(url, str) and url not in attempted:
                    return AgentAction(
                        AgentActionType.FETCH_URL,
                        tool_name="fetch_url",
                        arguments={
                            "url": url,
                            "title": item.get("title", "Untitled source"),
                            "summary": item.get("summary", ""),
                        },
                    )
        return action
