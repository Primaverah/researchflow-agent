"""State-driven action selection for the offline agent."""

from researchflow.agent.models import AgentAction, AgentActionType
from researchflow.domain import AgentState, ToolResult


class StateSelector:
    """Select the next deterministic action from accumulated state."""

    def __init__(self, search_limit: int = 5, max_documents: int = 3) -> None:
        if not 1 <= search_limit <= 100:
            raise ValueError("search_limit must be between 1 and 100")
        if max_documents < 1:
            raise ValueError("max_documents must be positive")
        self.search_limit = search_limit
        self.max_documents = max_documents

    def select(self, state: AgentState) -> AgentAction:
        """Choose exactly one next action without performing side effects."""
        search_result = self._latest_result(state, "search_documents")
        if search_result is None:
            return AgentAction(
                AgentActionType.SEARCH,
                tool_name="search_documents",
                arguments={"query": state.query, "limit": self.search_limit},
            )
        if not search_result.success:
            return AgentAction(AgentActionType.FINISH)

        hits = (search_result.output or {}).get("hits", [])[: self.search_limit]
        successful_reads = [
            result
            for result in state.tool_results
            if result.tool_name == "read_document" and result.success
        ]
        attempted_paths = {
            call.arguments.get("path")
            for call in state.tool_calls
            if call.tool_name == "read_document"
        }
        if len(successful_reads) < self.max_documents:
            for hit in hits:
                path = hit.get("path")
                if isinstance(path, str) and path not in attempted_paths:
                    return AgentAction(
                        AgentActionType.READ,
                        tool_name="read_document",
                        arguments={"path": path},
                    )

        if state.final_answer is None:
            return AgentAction(AgentActionType.SUMMARIZE)
        if self._latest_result(state, "save_note") is None:
            return AgentAction(
                AgentActionType.SAVE,
                tool_name="save_note",
                arguments={
                    "path": f"notes/{state.run_id}.md",
                    "content": state.final_answer,
                    "overwrite": False,
                },
            )
        return AgentAction(AgentActionType.FINISH)

    @staticmethod
    def _latest_result(state: AgentState, tool_name: str) -> ToolResult | None:
        for result in reversed(state.tool_results):
            if result.tool_name == tool_name:
                return result
        return None
