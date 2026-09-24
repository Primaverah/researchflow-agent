"""Synchronous rule-driven agent execution loop."""

from datetime import UTC, datetime

from researchflow.agent.models import AgentAction, AgentActionType
from researchflow.agent.planner import RulePlanner
from researchflow.agent.selector import StateSelector
from researchflow.agent.summarizer import ExtractiveSummarizer
from researchflow.domain import (
    AgentState,
    AgentStatus,
    PlanStep,
    PlanStepStatus,
    ToolCall,
    ToolResult,
)
from researchflow.execution import ToolExecutor
from researchflow.tools import ToolContext
from researchflow.tools.offline import ReadDocumentOutput

MAX_STEPS_MESSAGE = "Agent 已达到最大步骤限制，研究流程已停止。"


class AgentRunner:
    """Coordinate planning, selection, tool execution, and summarization."""

    def __init__(
        self,
        planner: RulePlanner,
        selector: StateSelector,
        summarizer: ExtractiveSummarizer,
        executor: ToolExecutor,
        max_steps: int = 10,
    ) -> None:
        if max_steps < 1:
            raise ValueError("max_steps must be positive")
        self._planner = planner
        self._selector = selector
        self._summarizer = summarizer
        self._executor = executor
        self._max_steps = max_steps

    def run(self, query: str, context: ToolContext) -> AgentState:
        """Run the bounded offline research workflow to completion."""
        plan = self._planner.create_plan(query)
        state = AgentState(
            run_id=context.run_id,
            query=query,
            status=AgentStatus.PLANNED,
            plan=plan,
        )
        state.status = AgentStatus.RUNNING
        action_count = 0
        call_count = 0

        while True:
            if action_count >= self._max_steps:
                self._fail_for_step_limit(state)
                return state
            action = self._selector.select(state)
            action_count += 1

            if action.action_type is AgentActionType.SEARCH:
                call_count += 1
                result = self._execute_tool(
                    state, action, context, call_count, "search"
                )
                if not result.success:
                    state.final_answer = self._failure_report(
                        query, f"搜索失败：{result.error_message}"
                    )
                    self._touch(state)
                continue

            if action.action_type is AgentActionType.READ:
                call_count += 1
                self._execute_tool(state, action, context, call_count, "read")
                continue

            if action.action_type is AgentActionType.SUMMARIZE:
                self._summarize(state)
                continue

            if action.action_type is AgentActionType.SAVE:
                call_count += 1
                result = self._execute_tool(state, action, context, call_count, "save")
                if not result.success:
                    state.final_answer = (
                        f"{state.final_answer}\n\n> 保存失败：{result.error_message}"
                    )
                    self._touch(state)
                continue

            state.current_step_id = None
            state.status = (
                AgentStatus.FAILED
                if self._search_failed(state)
                else AgentStatus.COMPLETED
            )
            self._touch(state)
            return state

    def _execute_tool(
        self,
        state: AgentState,
        action: AgentAction,
        context: ToolContext,
        call_count: int,
        step_id: str,
    ) -> ToolResult:
        step = self._step(state, step_id)
        if step.status is PlanStepStatus.PENDING:
            step.status = PlanStepStatus.RUNNING
        state.current_step_id = step_id
        call = ToolCall(
            call_id=f"{state.run_id}-{call_count}",
            tool_name=action.tool_name or "invalid",
            arguments=action.arguments,
        )
        result = self._executor.execute(call, context)
        state.tool_calls.append(call)
        state.tool_results.append(result)

        if step_id in {"search", "save"}:
            if result.success:
                step.result_summary = f"{action.tool_name} completed"
                step.status = PlanStepStatus.COMPLETED
            else:
                self._mark_failed(
                    state,
                    step_id,
                    result.error_message or "tool execution failed",
                )
        self._touch(state)
        return result

    def _summarize(self, state: AgentState) -> None:
        read_step = self._step(state, "read")
        documents = [
            ReadDocumentOutput.model_validate(result.output)
            for result in state.tool_results
            if result.tool_name == "read_document" and result.success
        ]
        search_result = next(
            result
            for result in state.tool_results
            if result.tool_name == "search_documents"
        )
        if not search_result.output.get("hits"):
            read_step.status = PlanStepStatus.SKIPPED
        elif documents:
            read_step.result_summary = f"read {len(documents)} document(s)"
            read_step.status = PlanStepStatus.COMPLETED
        else:
            self._mark_failed(state, "read", "所有候选文档读取失败")

        step = self._step(state, "summarize")
        step.status = PlanStepStatus.RUNNING
        state.current_step_id = "summarize"
        state.final_answer = self._summarizer.summarize(state.query, documents)
        step.result_summary = "report generated"
        step.status = PlanStepStatus.COMPLETED
        self._touch(state)

    def _fail_for_step_limit(self, state: AgentState) -> None:
        if state.current_step_id is not None:
            step = self._step(state, state.current_step_id)
            if step.status is PlanStepStatus.RUNNING:
                self._mark_failed(state, state.current_step_id, MAX_STEPS_MESSAGE)
        state.final_answer = MAX_STEPS_MESSAGE
        state.status = AgentStatus.FAILED
        self._touch(state)

    @staticmethod
    def _failure_report(query: str, message: str) -> str:
        return "\n".join(
            [
                "# 研究报告",
                "",
                "## 问题",
                "",
                query,
                "",
                "## 汇总",
                "",
                f"- {message}",
                "",
                "## 来源",
                "",
                "- 无",
            ]
        )

    @staticmethod
    def _search_failed(state: AgentState) -> bool:
        return any(
            result.tool_name == "search_documents" and not result.success
            for result in state.tool_results
        )

    @staticmethod
    def _step(state: AgentState, step_id: str) -> PlanStep:
        if state.plan is None:
            raise RuntimeError("agent state has no plan")
        return next(step for step in state.plan.steps if step.step_id == step_id)

    @staticmethod
    def _mark_failed(state: AgentState, step_id: str, message: str) -> None:
        if state.plan is None:
            raise RuntimeError("agent state has no plan")
        for index, step in enumerate(state.plan.steps):
            if step.step_id == step_id:
                state.plan.steps[index] = PlanStep.model_validate(
                    {
                        **step.model_dump(),
                        "status": PlanStepStatus.FAILED,
                        "error_message": message,
                    }
                )
                return
        raise RuntimeError(f"plan step '{step_id}' does not exist")

    @staticmethod
    def _touch(state: AgentState) -> None:
        state.updated_at = datetime.now(UTC)
