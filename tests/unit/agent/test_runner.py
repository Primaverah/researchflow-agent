"""Tests for the bounded rule-driven agent runner."""

from pathlib import Path

from researchflow.agent import (
    AgentRunner,
    ExtractiveSummarizer,
    RulePlanner,
    StateSelector,
    WebRulePlanner,
    WebStateSelector,
)
from researchflow.domain import (
    AgentStatus,
    ExecutionStatus,
    ExecutionTrace,
    PlanStepStatus,
    ToolCall,
    ToolResult,
)
from researchflow.tools import ToolContext


class FakeExecutor:
    def __init__(self, handler) -> None:
        self.handler = handler
        self.calls: list[ToolCall] = []

    def execute_with_trace(
        self, call: ToolCall, context: ToolContext
    ) -> tuple[ToolResult, ExecutionTrace]:
        self.calls.append(call)
        result = self.handler(call)
        trace = ExecutionTrace(
            trace_id=f"trace-{len(self.calls)}",
            run_id=context.run_id,
            call_id=call.call_id,
            tool_name=call.tool_name,
            arguments=call.arguments,
            status=(
                ExecutionStatus.SUCCEEDED if result.success else ExecutionStatus.FAILED
            ),
            duration_ms=0,
            error_type=result.error_type,
            error_message=result.error_message,
        )
        return result, trace


def context(tmp_path: Path) -> ToolContext:
    return ToolContext(
        working_directory=tmp_path,
        output_directory=tmp_path,
        run_id="run-1",
    )


def result(call: ToolCall, *, success: bool = True, output=None) -> ToolResult:
    return ToolResult(
        call_id=call.call_id,
        tool_name=call.tool_name,
        success=success,
        output=output,
        error_type=None if success else "expected_failure",
        error_message=None if success else "expected failure",
    )


def runner(executor: FakeExecutor, max_steps: int = 10) -> AgentRunner:
    return AgentRunner(
        RulePlanner(),
        StateSelector(),
        ExtractiveSummarizer(),
        executor,  # type: ignore[arg-type]
        max_steps=max_steps,
    )


def test_runs_search_read_summary_and_save(tmp_path: Path) -> None:
    def handler(call: ToolCall) -> ToolResult:
        if call.tool_name == "search_documents":
            return result(call, output={"hits": [{"path": "a.md"}]})
        if call.tool_name == "read_document":
            return result(
                call,
                output={
                    "path": "a.md",
                    "title": "Agent",
                    "content": "Agent 使用工具。",
                    "char_count": 10,
                },
            )
        return result(call, output={"path": "notes/run-1.md", "char_count": 10})

    executor = FakeExecutor(handler)
    state = runner(executor).run("Agent 工具", context(tmp_path))

    assert state.status is AgentStatus.COMPLETED
    assert [call.tool_name for call in executor.calls] == [
        "search_documents",
        "read_document",
        "save_note",
    ]
    assert state.tool_calls == executor.calls
    assert len(state.tool_results) == 3
    assert "Agent — a.md" in state.final_answer
    assert [step.status for step in state.plan.steps] == [
        PlanStepStatus.COMPLETED,
        PlanStepStatus.COMPLETED,
        PlanStepStatus.COMPLETED,
        PlanStepStatus.COMPLETED,
    ]


def test_read_failure_continues_and_excludes_failed_source(tmp_path: Path) -> None:
    def handler(call: ToolCall) -> ToolResult:
        if call.tool_name == "search_documents":
            return result(
                call, output={"hits": [{"path": "bad.md"}, {"path": "good.md"}]}
            )
        if call.tool_name == "read_document" and call.arguments["path"] == "bad.md":
            return result(call, success=False)
        if call.tool_name == "read_document":
            return result(
                call,
                output={
                    "path": "good.md",
                    "title": "Good",
                    "content": "真实内容。",
                    "char_count": 5,
                },
            )
        return result(call, output={"path": "notes/run-1.md", "char_count": 1})

    state = runner(FakeExecutor(handler)).run("问题", context(tmp_path))

    assert state.status is AgentStatus.COMPLETED
    assert "good.md" in state.final_answer
    assert "bad.md" not in state.final_answer


def test_search_failure_stops_safely(tmp_path: Path) -> None:
    executor = FakeExecutor(lambda call: result(call, success=False))

    state = runner(executor).run("问题", context(tmp_path))

    assert state.status is AgentStatus.FAILED
    assert len(executor.calls) == 1
    assert "搜索失败" in state.final_answer
    assert "- 无" in state.final_answer


def test_all_reads_fail_but_report_is_saved(tmp_path: Path) -> None:
    def handler(call: ToolCall) -> ToolResult:
        if call.tool_name == "search_documents":
            return result(call, output={"hits": [{"path": "bad.md"}]})
        if call.tool_name == "read_document":
            return result(call, success=False)
        return result(call, output={"path": "notes/run-1.md", "char_count": 1})

    executor = FakeExecutor(handler)
    state = runner(executor).run("问题", context(tmp_path))

    assert state.status is AgentStatus.COMPLETED
    assert executor.calls[-1].tool_name == "save_note"
    assert "- 无" in state.final_answer
    assert state.plan.steps[1].status is PlanStepStatus.FAILED


def test_save_failure_preserves_report(tmp_path: Path) -> None:
    def handler(call: ToolCall) -> ToolResult:
        if call.tool_name == "search_documents":
            return result(call, output={"hits": []})
        return result(call, success=False)

    state = runner(FakeExecutor(handler)).run("问题", context(tmp_path))

    assert state.status is AgentStatus.COMPLETED
    assert "未找到相关文档" in state.final_answer
    assert "保存失败" in state.final_answer
    assert state.plan.steps[-1].status is PlanStepStatus.FAILED


def test_max_steps_stops_before_next_action(tmp_path: Path) -> None:
    executor = FakeExecutor(
        lambda call: result(call, output={"hits": [{"path": "a.md"}]})
    )

    state = runner(executor, max_steps=1).run("问题", context(tmp_path))

    assert state.status is AgentStatus.FAILED
    assert len(executor.calls) == 1
    assert "最大步骤" in state.final_answer
    assert state.current_step_id is None
    assert state.plan.steps[0].status is PlanStepStatus.COMPLETED
    assert state.plan.steps[1].status is PlanStepStatus.PENDING


def test_state_owns_the_exact_trace_for_each_tool_call(tmp_path: Path) -> None:
    def handler(call: ToolCall) -> ToolResult:
        if call.tool_name == "search_documents":
            return result(call, output={"hits": []})
        return result(call, output={"path": "notes/run-1.md", "char_count": 1})

    state = runner(FakeExecutor(handler)).run("问题", context(tmp_path))

    assert len(state.traces) == len(state.tool_calls) == 2
    assert [trace.call_id for trace in state.traces] == [
        call.call_id for call in state.tool_calls
    ]


def test_web_workflow_converges_all_plan_steps_before_finishing(tmp_path: Path) -> None:
    def handler(call: ToolCall) -> ToolResult:
        if call.tool_name == "search_documents":
            return result(call, output={"hits": []})
        if call.tool_name == "web_search":
            return result(
                call,
                output={
                    "results": [
                        {
                            "url": "https://docs.python.org/3/",
                            "title": "Python docs",
                            "summary": "",
                        }
                    ]
                },
            )
        if call.tool_name == "fetch_url":
            return result(
                call,
                output={
                    "title": "Python docs",
                    "url": "https://docs.python.org/3/",
                    "summary": "",
                    "accessed_at": "2026-01-01T00:00:00Z",
                    "content": "Official Python evidence.",
                },
            )
        return result(call, output={"path": "notes/run-1.md", "char_count": 1})

    state = AgentRunner(
        WebRulePlanner(),
        WebStateSelector(allowed_domains=("docs.python.org",)),
        ExtractiveSummarizer(allowed_domains=("docs.python.org",)),
        FakeExecutor(handler),  # type: ignore[arg-type]
    ).run("Python", context(tmp_path))

    assert state.status is AgentStatus.COMPLETED
    assert all(step.status is not PlanStepStatus.RUNNING for step in state.plan.steps)
    assert "https://docs.python.org/3/" in state.final_answer


def test_web_workflow_ends_without_sources_when_no_allowed_result_exists(
    tmp_path: Path,
) -> None:
    def handler(call: ToolCall) -> ToolResult:
        if call.tool_name == "search_documents":
            return result(call, output={"hits": []})
        if call.tool_name == "web_search":
            return result(call, output={"results": []})
        return result(call, output={"path": "notes/run-1.md", "char_count": 1})

    state = AgentRunner(
        WebRulePlanner(),
        WebStateSelector(allowed_domains=("docs.python.org",)),
        ExtractiveSummarizer(allowed_domains=("docs.python.org",)),
        FakeExecutor(handler),  # type: ignore[arg-type]
    ).run("Python", context(tmp_path))

    assert state.status is AgentStatus.COMPLETED
    assert "未找到相关文档" in state.final_answer
    assert "https://" not in state.final_answer
    assert "## 来源\n\n- 无" in state.final_answer
