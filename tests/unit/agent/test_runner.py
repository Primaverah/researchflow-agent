"""Tests for the bounded rule-driven agent runner."""

from pathlib import Path

from researchflow.agent import (
    AgentRunner,
    ExtractiveSummarizer,
    RulePlanner,
    StateSelector,
)
from researchflow.domain import AgentStatus, PlanStepStatus, ToolCall, ToolResult
from researchflow.tools import ToolContext


class FakeExecutor:
    def __init__(self, handler) -> None:
        self.handler = handler
        self.calls: list[ToolCall] = []

    def execute(self, call: ToolCall, context: ToolContext) -> ToolResult:
        self.calls.append(call)
        return self.handler(call)


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
