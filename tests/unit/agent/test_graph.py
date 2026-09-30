"""Tests for the fixed state-graph research orchestration."""

from pathlib import Path

from researchflow.agent import (
    AgentGraphState,
    ExtractiveSummarizer,
    GraphAgentRunner,
    GraphEndReason,
    GraphNode,
    RulePlanner,
    StateSelector,
    WebRulePlanner,
)
from researchflow.domain import (
    AgentStatus,
    ExecutionStatus,
    ExecutionTrace,
    ToolCall,
    ToolResult,
)
from researchflow.tools import ToolContext


class FakeExecutor:
    def __init__(self, handler) -> None:
        self.handler = handler
        self.calls: list[ToolCall] = []
        self.events: list[dict[str, object]] = []

    def execute_with_trace(self, call: ToolCall, context: ToolContext):
        self.calls.append(call)
        result = self.handler(call)
        return result, ExecutionTrace(
            trace_id=f"trace-{len(self.calls)}",
            run_id=context.run_id,
            call_id=call.call_id,
            tool_name=call.tool_name,
            arguments=call.arguments,
            status=ExecutionStatus.SUCCEEDED
            if result.success
            else ExecutionStatus.FAILED,
            duration_ms=0,
            error_type=result.error_type,
            error_message=result.error_message,
        )

    def record_graph(self, payload: dict[str, object], context: ToolContext) -> None:
        self.events.append(payload)


def make_result(call: ToolCall, *, success: bool = True, output=None) -> ToolResult:
    return ToolResult(
        call_id=call.call_id,
        tool_name=call.tool_name,
        success=success,
        output=output,
        error_type=None if success else "expected_failure",
        error_message=None if success else "expected failure",
    )


def make_context(tmp_path: Path) -> ToolContext:
    return ToolContext(
        working_directory=tmp_path, output_directory=tmp_path, run_id="graph"
    )


def test_graph_state_is_serializable_and_routes_are_side_effect_free() -> None:
    state = AgentGraphState(run_id="graph", query="question")

    assert AgentGraphState.model_validate_json(state.model_dump_json()) == state
    assert GraphAgentRunner.route(state) is GraphNode.PLAN
    assert GraphNode.ASSESS_EVIDENCE.value == "assess_evidence"


def test_graph_deduplicates_candidates_and_uses_only_successful_reads(
    tmp_path: Path,
) -> None:
    def handler(call: ToolCall) -> ToolResult:
        if call.tool_name == "search_documents":
            return make_result(
                call,
                output={
                    "hits": [
                        {"path": "bad.md", "title": "Bad"},
                        {"path": "good.md", "title": "Good"},
                        {"path": "good.md", "title": "Duplicate"},
                    ]
                },
            )
        if call.tool_name == "read_document" and call.arguments["path"] == "bad.md":
            return make_result(call, success=False)
        if call.tool_name == "read_document":
            return make_result(
                call,
                output={
                    "path": "good.md",
                    "title": "Good",
                    "content": "verified evidence",
                    "char_count": 17,
                },
            )
        return make_result(call, output={"path": "notes/graph.md", "char_count": 1})

    executor = FakeExecutor(handler)
    state = GraphAgentRunner(
        RulePlanner(), StateSelector(), ExtractiveSummarizer(), executor
    ).run("evidence", make_context(tmp_path))

    assert state.status is AgentStatus.COMPLETED
    assert [
        call.arguments.get("path")
        for call in executor.calls
        if call.tool_name == "read_document"
    ] == ["bad.md", "good.md"]
    assert "good.md" in state.final_answer
    assert "bad.md" not in state.final_answer
    assert any(
        event["node"] == "finish" and event["end_reason"] == "completed"
        for event in executor.events
    )


def test_graph_replans_once_then_finishes_without_results(tmp_path: Path) -> None:
    calls = 0

    def handler(call: ToolCall) -> ToolResult:
        nonlocal calls
        if call.tool_name == "search_documents":
            calls += 1
            return make_result(call, output={"hits": []})
        return make_result(call, output={"path": "notes/graph.md", "char_count": 1})

    executor = FakeExecutor(handler)
    state = GraphAgentRunner(
        RulePlanner(), StateSelector(), ExtractiveSummarizer(), executor
    ).run("none", make_context(tmp_path))

    assert state.status is AgentStatus.COMPLETED
    assert calls == 2
    assert "未找到相关文档" in state.final_answer
    assert any(
        event.get("replan_reason") == "insufficient_evidence"
        for event in executor.events
    )


def test_graph_stops_normally_at_maximum_node_steps(tmp_path: Path) -> None:
    executor = FakeExecutor(lambda call: make_result(call, output={"hits": []}))
    state = GraphAgentRunner(
        RulePlanner(), StateSelector(), ExtractiveSummarizer(), executor, max_steps=2
    ).run("limit", make_context(tmp_path))

    assert state.status is AgentStatus.COMPLETED
    assert "最大步骤限制" in state.final_answer
    assert any(
        event.get("end_reason") == GraphEndReason.MAX_STEPS.value
        for event in executor.events
    )


def test_web_retrieval_runs_local_and_web_and_merges_in_stable_order(
    tmp_path: Path,
) -> None:
    def handler(call: ToolCall) -> ToolResult:
        if call.tool_name == "search_documents":
            return make_result(
                call, output={"hits": [{"path": "local.md", "title": "Local"}]}
            )
        if call.tool_name == "web_search":
            return make_result(
                call,
                output={
                    "results": [
                        {"url": "https://example.com/a", "title": "Web", "summary": "s"}
                    ]
                },
            )
        if call.tool_name == "read_document":
            return make_result(
                call,
                output={
                    "path": "local.md",
                    "title": "Local",
                    "content": "local proof",
                    "char_count": 11,
                },
            )
        if call.tool_name == "fetch_url":
            return make_result(
                call,
                output={
                    "url": "https://example.com/a",
                    "title": "Web",
                    "summary": "s",
                    "accessed_at": "2026-01-01T00:00:00Z",
                    "content": "web proof",
                },
            )
        return make_result(call, output={"path": "notes/graph.md", "char_count": 1})

    executor = FakeExecutor(handler)
    state = GraphAgentRunner(
        WebRulePlanner(), StateSelector(), ExtractiveSummarizer(), executor
    ).run("proof", make_context(tmp_path))

    assert state.status is AgentStatus.COMPLETED
    assert [call.tool_name for call in executor.calls[:2]] == [
        "search_documents",
        "web_search",
    ]
    assert (
        "local.md" in state.final_answer
        and "https://example.com/a" in state.final_answer
    )
