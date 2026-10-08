"""Tests for the fixed state-graph research orchestration."""

from pathlib import Path

from researchflow.agent import (
    AgentGraphState,
    EvidenceStatus,
    ExtractiveSummarizer,
    GraphAgentRunner,
    GraphEndReason,
    GraphNode,
    RulePlanner,
    StateSelector,
    WebRulePlanner,
)
from researchflow.agent.llm_components import LLMDecision
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


class DecisionPlanner(RulePlanner):
    def create_plan(self, query: str):
        self.last_decision = LLMDecision(
            component="planner",
            model="test-model",
            input_tokens=1,
            output_tokens=1,
            finish_reason="stop",
        )
        return super().create_plan(query)


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


def test_graph_records_dataclass_llm_decision_without_model_dump(
    tmp_path: Path,
) -> None:
    executor = FakeExecutor(
        lambda call: make_result(
            call,
            output={"hits": []}
            if call.tool_name == "search_documents"
            else {"path": "notes/graph.md", "char_count": 1},
        )
    )

    state = GraphAgentRunner(
        DecisionPlanner(), StateSelector(), ExtractiveSummarizer(), executor
    ).run("decision", make_context(tmp_path))

    assert state.decision_traces[0].component == "planner"


def test_graph_reports_safe_summary_generation_status(tmp_path: Path) -> None:
    def handler(call: ToolCall) -> ToolResult:
        if call.tool_name == "search_documents":
            return make_result(
                call, output={"hits": [{"path": "evidence.md", "title": "Evidence"}]}
            )
        if call.tool_name == "read_document":
            return make_result(
                call,
                output={
                    "path": "evidence.md",
                    "title": "Evidence",
                    "content": "Validated source content.",
                    "char_count": 25,
                },
            )
        return make_result(call, output={"path": "notes/graph.md", "char_count": 1})

    executor = FakeExecutor(handler)
    state = GraphAgentRunner(
        RulePlanner(), StateSelector(), ExtractiveSummarizer(), executor
    ).run("proof", make_context(tmp_path))

    assert "## 回答生成状态\n\nextractive" in state.final_answer
    synthesis_event = next(
        event for event in executor.events if event["node"] == "synthesize"
    )
    assert synthesis_event["generation_mode"] == "extractive"


def test_graph_event_sink_emits_safe_source_metadata_not_body(tmp_path: Path) -> None:
    def handler(call: ToolCall) -> ToolResult:
        if call.tool_name == "search_documents":
            return make_result(
                call, output={"hits": [{"path": "evidence.md", "title": "Evidence"}]}
            )
        if call.tool_name == "read_document":
            return make_result(
                call,
                output={
                    "path": "evidence.md",
                    "title": "Evidence",
                    "content": "This is source-only content.",
                    "char_count": 28,
                },
            )
        return make_result(call, output={"path": "notes/graph.md", "char_count": 1})

    events: list[tuple[str, dict[str, object]]] = []
    GraphAgentRunner(
        RulePlanner(), StateSelector(), ExtractiveSummarizer(), FakeExecutor(handler)
    ).run(
        "proof",
        make_context(tmp_path),
        event_sink=lambda event_type, data: events.append((event_type, data)),
    )

    source_event = next(
        data for event_type, data in events if event_type == "source_read"
    )
    assert source_event["content_length"] == 28
    assert "content" not in source_event
    assert "This is source-only content." not in str(events)


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
    assert "没有成功读取任何候选来源" in state.final_answer
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


def test_compiler_page_requires_compiler_evidence_not_only_subject() -> None:
    assert (
        GraphAgentRunner._is_relevant(
            "C和C++都使用哪些常见编译器", "C和C++", "C和C++是编程语言。"
        )
        is False
    )


def test_person_profile_requires_biographical_evidence_not_a_name_mention() -> None:
    unrelated = GraphAgentRunner._evidence_rejection_reason(
        "村上春树是谁",
        "读者讨论",
        "读者讨论村上春树小说中的城市意象和阅读感受。",
    )
    biography = GraphAgentRunner._evidence_rejection_reason(
        "村上春树是谁",
        "村上春树简介",
        "村上春树是日本小说家，代表作被翻译为多种语言并在世界各地出版。",
    )

    assert unrelated == "missing_person_identity_evidence"
    assert biography is None
    assert (
        GraphAgentRunner._is_relevant(
            "C和C++都使用哪些常见编译器",
            "C/C++ compilers",
            "GCC、Clang 和 MSVC 都支持 C++。",
        )
        is True
    )


def test_candidate_deduplication_keeps_provider_order_and_limit(tmp_path: Path) -> None:
    runner = GraphAgentRunner(
        RulePlanner(),
        StateSelector(),
        ExtractiveSummarizer(),
        FakeExecutor(lambda _: None),
        candidate_limit=2,
    )
    candidates = [
        runner._web_candidates(
            {
                "results": [
                    {"url": "https://example.com/z", "title": "Z"},
                    {"url": "https://example.com/a", "title": "A"},
                    {"url": "https://example.com/z", "title": "Duplicate"},
                ]
            }
        )
    ][0]

    assert [item.locator for item in runner._deduplicate(candidates)] == [
        "https://example.com/z",
        "https://example.com/a",
    ]


def test_no_successful_relevant_read_is_insufficient_evidence(tmp_path: Path) -> None:
    def handler(call: ToolCall) -> ToolResult:
        if call.tool_name == "search_documents":
            return make_result(
                call, output={"hits": [{"path": "empty.md", "title": "Empty"}]}
            )
        if call.tool_name == "read_document":
            return make_result(call, success=False)
        return make_result(call, output={"path": "notes/graph.md", "char_count": 1})

    runner = GraphAgentRunner(
        RulePlanner(), StateSelector(), ExtractiveSummarizer(), FakeExecutor(handler)
    )
    graph_state = AgentGraphState(run_id="graph", query="evidence", max_replans=0)
    initialized = runner._initialize(graph_state, make_context(tmp_path))
    graph_state = graph_state.model_copy(update=initialized)
    retrieved = runner._retrieve(graph_state, make_context(tmp_path))
    graph_state = graph_state.model_copy(update=retrieved)
    read = runner._read_sources(graph_state, make_context(tmp_path))
    graph_state = graph_state.model_copy(update=read)

    assessment = runner._assess_evidence(graph_state, make_context(tmp_path))
    assert assessment["evidence_status"] is EvidenceStatus.INSUFFICIENT


def test_all_web_reads_failed_returns_candidate_links_and_failure_reasons(
    tmp_path: Path,
) -> None:
    def handler(call: ToolCall) -> ToolResult:
        if call.tool_name == "search_documents":
            return make_result(call, output={"hits": []})
        if call.tool_name == "web_search":
            return make_result(
                call,
                output={
                    "results": [
                        {
                            "url": "https://example.com/jackie",
                            "title": "Jackie Chan profile",
                            "summary": "Search snippet only",
                        }
                    ]
                },
            )
        if call.tool_name == "fetch_url":
            return make_result(call, success=False)
        return make_result(call, output={"path": "notes/graph.md", "char_count": 1})

    state = GraphAgentRunner(
        WebRulePlanner(), StateSelector(), ExtractiveSummarizer(), FakeExecutor(handler)
    ).run("成龙是谁", make_context(tmp_path))

    assert "没有成功读取任何候选来源" in state.final_answer
    assert "https://example.com/jackie" in state.final_answer
    assert "expected failure" in state.final_answer


def test_read_failure_tries_remaining_candidates_before_reporting_no_results(
    tmp_path: Path,
) -> None:
    def handler(call: ToolCall) -> ToolResult:
        if call.tool_name == "search_documents":
            return make_result(call, output={"hits": []})
        if call.tool_name == "web_search":
            return make_result(
                call,
                output={
                    "results": [
                        {"url": f"https://example.com/{index}", "title": str(index)}
                        for index in range(6)
                    ]
                },
            )
        if call.tool_name == "fetch_url" and call.arguments["url"].endswith("/5"):
            return make_result(
                call,
                output={
                    "url": call.arguments["url"],
                    "title": "Relevant source",
                    "summary": "",
                    "accessed_at": "2026-09-30T00:00:00Z",
                    "content": "成龙是演员。",
                },
            )
        if call.tool_name == "fetch_url":
            return make_result(call, success=False)
        return make_result(call, output={"path": "notes/graph.md", "char_count": 1})

    executor = FakeExecutor(handler)
    state = GraphAgentRunner(
        WebRulePlanner(),
        StateSelector(),
        ExtractiveSummarizer(),
        executor,
        read_limit=5,
        max_steps=12,
    ).run("成龙是谁", make_context(tmp_path))

    assert any(call.arguments.get("url", "").endswith("/5") for call in executor.calls)
    assert "https://example.com/5" in state.final_answer
