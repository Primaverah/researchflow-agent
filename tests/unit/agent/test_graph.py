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
from researchflow.agent.query_understanding import build_deterministic_understanding
from researchflow.domain import (
    AgentStatus,
    ExecutionStatus,
    ExecutionTrace,
    ToolCall,
    ToolResult,
)
from researchflow.reranking.models import CandidateRanking, RankedSource
from researchflow.tools import ToolContext


class ReverseCandidateRanker:
    def rank(self, query, candidates):
        ordered = list(reversed(candidates))
        return CandidateRanking(
            source_ids=[candidate.source_id for candidate in ordered],
            records=[
                RankedSource(
                    source_id=candidate.source_id,
                    position=index,
                    scores={"rrf": 1.0 / index},
                    reason="test_ranker",
                )
                for index, candidate in enumerate(ordered, start=1)
            ],
        )


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


def test_graph_reads_hybrid_top_web_candidate_before_provider_first_result(
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
                        {"url": "https://example.test/first", "title": "First"},
                        {"url": "https://example.test/better", "title": "Better"},
                    ]
                },
            )
        if call.tool_name == "fetch_url":
            return make_result(
                call,
                output={
                    "url": call.arguments["url"],
                    "title": call.arguments["title"],
                    "summary": "",
                    "accessed_at": "2026-10-14T00:00:00Z",
                    "content": "C++ compiler evidence: GCC and Clang.",
                },
            )
        return make_result(call, output={"path": "notes/graph.md", "char_count": 1})

    executor = FakeExecutor(handler)
    GraphAgentRunner(
        WebRulePlanner(),
        StateSelector(),
        ExtractiveSummarizer(),
        executor,
        candidate_ranker=ReverseCandidateRanker(),
        max_concurrency=1,
    ).run("C++ compiler", make_context(tmp_path))

    fetched = [
        call.arguments["url"]
        for call in executor.calls
        if call.tool_name == "fetch_url"
    ]
    assert fetched[0] == "https://example.test/better"


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


def test_graph_completion_event_includes_budget_diagnostics(tmp_path: Path) -> None:
    events: list[tuple[str, dict[str, object]]] = []
    GraphAgentRunner(
        RulePlanner(),
        StateSelector(),
        ExtractiveSummarizer(),
        FakeExecutor(lambda call: make_result(call, output={"hits": []})),
        max_steps=2,
    ).run(
        "limit",
        make_context(tmp_path),
        event_sink=lambda event_type, data: events.append((event_type, data)),
    )

    completed = next(
        data for event_type, data in events if event_type == "run_completed"
    )
    assert completed["end_reason"] == "max_steps"
    assert completed["node_steps"] == 3
    assert completed["max_steps"] == 2


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


def test_c_cpp_history_rejects_candidates_without_both_subjects_before_read(
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
                            "url": "https://example.test/marx",
                            "title": "马克思主义发展历史",
                        },
                        {"url": "https://example.test/gis", "title": "GIS发展历史"},
                        {
                            "url": "https://example.test/cpp",
                            "title": "C和C++的发展历史",
                        },
                    ]
                },
            )
        if call.tool_name == "fetch_url":
            return make_result(
                call,
                output={
                    "url": call.arguments["url"],
                    "title": "C和C++的发展历史",
                    "summary": "",
                    "accessed_at": "2026-10-10T00:00:00Z",
                    "content": "C和C++的发展历史包括标准演进。",
                },
            )
        return make_result(call, output={"path": "notes/graph.md", "char_count": 1})

    executor = FakeExecutor(handler)
    understanding = build_deterministic_understanding(
        "它们的发展历史",
        prior_subject="C和C++",
        prior_query="C和C++有什么区别",
        prior_answer_target="C和C++有什么区别",
        answer_language="zh",
    )
    GraphAgentRunner(
        WebRulePlanner(), StateSelector(), ExtractiveSummarizer(), executor
    ).run("C和C++的发展历史", make_context(tmp_path), understanding=understanding)

    fetched = [
        call.arguments["url"]
        for call in executor.calls
        if call.tool_name == "fetch_url"
    ]
    assert fetched == ["https://example.test/cpp"]


def test_current_nobel_list_rejects_non_nobel_candidate_before_read(
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
                            "url": "https://example.test/microsoft",
                            "title": "Microsoft 2026 products",
                        },
                        {
                            "url": "https://example.test/nobel",
                            "title": "2026 诺贝尔奖获奖者完整名单",
                        },
                    ]
                },
            )
        if call.tool_name == "fetch_url":
            return make_result(
                call,
                output={
                    "url": call.arguments["url"],
                    "title": "2026 诺贝尔奖获奖者完整名单",
                    "summary": "",
                    "accessed_at": "2026-10-10T00:00:00Z",
                    "content": "2026 年诺贝尔奖获奖者完整名单。",
                },
            )
        return make_result(call, output={"path": "notes/graph.md", "char_count": 1})

    executor = FakeExecutor(handler)
    understanding = build_deterministic_understanding(
        "今年的诺贝尔奖获得者有哪些",
        prior_subject="",
        prior_query="",
        prior_answer_target="",
        answer_language="zh",
    )
    GraphAgentRunner(
        WebRulePlanner(), StateSelector(), ExtractiveSummarizer(), executor
    ).run(
        "今年的诺贝尔奖获得者有哪些",
        make_context(tmp_path),
        understanding=understanding,
    )

    fetched = [
        call.arguments["url"]
        for call in executor.calls
        if call.tool_name == "fetch_url"
    ]
    assert fetched == ["https://example.test/nobel"]


def test_pre_read_rejection_is_exposed_with_its_reason(tmp_path: Path) -> None:
    executor = FakeExecutor(
        lambda call: make_result(
            call,
            output={"hits": []}
            if call.tool_name == "search_documents"
            else {
                "results": [
                    {"url": "https://example.test/microsoft", "title": "Microsoft 2026"}
                ]
            },
        )
    )
    understanding = build_deterministic_understanding(
        "今年的诺贝尔奖获得者有哪些",
        prior_subject="",
        prior_query="",
        prior_answer_target="",
        answer_language="zh",
    )
    events: list[tuple[str, dict[str, object]]] = []

    GraphAgentRunner(
        WebRulePlanner(), StateSelector(), ExtractiveSummarizer(), executor
    ).run(
        understanding.standalone_query,
        make_context(tmp_path),
        understanding=understanding,
        event_sink=lambda event_type, data: events.append((event_type, data)),
    )

    assert (
        "source_rejected",
        {
            "source_id": "https://example.test/microsoft",
            "reason": "missing_required_facet:nobel_prize",
        },
    ) in events


def test_retrieve_uses_each_bounded_query_from_understanding(tmp_path: Path) -> None:
    executor = FakeExecutor(
        lambda call: make_result(
            call,
            output={"hits": []}
            if call.tool_name == "search_documents"
            else {"results": []},
        )
    )
    runner = GraphAgentRunner(
        WebRulePlanner(), StateSelector(), ExtractiveSummarizer(), executor
    )
    understanding = build_deterministic_understanding(
        "它们的发展历史",
        prior_subject="C和C++",
        prior_query="C和C++有什么区别",
        prior_answer_target="C和C++有什么区别",
        answer_language="zh",
    ).model_copy(
        update={
            "search_queries": [
                "C和C++的发展历史",
                "C语言的发展历史",
                "C++的发展历史",
            ]
        }
    )
    state = AgentGraphState(
        run_id="graph",
        query=understanding.standalone_query,
        understanding=understanding,
    )
    context = make_context(tmp_path)
    state = state.model_copy(update=runner._initialize(state, context))
    state = state.model_copy(update=runner._plan(state, context))

    runner._retrieve(state, context)

    assert [
        call.arguments["query"]
        for call in executor.calls
        if call.tool_name == "web_search"
    ] == understanding.search_queries


def test_body_admission_requires_all_resolved_subjects() -> None:
    understanding = build_deterministic_understanding(
        "它们的发展历史",
        prior_subject="C和C++",
        prior_query="C和C++有什么区别",
        prior_answer_target="C和C++有什么区别",
        answer_language="zh",
    )

    reason = GraphAgentRunner._evidence_rejection_reason(
        "C和C++的发展历史",
        "C和C++的发展历史",
        "C 语言的历史可以追溯到 1970 年代。",
        understanding=understanding,
    )

    assert reason == "missing_subject_coverage"


def test_graph_trace_records_safe_query_understanding(tmp_path: Path) -> None:
    executor = FakeExecutor(
        lambda call: make_result(
            call,
            output={"hits": []}
            if call.tool_name == "search_documents"
            else {"path": "notes/graph.md", "char_count": 1},
        )
    )
    understanding = build_deterministic_understanding(
        "它们的发展历史",
        prior_subject="C和C++",
        prior_query="C和C++有什么区别",
        prior_answer_target="C和C++有什么区别",
        answer_language="zh",
    ).model_copy(
        update={
            "decision_source": "deterministic_fallback",
            "fallback_reason": "structured_output_invalid",
        }
    )

    GraphAgentRunner(
        RulePlanner(), StateSelector(), ExtractiveSummarizer(), executor
    ).run("C和C++的发展历史", make_context(tmp_path), understanding=understanding)

    assert any(
        event.get("standalone_query") == "C和C++的发展历史"
        and event.get("resolved_subject") == "C和C++"
        and event.get("required_facets") == ["C", "C++", "history"]
        and event.get("query_rewrite_decision_source") == "deterministic_fallback"
        and event.get("query_rewrite_fallback_reason") == "structured_output_invalid"
        for event in executor.events
    )


def test_replan_replaces_queries_with_subject_preserving_facet_query(
    tmp_path: Path,
) -> None:
    runner = GraphAgentRunner(
        WebRulePlanner(),
        StateSelector(),
        ExtractiveSummarizer(),
        FakeExecutor(lambda call: make_result(call, output={"results": []})),
    )
    understanding = build_deterministic_understanding(
        "今年的诺贝尔奖获得者有哪些",
        prior_subject="",
        prior_query="",
        prior_answer_target="",
        answer_language="zh",
    )
    state = AgentGraphState(
        run_id="graph",
        query=understanding.standalone_query,
        understanding=understanding,
        evidence_gaps=["missing_required_facet:winners"],
    )
    state = state.model_copy(update=runner._initialize(state, make_context(tmp_path)))

    update = runner._replan(state, make_context(tmp_path))

    assert update["replans"] == 1
    assert update["understanding"].search_queries == [
        "今年的诺贝尔奖获得者有哪些 获奖者 官方来源"
    ]


def test_insufficient_evidence_routes_to_single_replan_after_reads() -> None:
    state = AgentGraphState(
        run_id="graph",
        query="question",
        current_node=GraphNode.ASSESS_EVIDENCE,
        candidates=[
            {
                "source_type": "web",
                "locator": "https://example.test/source",
                "title": "source",
            }
        ],
        attempted_candidates=["https://example.test/source"],
        evidence_status=EvidenceStatus.INSUFFICIENT,
        max_replans=1,
        replans=0,
    )

    assert GraphAgentRunner.route(state) is GraphNode.REPLAN


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
    assert "Search snippet only" not in state.final_answer


def test_read_limit_bounds_total_source_reads_before_insufficient_report(
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

    fetched = [call for call in executor.calls if call.tool_name == "fetch_url"]
    assert len(fetched) == 5
    assert not any(call.arguments["url"].endswith("/5") for call in fetched)
    assert "没有成功读取任何候选来源" in state.final_answer


def test_current_complete_list_does_not_summarize_one_non_official_source(
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
                            "url": "https://news.example/winners",
                            "title": "2026 诺贝尔奖获奖者名单",
                            "summary": "只可用于内部排序的搜索摘要",
                        }
                    ]
                },
            )
        if call.tool_name == "fetch_url":
            return make_result(
                call,
                output={
                    "url": "https://news.example/winners",
                    "title": "2026 诺贝尔奖获奖者完整名单",
                    "summary": "",
                    "accessed_at": "2026-01-01T00:00:00Z",
                    "content": "2026 年诺贝尔奖获奖者完整名单。",
                },
            )
        return make_result(call, output={"path": "notes/graph.md", "char_count": 1})

    state = GraphAgentRunner(
        WebRulePlanner(),
        StateSelector(),
        ExtractiveSummarizer(),
        FakeExecutor(handler),
        max_replans=0,
    ).run("今年诺贝尔奖获奖者名单", make_context(tmp_path))

    assert "证据门槛" in state.final_answer
    assert "回答生成状态" not in state.final_answer
    assert "只可用于内部排序的搜索摘要" not in state.final_answer


def test_current_list_rejects_old_year_body_and_reads_current_candidate_first(
    tmp_path: Path,
) -> None:
    runner = GraphAgentRunner(
        WebRulePlanner(),
        StateSelector(),
        ExtractiveSummarizer(),
        FakeExecutor(lambda _: None),
    )
    candidates = runner._web_candidates(
        {
            "results": [
                {"url": "https://example.test/2022", "title": "2022 诺贝尔奖"},
                {"url": "https://example.test/2026", "title": "2026 诺贝尔奖"},
            ]
        }
    )

    ordered = runner._deduplicate(candidates, "今年的诺贝尔奖目前出炉了哪些")

    assert [candidate.locator for candidate in ordered] == [
        "https://example.test/2026",
        "https://example.test/2022",
    ]
    assert (
        runner._evidence_rejection_reason(
            "今年的诺贝尔奖目前出炉了哪些",
            "2022 诺贝尔奖",
            "2022 年诺贝尔奖获奖者名单。",
        )
        == "stale_for_current_query"
    )


def test_current_complete_list_accepts_one_configured_official_complete_source(
    tmp_path: Path,
) -> None:
    runner = GraphAgentRunner(
        WebRulePlanner(),
        StateSelector(),
        ExtractiveSummarizer(),
        FakeExecutor(lambda _: None),
        official_domains=("official.example",),
    )
    state = AgentGraphState(
        run_id="official",
        query="今年诺贝尔奖获奖者名单",
        web_sources=[
            {
                "url": "https://official.example/winners",
                "title": "2026 诺贝尔奖获奖者完整名单",
                "summary": "",
                "accessed_at": "2026-01-01T00:00:00Z",
                "content": "2026 年诺贝尔奖获奖者完整名单。",
            }
        ],
    )

    assessment = runner._assess_evidence(state, make_context(tmp_path))

    assert assessment["evidence_status"] is EvidenceStatus.SUFFICIENT
    assert (
        assessment["official_complete_source_id"] == "https://official.example/winners"
    )
