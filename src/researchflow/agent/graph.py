"""Fixed state-graph orchestration for bounded research runs.

This deliberately models one research workflow rather than providing a general
graph framework.  Nodes produce patches; the small engine owns routing and
patch application so the nodes can later be adapted to a graph runtime.
"""

import re
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from enum import StrEnum
from typing import Any, Protocol

from pydantic import Field

from researchflow.agent.evidence_policy import evaluate_evidence_policy
from researchflow.agent.planner import RulePlanner
from researchflow.agent.selector import StateSelector
from researchflow.agent.summarizer import ExtractiveSummarizer, SummaryGenerationStatus
from researchflow.domain import (
    AgentState,
    AgentStatus,
    DecisionTrace,
    ExecutionTrace,
    ToolCall,
    ToolResult,
)
from researchflow.domain.models import DomainModel
from researchflow.execution import ToolExecutor
from researchflow.tools import ToolContext
from researchflow.tools.offline import ReadDocumentOutput
from researchflow.tools.web import WebSource


class GraphNode(StrEnum):
    INITIALIZE = "initialize"
    PLAN = "plan"
    RETRIEVE = "retrieve"
    ASSESS_EVIDENCE = "assess_evidence"
    REPLAN = "replan"
    READ_SOURCES = "read_sources"
    SYNTHESIZE = "synthesize"
    VERIFY = "verify"
    SAVE = "save"
    FINISH = "finish"


class GraphEndReason(StrEnum):
    COMPLETED = "completed"
    NO_RESULTS = "no_results"
    MAX_STEPS = "max_steps"


class EvidenceStatus(StrEnum):
    SUFFICIENT = "sufficient"
    PARTIAL = "partial"
    INSUFFICIENT = "insufficient"


class GraphCandidate(DomainModel):
    """A normalized source candidate, ordered independently of task timing."""

    source_type: str
    locator: str
    title: str
    summary: str = ""


class AgentGraphState(DomainModel):
    """Serializable state owned by the fixed graph orchestration."""

    run_id: str
    query: str
    answer_target: str = ""
    answer_language: str = ""
    current_node: GraphNode = GraphNode.INITIALIZE
    agent: AgentState | None = None
    candidates: list[GraphCandidate] = Field(default_factory=list)
    documents: list[ReadDocumentOutput] = Field(default_factory=list)
    web_sources: list[WebSource] = Field(default_factory=list)
    rejected_sources: list[str] = Field(default_factory=list)
    attempted_candidates: list[str] = Field(default_factory=list)
    evidence_status: EvidenceStatus | None = None
    evidence_gaps: list[str] = Field(default_factory=list)
    evidence_policy: str | None = None
    accepted_source_count: int = 0
    required_source_count: int = 0
    official_complete_source_id: str | None = None
    generation_mode: str | None = None
    generation_fallback_reason: str | None = None
    generation_error_type: str | None = None
    replans: int = 0
    max_replans: int = 1
    read_limit: int = 5
    replan_reason: str | None = None
    node_steps: int = 0
    end_reason: GraphEndReason | None = None


class AgentOrchestrator(Protocol):
    """Stable boundary consumed by CLI orchestration selection."""

    def run(self, query: str, context: ToolContext) -> AgentState:
        """Run one research request."""
        ...


GraphEventSink = Callable[[str, dict[str, object]], None]


class GraphAgentRunner:
    """Run the fixed research graph while retaining the old runner API."""

    def __init__(
        self,
        planner: RulePlanner,
        selector: StateSelector,
        summarizer: ExtractiveSummarizer,
        executor: ToolExecutor,
        max_steps: int = 10,
        candidate_limit: int = 10,
        read_limit: int = 5,
        max_concurrency: int = 3,
        max_replans: int = 1,
        official_domains: tuple[str, ...] = (),
    ) -> None:
        if min(max_steps, candidate_limit, read_limit, max_concurrency) < 1:
            raise ValueError("graph limits must be positive")
        if max_replans < 0:
            raise ValueError("max_replans cannot be negative")
        self._planner = planner
        self._selector = selector  # Retained for the shared orchestration contract.
        self._summarizer = summarizer
        self._executor = executor
        self._max_steps = max_steps
        self._candidate_limit = candidate_limit
        self._read_limit = read_limit
        self._max_concurrency = max_concurrency
        self._max_replans = max_replans
        self._official_domains = official_domains
        self._call_count = 0

    def run(
        self,
        query: str,
        context: ToolContext,
        *,
        answer_target: str | None = None,
        answer_language: str = "",
        event_sink: GraphEventSink | None = None,
    ) -> AgentState:
        state = AgentGraphState(
            run_id=context.run_id,
            query=query,
            answer_target=answer_target or query,
            answer_language=answer_language,
            max_replans=self._max_replans,
            read_limit=self._read_limit,
        )
        while True:
            if (
                state.current_node is not GraphNode.FINISH
                and state.node_steps >= self._max_steps
            ):
                state = state.model_copy(
                    update={
                        "current_node": GraphNode.FINISH,
                        "end_reason": GraphEndReason.MAX_STEPS,
                    }
                )
            node = state.current_node
            self._emit(event_sink, "graph_node_started", {"node": node.value})
            update = self._node_update(node, state, context)
            previous = state
            state = state.model_copy(
                update={**update, "node_steps": state.node_steps + 1}
            )
            next_node = self.route(state)
            self._record_graph(node, next_node, state, context)
            self._publish_node_events(event_sink, node, previous, state, next_node)
            if next_node is None:
                if state.agent is None:
                    raise RuntimeError("graph finished without agent state")
                return state.agent
            state = state.model_copy(update={"current_node": next_node})

    @staticmethod
    def route(state: AgentGraphState) -> GraphNode | None:
        """Choose an edge from state only; never execute tools here."""
        node = state.current_node
        if node is GraphNode.INITIALIZE:
            return GraphNode.PLAN
        if node is GraphNode.PLAN:
            return GraphNode.RETRIEVE
        if node is GraphNode.RETRIEVE:
            return GraphNode.ASSESS_EVIDENCE
        if node is GraphNode.ASSESS_EVIDENCE:
            if state.evidence_status is EvidenceStatus.SUFFICIENT:
                return GraphNode.SYNTHESIZE
            if GraphAgentRunner._has_read_budget(state):
                return GraphNode.READ_SOURCES
            if state.candidates:
                return GraphNode.SYNTHESIZE
            if state.replans < state.max_replans and state.end_reason is None:
                return GraphNode.REPLAN
            return GraphNode.SYNTHESIZE
        if node is GraphNode.REPLAN:
            return GraphNode.RETRIEVE
        if node is GraphNode.READ_SOURCES:
            return GraphNode.ASSESS_EVIDENCE
        if node is GraphNode.SYNTHESIZE:
            return GraphNode.VERIFY
        if node is GraphNode.VERIFY:
            return GraphNode.SAVE
        if node is GraphNode.SAVE:
            return GraphNode.FINISH
        return None

    def _node_update(
        self, node: GraphNode, state: AgentGraphState, context: ToolContext
    ) -> dict[str, Any]:
        return {
            GraphNode.INITIALIZE: self._initialize,
            GraphNode.PLAN: self._plan,
            GraphNode.RETRIEVE: self._retrieve,
            GraphNode.ASSESS_EVIDENCE: self._assess_evidence,
            GraphNode.REPLAN: self._replan,
            GraphNode.READ_SOURCES: self._read_sources,
            GraphNode.SYNTHESIZE: self._synthesize,
            GraphNode.VERIFY: self._verify,
            GraphNode.SAVE: self._save,
            GraphNode.FINISH: self._finish,
        }[node](state, context)

    def _initialize(self, state: AgentGraphState, _: ToolContext) -> dict[str, Any]:
        return {
            "agent": AgentState(
                run_id=state.run_id, query=state.query, status=AgentStatus.RUNNING
            )
        }

    def _plan(self, state: AgentGraphState, context: ToolContext) -> dict[str, Any]:
        agent = self._agent(state)
        agent.plan = self._planner.create_plan(state.query)
        self._record_decision(agent, self._planner, context)
        return {"agent": agent}

    def _retrieve(self, state: AgentGraphState, context: ToolContext) -> dict[str, Any]:
        calls = [
            ("search_documents", {"query": state.query, "limit": self._candidate_limit})
        ]
        if self._has_web_plan(state):
            calls.append(
                ("web_search", {"query": state.query, "limit": self._candidate_limit})
            )
        outcomes = self._parallel_calls(state, context, calls)
        candidates: list[GraphCandidate] = []
        for call, result, trace in outcomes:
            self._append_tool(self._agent(state), call, result, trace)
            if result.success and call.tool_name == "search_documents":
                candidates.extend(self._local_candidates(result.output))
            if result.success and call.tool_name == "web_search":
                candidates.extend(self._web_candidates(result.output))
        return {
            "agent": self._agent(state),
            "candidates": self._deduplicate(candidates),
        }

    def _assess_evidence(
        self, state: AgentGraphState, _: ToolContext
    ) -> dict[str, Any]:
        assessment = evaluate_evidence_policy(
            state.query,
            [*state.documents, *state.web_sources],
            official_domains=self._official_domains,
        )
        unread_candidates = len(state.attempted_candidates) < len(state.candidates)
        if assessment.sufficient:
            status = EvidenceStatus.SUFFICIENT
        elif state.documents or state.web_sources or unread_candidates:
            status = EvidenceStatus.PARTIAL
        else:
            status = EvidenceStatus.INSUFFICIENT
        return {
            "evidence_status": status,
            "evidence_gaps": list(assessment.gaps),
            "evidence_policy": assessment.policy.name,
            "accepted_source_count": assessment.accepted_source_count,
            "required_source_count": assessment.required_source_count,
            "official_complete_source_id": assessment.official_complete_source_id,
            "replan_reason": None if assessment.sufficient else "insufficient_evidence",
        }

    def _replan(self, state: AgentGraphState, context: ToolContext) -> dict[str, Any]:
        agent = self._agent(state)
        agent.plan = self._planner.create_plan(state.query)
        self._record_decision(agent, self._planner, context)
        return {"agent": agent, "replans": state.replans + 1}

    def _read_sources(
        self, state: AgentGraphState, context: ToolContext
    ) -> dict[str, Any]:
        calls = []
        attempted = set(state.attempted_candidates)
        remaining_budget = max(0, self._read_limit - len(state.attempted_candidates))
        unread = [
            candidate
            for candidate in state.candidates
            if candidate.locator not in attempted
        ][:remaining_budget]
        for candidate in unread:
            if candidate.source_type == "local":
                calls.append(("read_document", {"path": candidate.locator}))
            else:
                calls.append(
                    (
                        "fetch_url",
                        {
                            "url": candidate.locator,
                            "title": candidate.title,
                            "summary": candidate.summary,
                        },
                    )
                )
        documents = [*state.documents]
        web_sources = [*state.web_sources]
        rejected = [*state.rejected_sources]
        for call, result, trace in self._parallel_calls(state, context, calls):
            self._append_tool(self._agent(state), call, result, trace)
            if not result.success:
                locator = call.arguments.get("path") or call.arguments.get("url")
                reason = result.error_message or "unknown read failure"
                rejected.append(f"{locator} — {reason}")
                continue
            if call.tool_name == "read_document":
                document = ReadDocumentOutput.model_validate(result.output)
                rejection = self._evidence_rejection_reason(
                    state.query, document.title, document.content
                )
                if rejection is None:
                    documents.append(document)
                else:
                    rejected.append(f"{document.path} — {rejection}")
            else:
                source = WebSource.model_validate(result.output)
                rejection = self._evidence_rejection_reason(
                    state.query, source.title, source.content
                )
                if rejection is None:
                    web_sources.append(source)
                else:
                    rejected.append(f"{source.url} — {rejection}")
        return {
            "agent": self._agent(state),
            "documents": documents,
            "web_sources": web_sources,
            "rejected_sources": rejected,
            "attempted_candidates": [
                *state.attempted_candidates,
                *(candidate.locator for candidate in unread),
            ],
        }

    @staticmethod
    def _has_read_budget(state: AgentGraphState) -> bool:
        return len(state.attempted_candidates) < state.read_limit and len(
            state.attempted_candidates
        ) < len(state.candidates)

    def _synthesize(
        self, state: AgentGraphState, context: ToolContext
    ) -> dict[str, Any]:
        agent = self._agent(state)
        if state.evidence_status is not EvidenceStatus.SUFFICIENT:
            agent.final_answer = self._read_failure_report(state)
            return {
                "agent": agent,
                "end_reason": GraphEndReason.NO_RESULTS,
            }
        answer = self._summarizer.summarize(
            state.answer_target,
            state.documents,
            state.web_sources,
            answer_language=state.answer_language,
        )
        generation = getattr(self._summarizer, "last_generation_status", None)
        if not isinstance(generation, SummaryGenerationStatus):
            generation = SummaryGenerationStatus(mode="extractive")
        agent.final_answer = "\n".join(
            [answer, "", "## 回答生成状态", "", generation.render()]
        )
        self._record_decision(agent, self._summarizer, context)
        return {
            "agent": agent,
            "generation_mode": generation.mode,
            "generation_fallback_reason": generation.fallback_reason,
            "generation_error_type": generation.error_type,
        }

    @staticmethod
    def _read_failure_report(state: AgentGraphState) -> str:
        result_message = (
            "没有成功读取任何候选来源，因此无法提供经来源验证的事实性回答。"
            if not (state.documents or state.web_sources)
            else "证据不足：已读取来源未满足当前问题的证据门槛。"
        )
        candidates = [
            f"- {candidate.title} — {candidate.locator}"
            for candidate in state.candidates
        ] or ["- 无"]
        failures = [f"- {item}" for item in state.rejected_sources] or ["- 无"]
        search_failures = [
            f"- {result.tool_name}: {result.error_message or 'unknown search failure'}"
            for result in state.agent.tool_results
            if not result.success
            and result.tool_name in {"web_search", "search_documents"}
        ] or ["- 无"]
        return "\n".join(
            [
                "# 研究报告",
                "",
                "## 问题",
                "",
                state.query,
                "",
                "## 结果",
                "",
                result_message,
                "",
                "## 搜索候选（未读取正文）",
                "",
                *candidates,
                "",
                "## 搜索失败原因",
                "",
                *search_failures,
                "",
                "## 正文读取失败原因",
                "",
                *failures,
            ]
        )

    def _verify(self, state: AgentGraphState, _: ToolContext) -> dict[str, Any]:
        # Inputs are constructed only from successful result validation above.
        return {}

    def _save(self, state: AgentGraphState, context: ToolContext) -> dict[str, Any]:
        agent = self._agent(state)
        call, result, trace = self._one_call(
            state,
            context,
            "save_note",
            {
                "path": f"notes/{state.run_id}.md",
                "content": agent.final_answer or "# 研究报告",
                "overwrite": False,
            },
        )
        self._append_tool(agent, call, result, trace)
        return {"agent": agent}

    def _finish(self, state: AgentGraphState, _: ToolContext) -> dict[str, Any]:
        agent = self._agent(state)
        if state.end_reason is GraphEndReason.MAX_STEPS:
            agent.final_answer = "Agent 已达到最大步骤限制，研究流程已停止。"
        elif agent.final_answer is None:
            agent.final_answer = self._summarizer.summarize(state.query, [], [])
        agent.status = AgentStatus.COMPLETED
        return {
            "agent": agent,
            "end_reason": state.end_reason or GraphEndReason.COMPLETED,
        }

    def _parallel_calls(
        self,
        state: AgentGraphState,
        context: ToolContext,
        specs: list[tuple[str, dict[str, Any]]],
    ) -> list[tuple[ToolCall, ToolResult, ExecutionTrace]]:
        calls = [self._new_call(state, name, args) for name, args in specs]
        if len(calls) <= 1:
            return [self._execute(call, context) for call in calls]
        with ThreadPoolExecutor(
            max_workers=min(self._max_concurrency, len(calls))
        ) as pool:
            futures = [pool.submit(self._execute, call, context) for call in calls]
            return [future.result() for future in futures]

    def _one_call(
        self,
        state: AgentGraphState,
        context: ToolContext,
        name: str,
        arguments: dict[str, Any],
    ) -> tuple[ToolCall, ToolResult, ExecutionTrace]:
        return self._execute(self._new_call(state, name, arguments), context)

    def _execute(
        self, call: ToolCall, context: ToolContext
    ) -> tuple[ToolCall, ToolResult, ExecutionTrace]:
        result, trace = self._executor.execute_with_trace(call, context)
        return call, result, trace

    def _new_call(
        self, state: AgentGraphState, name: str, arguments: dict[str, Any]
    ) -> ToolCall:
        self._call_count += 1
        return ToolCall(
            call_id=f"{state.run_id}-{self._call_count}",
            tool_name=name,
            arguments=arguments,
        )

    @staticmethod
    def _is_relevant(query: str, title: str, content: str) -> bool:
        return (
            GraphAgentRunner._evidence_rejection_reason(query, title, content) is None
        )

    @staticmethod
    def _evidence_rejection_reason(query: str, title: str, content: str) -> str | None:
        text = f"{title}\n{content}".casefold()
        if "编译器" in query or "compiler" in query.casefold():
            if any(
                term in text for term in ("编译器", "compiler", "gcc", "clang", "msvc")
            ):
                return None
            return "missing_compiler_evidence"
        if any(term in query for term in ("代表作", "代表电影", "作品有哪些")):
            if any(term in text for term in ("电影", "作品", "主演", "代表作")):
                return None
            return "missing_work_evidence"
        if any(term in query for term in ("出生日期", "年龄", "多大")):
            if any(term in text for term in ("出生", "生于", "born")):
                return None
            return "missing_birth_evidence"
        subject = GraphAgentRunner._person_subject(query)
        if subject:
            if subject.casefold() not in text:
                return "missing_subject_coverage"
            if not any(
                term in text
                for term in (
                    "作家",
                    "小说家",
                    "演员",
                    "科学家",
                    "记者",
                    "出生",
                    "国籍",
                    "代表作",
                    "writer",
                    "novelist",
                    "actor",
                    "scientist",
                    "journalist",
                    "born",
                    "nationality",
                    "known for",
                )
            ):
                return "missing_person_identity_evidence"
            if len(content.strip()) < 24:
                return "insufficient_person_profile_content"
        return None

    @staticmethod
    def _person_subject(query: str) -> str:
        match = re.fullmatch(
            r"\s*(.+?)(?:是谁|是誰|who\s+is)\s*[?？.!！]*\s*", query, re.I
        )
        return match.group(1).strip(" ，。？?!！") if match else ""

    @staticmethod
    def _local_candidates(output: Any) -> list[GraphCandidate]:
        return [
            GraphCandidate(
                source_type="local",
                locator=item["path"],
                title=item.get("title", item["path"]),
            )
            for item in (output or {}).get("hits", [])
            if isinstance(item, dict) and isinstance(item.get("path"), str)
        ]

    @staticmethod
    def _web_candidates(output: Any) -> list[GraphCandidate]:
        return [
            GraphCandidate(
                source_type="web",
                locator=item["url"],
                title=item.get("title", item["url"]),
                summary=item.get("summary", ""),
            )
            for item in (output or {}).get("results", [])
            if isinstance(item, dict) and isinstance(item.get("url"), str)
        ]

    def _deduplicate(self, candidates: list[GraphCandidate]) -> list[GraphCandidate]:
        unique: dict[tuple[str, str], GraphCandidate] = {}
        for candidate in candidates:
            unique.setdefault((candidate.source_type, candidate.locator), candidate)
        return list(unique.values())[: self._candidate_limit]

    @staticmethod
    def _has_web_plan(state: AgentGraphState) -> bool:
        return bool(
            state.agent
            and state.agent.plan
            and any(step.tool_name == "web_search" for step in state.agent.plan.steps)
        )

    @staticmethod
    def _agent(state: AgentGraphState) -> AgentState:
        if state.agent is None:
            raise RuntimeError("graph node requires initialized agent state")
        return state.agent

    @staticmethod
    def _append_tool(
        agent: AgentState, call: ToolCall, result: ToolResult, trace: ExecutionTrace
    ) -> None:
        agent.tool_calls.append(call)
        agent.tool_results.append(result)
        agent.traces.append(trace)

    def _record_decision(
        self, agent: AgentState, component: object, context: ToolContext
    ) -> None:
        decision = getattr(component, "last_decision", None)
        if decision is None:
            return
        trace = DecisionTrace.model_validate(asdict(decision))
        agent.decision_traces.append(trace)
        record = getattr(self._executor, "record_decision", None)
        if callable(record):
            record(trace, context)

    def _record_graph(
        self,
        node: GraphNode,
        next_node: GraphNode | None,
        state: AgentGraphState,
        context: ToolContext,
    ) -> None:
        record = getattr(self._executor, "record_graph", None)
        if callable(record):
            record(
                {
                    "node": node.value,
                    "node_status": "completed",
                    "edge": None if next_node is None else next_node.value,
                    "replan_reason": state.replan_reason,
                    "step_count": state.node_steps,
                    "source_count": len(state.documents) + len(state.web_sources),
                    "rejected_sources": state.rejected_sources,
                    "end_reason": None
                    if state.end_reason is None
                    else state.end_reason.value,
                    "generation_mode": state.generation_mode,
                    "generation_fallback_reason": state.generation_fallback_reason,
                    "generation_error_type": state.generation_error_type,
                },
                context,
            )

    @staticmethod
    def _emit(
        event_sink: GraphEventSink | None,
        event_type: str,
        data: dict[str, object],
    ) -> None:
        if event_sink is not None:
            event_sink(event_type, data)

    def _publish_node_events(
        self,
        event_sink: GraphEventSink | None,
        node: GraphNode,
        previous: AgentGraphState,
        state: AgentGraphState,
        next_node: GraphNode | None,
    ) -> None:
        self._emit(
            event_sink,
            "graph_node_finished",
            {
                "node": node.value,
                "next_node": None if next_node is None else next_node.value,
            },
        )
        if node is GraphNode.RETRIEVE:
            self._emit(
                event_sink,
                "search_completed",
                {"candidate_count": len(state.candidates)},
            )
            for candidate in state.candidates:
                self._emit(
                    event_sink,
                    "candidate_selected",
                    {
                        "source_id": candidate.locator,
                        "title": candidate.title,
                        "kind": candidate.source_type,
                    },
                )
        if node is GraphNode.READ_SOURCES:
            for document in state.documents[len(previous.documents) :]:
                self._emit(
                    event_sink,
                    "source_read",
                    {
                        "source_id": document.path,
                        "title": document.title,
                        "kind": "document",
                        "content_length": len(document.content),
                    },
                )
            for source in state.web_sources[len(previous.web_sources) :]:
                self._emit(
                    event_sink,
                    "source_read",
                    {
                        "source_id": source.url,
                        "title": source.title,
                        "kind": "web",
                        "content_length": len(source.content),
                    },
                )
            for rejected in state.rejected_sources[len(previous.rejected_sources) :]:
                source_id, separator, reason = rejected.rpartition(" — ")
                self._emit(
                    event_sink,
                    "source_rejected",
                    {
                        "source_id": source_id if separator else rejected,
                        "reason": reason if separator else "rejected",
                    },
                )
        if node is GraphNode.ASSESS_EVIDENCE and state.evidence_status is not None:
            self._emit(
                event_sink,
                "evidence_assessed",
                {
                    "status": state.evidence_status.value,
                    "gaps": state.evidence_gaps,
                    "policy": state.evidence_policy,
                    "accepted_source_count": state.accepted_source_count,
                    "required_source_count": state.required_source_count,
                    "official_complete_source_id": state.official_complete_source_id,
                },
            )
        if node is GraphNode.SYNTHESIZE:
            self._emit(
                event_sink,
                "generation_status",
                {"mode": state.generation_mode},
            )
        if node is GraphNode.FINISH:
            self._emit(
                event_sink,
                "run_completed",
                {
                    "end_reason": None
                    if state.end_reason is None
                    else state.end_reason.value,
                },
            )
