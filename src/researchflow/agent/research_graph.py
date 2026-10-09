"""Checkpointed LangGraph implementation of research planning and retrieval."""

import sqlite3
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypedDict

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from researchflow.agent.graph import (
    AgentGraphState,
    EvidenceStatus,
    GraphAgentRunner,
    GraphEndReason,
    GraphEventSink,
    GraphNode,
)
from researchflow.agent.planner import RulePlanner
from researchflow.agent.selector import StateSelector
from researchflow.agent.summarizer import ExtractiveSummarizer
from researchflow.domain import AgentStatus, PlanStepStatus
from researchflow.execution import ToolExecutor
from researchflow.tools import ToolContext


class ResearchGraphState(TypedDict, total=False):
    data: dict[str, Any]


class LangGraphResearchRunner:
    """Checkpoint initialize/plan/retrieve/assess nodes without runtime state."""

    def __init__(
        self,
        database: Path,
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
        self._legacy = GraphAgentRunner(
            planner,
            selector,
            summarizer,
            executor,
            max_steps=max_steps,
            candidate_limit=candidate_limit,
            read_limit=read_limit,
            max_concurrency=max_concurrency,
            max_replans=max_replans,
            official_domains=official_domains,
        )
        self._max_steps = max_steps
        self._context: ToolContext | None = None
        self._event_sink: GraphEventSink | None = None
        self._connection = sqlite3.connect(database, check_same_thread=False)
        self._graph = self._build_graph().compile(
            checkpointer=SqliteSaver(self._connection)
        )

    def run(
        self,
        query: str,
        context: ToolContext,
        *,
        thread_id: str,
        answer_target: str | None = None,
        answer_language: str = "",
        event_sink: GraphEventSink | None = None,
    ) -> AgentGraphState:
        self._context = context
        self._event_sink = event_sink
        config = {"configurable": {"thread_id": thread_id}}
        existing = self._graph.get_state(config)
        if existing.values.get("data"):
            restored = AgentGraphState.model_validate(existing.values["data"])
            if (
                restored.agent is not None
                and restored.agent.status is AgentStatus.COMPLETED
            ):
                return restored
        result = self._graph.invoke(
            {
                "data": AgentGraphState(
                    run_id=context.run_id,
                    query=query,
                    answer_target=answer_target or query,
                    answer_language=answer_language,
                ).model_dump(mode="json")
            },
            config=config,
        )
        return AgentGraphState.model_validate(result["data"])

    def get_state(self, thread_id: str) -> AgentGraphState:
        """Load a persisted planning snapshot without executing any node."""
        snapshot = self._graph.get_state({"configurable": {"thread_id": thread_id}})
        return AgentGraphState.model_validate(snapshot.values["data"])

    def _build_graph(self) -> StateGraph:
        graph = StateGraph(ResearchGraphState)
        graph.add_node("initialize", self._initialize)
        graph.add_node("plan", self._plan)
        graph.add_node("retrieve", self._retrieve)
        graph.add_node("assess", self._assess)
        graph.add_node("replan", self._replan)
        graph.add_node("read_sources", self._read_sources)
        graph.add_node("synthesize", self._synthesize)
        graph.add_node("verify", self._verify)
        graph.add_node("save", self._save)
        graph.add_node("finish", self._finish)
        graph.add_edge(START, "initialize")
        graph.add_conditional_edges(
            "initialize",
            self._route_after_initialize,
            {"plan": "plan", "finish": "finish"},
        )
        graph.add_conditional_edges(
            "plan", self._route_after_plan, {"retrieve": "retrieve", "finish": "finish"}
        )
        graph.add_conditional_edges(
            "retrieve",
            self._route_after_retrieve,
            {"assess": "assess", "finish": "finish"},
        )
        graph.add_conditional_edges(
            "read_sources",
            self._route_after_read,
            {
                "read_sources": "read_sources",
                "assess": "assess",
                "synthesize": "synthesize",
                "finish": "finish",
            },
        )
        graph.add_conditional_edges(
            "assess",
            self._route_after_assess,
            {
                "read_sources": "read_sources",
                "replan": "replan",
                "synthesize": "synthesize",
                "finish": "finish",
            },
        )
        graph.add_conditional_edges(
            "replan",
            self._route_after_replan,
            {"retrieve": "retrieve", "finish": "finish"},
        )
        graph.add_conditional_edges(
            "synthesize",
            self._route_after_synthesize,
            {"verify": "verify", "finish": "finish"},
        )
        graph.add_conditional_edges(
            "verify", self._route_after_verify, {"save": "save", "finish": "finish"}
        )
        graph.add_conditional_edges(
            "save", self._route_after_save, {"finish": "finish"}
        )
        graph.add_edge("finish", END)
        return graph

    def _state(self, value: ResearchGraphState) -> AgentGraphState:
        return AgentGraphState.model_validate(value["data"])

    def _update(
        self, state: AgentGraphState, patch: dict[str, Any]
    ) -> ResearchGraphState:
        return {"data": state.model_copy(update=patch).model_dump(mode="json")}

    def _run_node(
        self,
        node: GraphNode,
        value: ResearchGraphState,
        operation: Callable[[AgentGraphState], dict[str, Any]],
    ) -> ResearchGraphState:
        """Run a legacy-compatible node while projecting safe lifecycle events."""
        previous = self._state(value)
        self._legacy._emit(self._event_sink, "graph_node_started", {"node": node.value})
        completed = previous.model_copy(
            update={
                **operation(previous),
                "current_node": node,
                "node_steps": previous.node_steps + 1,
            }
        )
        self._project_plan_progress(node, completed)
        next_node = self._next_node(node, completed)
        if node is not GraphNode.FINISH and completed.node_steps >= self._max_steps:
            completed = completed.model_copy(
                update={"end_reason": GraphEndReason.MAX_STEPS}
            )
            next_node = GraphNode.FINISH
        self._legacy._publish_node_events(
            self._event_sink, node, previous, completed, next_node
        )
        return {"data": completed.model_dump(mode="json")}

    @staticmethod
    def _project_plan_progress(node: GraphNode, state: AgentGraphState) -> None:
        agent = state.agent
        if agent is None or agent.plan is None:
            return
        tool_to_step = {
            "search_documents": "search",
            "web_search": "web_search",
            "read_document": "read",
            "fetch_url": "fetch_url",
            "save_note": "save",
        }
        for result in agent.tool_results:
            step_id = tool_to_step.get(result.tool_name)
            if step_id is None:
                continue
            step = next(
                (item for item in agent.plan.steps if item.step_id == step_id), None
            )
            if step is None:
                continue
            if result.success:
                index = agent.plan.steps.index(step)
                agent.plan.steps[index] = type(step).model_validate(
                    {
                        **step.model_dump(),
                        "status": PlanStepStatus.COMPLETED,
                        "result_summary": f"{result.tool_name} completed",
                        "error_message": None,
                    }
                )
            else:
                index = agent.plan.steps.index(step)
                agent.plan.steps[index] = type(step).model_validate(
                    {
                        **step.model_dump(),
                        "status": PlanStepStatus.FAILED,
                        "error_message": result.error_message
                        or "tool execution failed",
                    }
                )
        if node is GraphNode.SYNTHESIZE:
            step = next(
                (item for item in agent.plan.steps if item.step_id == "summarize"), None
            )
            if step is not None:
                step.status = PlanStepStatus.COMPLETED
                step.result_summary = "report generated"
        if node is GraphNode.FINISH:
            for step in agent.plan.steps:
                if step.status in {PlanStepStatus.PENDING, PlanStepStatus.RUNNING}:
                    step.status = PlanStepStatus.SKIPPED
                    step.result_summary = "not completed before agent finished"

    def _next_node(self, node: GraphNode, state: AgentGraphState) -> GraphNode | None:
        if node is GraphNode.INITIALIZE:
            return GraphNode.PLAN
        if node is GraphNode.PLAN:
            return GraphNode.RETRIEVE
        if node is GraphNode.RETRIEVE:
            return GraphNode.ASSESS_EVIDENCE
        if node is GraphNode.READ_SOURCES:
            return GraphNode.ASSESS_EVIDENCE
        if node is GraphNode.ASSESS_EVIDENCE:
            return (
                GraphNode.SYNTHESIZE
                if state.evidence_status is EvidenceStatus.SUFFICIENT
                else GraphNode.READ_SOURCES
                if len(state.attempted_candidates) < len(state.candidates)
                else GraphNode.SYNTHESIZE
                if state.candidates
                else GraphNode.REPLAN
                if state.replans < state.max_replans
                else GraphNode.SYNTHESIZE
            )
        if node is GraphNode.REPLAN:
            return GraphNode.RETRIEVE
        if node is GraphNode.SYNTHESIZE:
            return GraphNode.VERIFY
        if node is GraphNode.VERIFY:
            return GraphNode.SAVE
        if node is GraphNode.SAVE:
            return GraphNode.FINISH
        return None

    @staticmethod
    def _continue_or_finish(state: ResearchGraphState, destination: str) -> str:
        return (
            "finish"
            if LangGraphResearchRunner._state_for_route(state).end_reason
            else destination
        )

    @staticmethod
    def _state_for_route(value: ResearchGraphState) -> AgentGraphState:
        return AgentGraphState.model_validate(value["data"])

    def _route_after_initialize(self, value: ResearchGraphState) -> str:
        return self._continue_or_finish(value, "plan")

    def _route_after_plan(self, value: ResearchGraphState) -> str:
        return self._continue_or_finish(value, "retrieve")

    def _route_after_retrieve(self, value: ResearchGraphState) -> str:
        return self._continue_or_finish(value, "assess")

    def _route_after_replan(self, value: ResearchGraphState) -> str:
        return self._continue_or_finish(value, "retrieve")

    def _route_after_synthesize(self, value: ResearchGraphState) -> str:
        return self._continue_or_finish(value, "verify")

    def _route_after_verify(self, value: ResearchGraphState) -> str:
        return self._continue_or_finish(value, "save")

    def _route_after_save(self, _: ResearchGraphState) -> str:
        return "finish"

    def _context_or_raise(self) -> ToolContext:
        if self._context is None:
            raise RuntimeError("research context is required")
        return self._context

    def _initialize(self, value: ResearchGraphState) -> ResearchGraphState:
        return self._run_node(
            GraphNode.INITIALIZE,
            value,
            lambda state: self._legacy._initialize(state, self._context_or_raise()),
        )

    def _plan(self, value: ResearchGraphState) -> ResearchGraphState:
        return self._run_node(
            GraphNode.PLAN,
            value,
            lambda state: self._legacy._plan(state, self._context_or_raise()),
        )

    def _retrieve(self, value: ResearchGraphState) -> ResearchGraphState:
        return self._run_node(
            GraphNode.RETRIEVE,
            value,
            lambda state: self._legacy._retrieve(state, self._context_or_raise()),
        )

    def _assess(self, value: ResearchGraphState) -> ResearchGraphState:
        return self._run_node(
            GraphNode.ASSESS_EVIDENCE,
            value,
            lambda state: self._legacy._assess_evidence(
                state, self._context_or_raise()
            ),
        )

    def _read_sources(self, value: ResearchGraphState) -> ResearchGraphState:
        return self._run_node(
            GraphNode.READ_SOURCES,
            value,
            lambda state: self._legacy._read_sources(state, self._context_or_raise()),
        )

    def _replan(self, value: ResearchGraphState) -> ResearchGraphState:
        return self._run_node(
            GraphNode.REPLAN,
            value,
            lambda state: self._legacy._replan(state, self._context_or_raise()),
        )

    def _route_after_assess(self, value: ResearchGraphState) -> str:
        state = self._state(value)
        if state.end_reason is not None:
            return "finish"
        if state.evidence_status is EvidenceStatus.SUFFICIENT:
            return "synthesize"
        if len(state.attempted_candidates) < len(state.candidates):
            return "read_sources"
        if state.candidates:
            return "synthesize"
        if state.replans < state.max_replans:
            return "replan"
        return "synthesize"

    def _route_after_read(self, value: ResearchGraphState) -> str:
        state = self._state(value)
        if state.end_reason is not None:
            return "finish"
        return "assess"

    def _synthesize(self, value: ResearchGraphState) -> ResearchGraphState:
        return self._run_node(
            GraphNode.SYNTHESIZE,
            value,
            lambda state: self._legacy._synthesize(state, self._context_or_raise()),
        )

    def _verify(self, value: ResearchGraphState) -> ResearchGraphState:
        return self._run_node(
            GraphNode.VERIFY,
            value,
            lambda state: self._legacy._verify(state, self._context_or_raise()),
        )

    def _save(self, value: ResearchGraphState) -> ResearchGraphState:
        return self._run_node(
            GraphNode.SAVE,
            value,
            lambda state: self._legacy._save(state, self._context_or_raise()),
        )

    def _finish(self, value: ResearchGraphState) -> ResearchGraphState:
        def finish(state: AgentGraphState) -> dict[str, Any]:
            patch = self._legacy._finish(state, self._context_or_raise())
            if state.end_reason is GraphEndReason.MAX_STEPS:
                agent = patch["agent"]
                agent.status = AgentStatus.FAILED
            return patch

        return self._run_node(
            GraphNode.FINISH,
            value,
            finish,
        )
