"""Checkpointed LangGraph implementation of research planning and retrieval."""

import sqlite3
from pathlib import Path
from typing import Any, TypedDict

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from researchflow.agent.graph import AgentGraphState, EvidenceStatus, GraphAgentRunner
from researchflow.agent.planner import RulePlanner
from researchflow.agent.selector import StateSelector
from researchflow.agent.summarizer import ExtractiveSummarizer
from researchflow.domain import AgentStatus
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
    ) -> None:
        self._legacy = GraphAgentRunner(planner, selector, summarizer, executor)
        self._context: ToolContext | None = None
        self._connection = sqlite3.connect(database, check_same_thread=False)
        self._graph = self._build_graph().compile(
            checkpointer=SqliteSaver(self._connection)
        )

    def run(
        self, query: str, context: ToolContext, *, thread_id: str
    ) -> AgentGraphState:
        self._context = context
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
                "data": AgentGraphState(run_id=context.run_id, query=query).model_dump(
                    mode="json"
                )
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
        graph.add_node("read_sources", self._read_sources)
        graph.add_node("synthesize", self._synthesize)
        graph.add_node("verify", self._verify)
        graph.add_node("save", self._save)
        graph.add_node("finish", self._finish)
        graph.add_edge(START, "initialize")
        graph.add_edge("initialize", "plan")
        graph.add_edge("plan", "retrieve")
        graph.add_edge("retrieve", "read_sources")
        graph.add_edge("read_sources", "assess")
        graph.add_edge("assess", "synthesize")
        graph.add_edge("synthesize", "verify")
        graph.add_edge("verify", "save")
        graph.add_edge("save", "finish")
        graph.add_edge("finish", END)
        return graph

    def _state(self, value: ResearchGraphState) -> AgentGraphState:
        return AgentGraphState.model_validate(value["data"])

    def _update(
        self, state: AgentGraphState, patch: dict[str, Any]
    ) -> ResearchGraphState:
        return {"data": state.model_copy(update=patch).model_dump(mode="json")}

    def _context_or_raise(self) -> ToolContext:
        if self._context is None:
            raise RuntimeError("research context is required")
        return self._context

    def _initialize(self, value: ResearchGraphState) -> ResearchGraphState:
        state = self._state(value)
        return self._update(
            state, self._legacy._initialize(state, self._context_or_raise())
        )

    def _plan(self, value: ResearchGraphState) -> ResearchGraphState:
        state = self._state(value)
        return self._update(state, self._legacy._plan(state, self._context_or_raise()))

    def _retrieve(self, value: ResearchGraphState) -> ResearchGraphState:
        state = self._state(value)
        return self._update(
            state, self._legacy._retrieve(state, self._context_or_raise())
        )

    def _assess(self, value: ResearchGraphState) -> ResearchGraphState:
        state = self._state(value)
        patch = self._legacy._assess_evidence(state, self._context_or_raise())
        if not patch and state.candidates:
            patch = {"evidence_status": EvidenceStatus.PARTIAL}
        return self._update(state, patch)

    def _read_sources(self, value: ResearchGraphState) -> ResearchGraphState:
        state = self._state(value)
        return self._update(
            state, self._legacy._read_sources(state, self._context_or_raise())
        )

    def _synthesize(self, value: ResearchGraphState) -> ResearchGraphState:
        state = self._state(value)
        return self._update(
            state, self._legacy._synthesize(state, self._context_or_raise())
        )

    def _verify(self, value: ResearchGraphState) -> ResearchGraphState:
        state = self._state(value)
        return self._update(
            state, self._legacy._verify(state, self._context_or_raise())
        )

    def _save(self, value: ResearchGraphState) -> ResearchGraphState:
        state = self._state(value)
        return self._update(state, self._legacy._save(state, self._context_or_raise()))

    def _finish(self, value: ResearchGraphState) -> ResearchGraphState:
        state = self._state(value)
        return self._update(
            state, self._legacy._finish(state, self._context_or_raise())
        )
