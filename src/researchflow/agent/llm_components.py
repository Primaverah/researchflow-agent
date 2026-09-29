"""Structured LLM implementations for the bounded local agent."""

import json
from dataclasses import dataclass
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field, ValidationError

from researchflow.agent.models import AgentAction, AgentActionType
from researchflow.agent.planner import RulePlanner
from researchflow.agent.selector import StateSelector
from researchflow.agent.summarizer import ExtractiveSummarizer
from researchflow.domain import AgentState, PlanStep, ResearchPlan
from researchflow.llm import (
    BaseLLMProvider,
    LLMConfigurationError,
    LLMError,
    LLMRequest,
    LLMStructuredOutputError,
)
from researchflow.tools.offline import ReadDocumentOutput
from researchflow.tools.web import WebSource

_PLAN_STEPS = (
    ("search", "search_documents"),
    ("read", "read_document"),
    ("summarize", None),
    ("save", "save_note"),
)
_TOOL_BY_ACTION = {
    AgentActionType.SEARCH: "search_documents",
    AgentActionType.READ: "read_document",
    AgentActionType.SAVE: "save_note",
}


@dataclass(frozen=True, slots=True)
class LLMDecision:
    """Sanitized metadata for one LLM decision."""

    component: str
    model: str
    input_tokens: int
    output_tokens: int
    fallback: bool = False
    fallback_reason: str | None = None
    error_type: str | None = None


class LLMPlanStep(BaseModel):
    """One permitted local workflow step returned by the model."""

    step_id: Literal["search", "read", "summarize", "save"]
    description: str = Field(min_length=1)
    tool_name: Literal["search_documents", "read_document", "save_note"] | None = None


class LLMPlanOutput(BaseModel):
    """Validated model output for research planning."""

    steps: list[LLMPlanStep] = Field(min_length=4, max_length=4)


class LLMActionOutput(BaseModel):
    """Validated model output for selecting one action."""

    action_type: Literal["search", "read", "summarize", "save", "finish"]
    tool_name: Literal["search_documents", "read_document", "save_note"] | None = None
    arguments: dict[str, Any] = Field(default_factory=dict)


class LLMSummaryOutput(BaseModel):
    """Validated model output for a source-constrained report."""

    summary: str = Field(min_length=1)
    source_paths: list[str] = Field(default_factory=list)


class _LLMComponent:
    def __init__(self, provider: BaseLLMProvider, model_name: str) -> None:
        self._provider = provider
        self._model_name = model_name
        self.last_decision: LLMDecision | None = None

    def _complete(self, prompt: str, schema: type[BaseModel]) -> BaseModel:
        result, usage = self._provider.complete_structured(
            LLMRequest(user_prompt=prompt), schema
        )
        self.last_decision = LLMDecision(
            component="",
            model=self._model_name,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
        )
        return result

    def _fallback(self, component: str, error: Exception) -> None:
        reason = (
            "configuration_error"
            if isinstance(error, LLMConfigurationError)
            else "structured_output_invalid"
            if isinstance(error, LLMStructuredOutputError)
            else "provider_error"
        )
        self.last_decision = LLMDecision(
            component=component,
            model=self._model_name,
            input_tokens=0,
            output_tokens=0,
            fallback=True,
            fallback_reason=reason,
            error_type=type(error).__name__,
        )


class LLMPlanner(_LLMComponent):
    """Create a validated plan, with the deterministic planner as fallback."""

    def __init__(
        self,
        provider: BaseLLMProvider,
        fallback: RulePlanner,
        *,
        model_name: str,
    ) -> None:
        super().__init__(provider, model_name)
        self._fallback_planner = fallback

    def create_plan(self, query: str) -> ResearchPlan:
        try:
            output = self._complete(
                "Create a local research plan as JSON with exactly these ordered "
                "steps: search/search_documents, read/read_document, summarize, "
                f"save/save_note. Query: {query}",
                LLMPlanOutput,
            )
            assert isinstance(output, LLMPlanOutput)
            self._validate_plan(output)
        except (LLMError, ValidationError, ValueError) as exc:
            self._fallback("planner", exc)
            return self._fallback_planner.create_plan(query)
        assert self.last_decision is not None
        self.last_decision = LLMDecision(
            component="planner",
            model=self.last_decision.model,
            input_tokens=self.last_decision.input_tokens,
            output_tokens=self.last_decision.output_tokens,
        )
        return ResearchPlan(
            plan_id=str(uuid4()),
            goal=query,
            steps=[
                PlanStep(
                    step_id=step.step_id,
                    description=step.description,
                    tool_name=step.tool_name,
                )
                for step in output.steps
            ],
        )

    @staticmethod
    def _validate_plan(output: LLMPlanOutput) -> None:
        actual = tuple((step.step_id, step.tool_name) for step in output.steps)
        if actual != _PLAN_STEPS:
            raise ValueError("LLM plan must use the fixed local workflow")


class LLMSelector(_LLMComponent):
    """Select a safe next action, falling back on invalid model decisions."""

    def __init__(
        self,
        provider: BaseLLMProvider,
        fallback: StateSelector,
        *,
        model_name: str,
    ) -> None:
        super().__init__(provider, model_name)
        self._fallback_selector = fallback

    def select(self, state: AgentState) -> AgentAction:
        fallback_action = self._fallback_selector.select(state)
        try:
            output = self._complete(
                "Choose the next local action as JSON. Only search_documents, "
                "read_document, and save_note are permitted. State: "
                + json.dumps(self._state_prompt(state), ensure_ascii=False),
                LLMActionOutput,
            )
            assert isinstance(output, LLMActionOutput)
            action = AgentAction(
                action_type=AgentActionType(output.action_type),
                tool_name=output.tool_name,
                arguments=output.arguments,
            )
            self._validate_action(action, fallback_action, state)
        except (LLMError, ValidationError, ValueError) as exc:
            self._fallback("selector", exc)
            return fallback_action
        assert self.last_decision is not None
        self.last_decision = LLMDecision(
            component="selector",
            model=self.last_decision.model,
            input_tokens=self.last_decision.input_tokens,
            output_tokens=self.last_decision.output_tokens,
        )
        return action

    @staticmethod
    def _state_prompt(state: AgentState) -> dict[str, object]:
        return {
            "query": state.query,
            "completed_tools": [
                result.tool_name for result in state.tool_results if result.success
            ],
            "has_report": state.final_answer is not None,
        }

    @staticmethod
    def _validate_action(
        action: AgentAction, fallback: AgentAction, state: AgentState
    ) -> None:
        if action.action_type is not fallback.action_type:
            raise ValueError("LLM action does not match safe workflow state")
        expected_tool = _TOOL_BY_ACTION.get(action.action_type)
        if action.tool_name != expected_tool:
            raise ValueError("LLM selected an unapproved tool")
        if action.action_type is AgentActionType.SEARCH:
            if action.arguments.get("query") != state.query:
                raise ValueError("LLM search query must match the research query")
            limit = action.arguments.get("limit")
            if not isinstance(limit, int) or not 1 <= limit <= 100:
                raise ValueError("LLM search limit is invalid")
        elif action.action_type is AgentActionType.READ:
            allowed_paths = {
                hit.get("path")
                for result in state.tool_results
                if result.tool_name == "search_documents" and result.success
                for hit in (result.output or {}).get("hits", [])
                if isinstance(hit, dict)
            }
            if action.arguments.get("path") not in allowed_paths:
                raise ValueError("LLM read path was not returned by search")
        elif action.action_type is AgentActionType.SAVE:
            if action.arguments != fallback.arguments:
                raise ValueError("LLM save arguments must use the safe report path")
        elif action.arguments:
            raise ValueError("LLM non-tool action cannot contain arguments")


class LLMSummarizer(_LLMComponent):
    """Summarize only documents that the runner successfully read."""

    def __init__(
        self,
        provider: BaseLLMProvider,
        fallback: ExtractiveSummarizer,
        *,
        model_name: str,
    ) -> None:
        super().__init__(provider, model_name)
        self._fallback_summarizer = fallback

    def summarize(
        self,
        query: str,
        documents: list[ReadDocumentOutput],
        web_sources: list[WebSource] | None = None,
    ) -> str:
        try:
            output = self._complete(
                "Summarize only these successfully read local documents as JSON: "
                + json.dumps(
                    [
                        {
                            "path": document.path,
                            "title": document.title,
                            "content": document.content,
                        }
                        for document in documents
                    ],
                    ensure_ascii=False,
                ),
                LLMSummaryOutput,
            )
            assert isinstance(output, LLMSummaryOutput)
            allowed_paths = {document.path for document in documents}
            if not set(output.source_paths).issubset(allowed_paths):
                raise ValueError("LLM cited a source that was not read")
        except (LLMError, ValidationError, ValueError) as exc:
            self._fallback("summarizer", exc)
            return self._fallback_summarizer.summarize(query, documents, web_sources)
        assert self.last_decision is not None
        self.last_decision = LLMDecision(
            component="summarizer",
            model=self.last_decision.model,
            input_tokens=self.last_decision.input_tokens,
            output_tokens=self.last_decision.output_tokens,
        )
        source_lines = [
            f"- {document.title} — {document.path}"
            for document in documents
            if document.path in set(output.source_paths)
        ]
        if not source_lines:
            source_lines = ["- 无"]
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
                output.summary,
                "",
                "## 来源",
                "",
                *source_lines,
            ]
        )
