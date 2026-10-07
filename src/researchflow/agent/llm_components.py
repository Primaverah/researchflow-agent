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
from researchflow.tools.web.domains import is_allowed_domain

_TOOL_BY_ACTION = {
    AgentActionType.SEARCH: "search_documents",
    AgentActionType.READ: "read_document",
    AgentActionType.WEB_SEARCH: "web_search",
    AgentActionType.FETCH_URL: "fetch_url",
    AgentActionType.SAVE: "save_note",
}


@dataclass(frozen=True, slots=True)
class LLMDecision:
    """Sanitized metadata for one LLM decision."""

    component: str
    model: str
    input_tokens: int
    output_tokens: int
    success: bool = True
    fallback: bool = False
    fallback_reason: str | None = None
    error_type: str | None = None
    finish_reason: str = "completed"
    diagnostic: dict[str, object] | None = None


class LLMPlanStep(BaseModel):
    """One permitted local workflow step returned by the model."""

    step_id: Literal["search", "read", "web_search", "fetch_url", "summarize", "save"]
    description: str = Field(min_length=1)
    tool_name: (
        Literal[
            "search_documents", "read_document", "web_search", "fetch_url", "save_note"
        ]
        | None
    ) = None


class LLMPlanOutput(BaseModel):
    """Validated model output for research planning."""

    steps: list[LLMPlanStep] = Field(min_length=4, max_length=6)


class LLMActionOutput(BaseModel):
    """Validated model output for selecting one action."""

    action_type: Literal[
        "search", "read", "web_search", "fetch_url", "summarize", "save", "finish"
    ]
    tool_name: (
        Literal[
            "search_documents", "read_document", "web_search", "fetch_url", "save_note"
        ]
        | None
    ) = None
    arguments: dict[str, Any] = Field(default_factory=dict)


class LLMSummaryOutput(BaseModel):
    """Validated model output for a source-constrained report."""

    summary: str = Field(min_length=1)
    source_paths: list[str] = Field(default_factory=list)


class _LLMComponent:
    def __init__(
        self,
        provider: BaseLLMProvider,
        model_name: str,
        max_output_tokens: int,
        response_format: str = "json_object",
        thinking: bool = False,
    ) -> None:
        self._provider = provider
        self._model_name = model_name
        self._max_output_tokens = max_output_tokens
        self._response_format = response_format
        self._thinking = thinking
        self.last_decision: LLMDecision | None = None

    def _complete(
        self,
        instruction: str,
        schema: type[BaseModel],
        example: dict[str, object],
    ) -> BaseModel:
        prompt = (
            f"{instruction}\n\nJSON Schema: "
            + json.dumps(schema.model_json_schema(), ensure_ascii=False)
            + "\n\nMinimum valid JSON example: "
            + json.dumps(example, ensure_ascii=False)
        )
        result, usage = self._provider.complete_structured(
            LLMRequest(
                user_prompt=prompt + " Return only JSON, without Markdown code fences.",
                system_prompt='Return only valid JSON. Example: {"result": "value"}.',
                max_output_tokens=self._max_output_tokens,
                response_format=self._response_format,
                thinking=self._thinking,
            ),
            schema,
        )
        self.last_decision = LLMDecision(
            component="",
            model=self._model_name,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            finish_reason=getattr(self._provider, "_last_finish_reason", "completed"),
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
            input_tokens=getattr(error, "input_tokens", 0),
            output_tokens=getattr(error, "output_tokens", 0),
            success=False,
            fallback=True,
            fallback_reason=reason,
            error_type=type(error).__name__,
            finish_reason=getattr(error, "finish_reason", "fallback"),
            diagnostic=getattr(error, "diagnostic", None),
        )


class LLMPlanner(_LLMComponent):
    """Create a validated plan, with the deterministic planner as fallback."""

    def __init__(
        self,
        provider: BaseLLMProvider,
        fallback: RulePlanner,
        *,
        model_name: str,
        max_output_tokens: int = 1024,
        response_format: str = "json_object",
        thinking: bool = False,
    ) -> None:
        super().__init__(
            provider, model_name, max_output_tokens, response_format, thinking
        )
        self._fallback_planner = fallback

    def create_plan(self, query: str) -> ResearchPlan:
        expected_plan = self._fallback_planner.create_plan(query)
        expected_steps = tuple(
            (step.step_id, step.tool_name) for step in expected_plan.steps
        )
        try:
            output = self._complete(
                "Create a research plan as JSON with exactly these ordered steps: "
                + json.dumps(expected_steps)
                + f". Query: {query}",
                LLMPlanOutput,
                {
                    "steps": [
                        {
                            "step_id": "search",
                            "description": "Search relevant local documents.",
                            "tool_name": "search_documents",
                        },
                        {
                            "step_id": "read",
                            "description": "Read selected documents.",
                            "tool_name": "read_document",
                        },
                        {
                            "step_id": "summarize",
                            "description": "Summarize verified evidence.",
                            "tool_name": None,
                        },
                        {
                            "step_id": "save",
                            "description": "Save the research report.",
                            "tool_name": "save_note",
                        },
                    ]
                },
            )
            assert isinstance(output, LLMPlanOutput)
            self._validate_plan(output, expected_steps)
        except (LLMError, ValidationError, ValueError) as exc:
            self._fallback("planner", exc)
            return expected_plan
        assert self.last_decision is not None
        self.last_decision = LLMDecision(
            component="planner",
            model=self.last_decision.model,
            input_tokens=self.last_decision.input_tokens,
            output_tokens=self.last_decision.output_tokens,
            finish_reason=self.last_decision.finish_reason,
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
    def _validate_plan(
        output: LLMPlanOutput, expected_steps: tuple[tuple[str, str | None], ...]
    ) -> None:
        actual = tuple((step.step_id, step.tool_name) for step in output.steps)
        if actual != expected_steps:
            raise ValueError("LLM plan must use the fixed safe workflow")


class LLMSelector(_LLMComponent):
    """Select a safe next action, falling back on invalid model decisions."""

    def __init__(
        self,
        provider: BaseLLMProvider,
        fallback: StateSelector,
        *,
        model_name: str,
        max_output_tokens: int = 1024,
        response_format: str = "json_object",
        thinking: bool = False,
    ) -> None:
        super().__init__(
            provider, model_name, max_output_tokens, response_format, thinking
        )
        self._fallback_selector = fallback

    def select(self, state: AgentState) -> AgentAction:
        fallback_action = self._fallback_selector.select(state)
        try:
            output = self._complete(
                "Choose the next safe workflow action as JSON. It must exactly "
                "match the permitted fallback action. State: "
                + json.dumps(self._state_prompt(state), ensure_ascii=False),
                LLMActionOutput,
                {
                    "action_type": fallback_action.action_type.value,
                    "tool_name": fallback_action.tool_name,
                    "arguments": fallback_action.arguments,
                },
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
            finish_reason=self.last_decision.finish_reason,
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
        elif action.action_type is AgentActionType.WEB_SEARCH:
            if action.arguments != fallback.arguments:
                raise ValueError("LLM web search must use the safe query and limit")
        elif action.action_type is AgentActionType.FETCH_URL:
            if action.arguments != fallback.arguments:
                raise ValueError("LLM fetch must use an approved search result")
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
        max_output_tokens: int = 2048,
        response_format: str = "json_object",
        thinking: bool = False,
        allowed_domains: tuple[str, ...] = (),
    ) -> None:
        super().__init__(
            provider, model_name, max_output_tokens, response_format, thinking
        )
        self._fallback_summarizer = fallback
        self._allowed_domains = allowed_domains

    def summarize(
        self,
        query: str,
        documents: list[ReadDocumentOutput],
        web_sources: list[WebSource] | None = None,
        *,
        answer_language: str = "",
    ) -> str:
        verified_web_sources = [
            source
            for source in web_sources or []
            if is_allowed_domain(source.url, self._allowed_domains)
        ]
        evidence = [
            {
                "source": document.path,
                "title": document.title,
                "content": document.content,
            }
            for document in documents
        ]
        evidence.extend(
            {
                "source": source.url,
                "title": source.title,
                "content": source.content,
            }
            for source in verified_web_sources
        )
        try:
            language_instruction = {
                "zh": "Write the summary in Simplified Chinese.",
                "en": "Write the summary in English.",
            }.get(answer_language, "Use the language requested by the question.")
            output = self._complete(
                "Answer the user's question directly; do not merely summarize the "
                "web pages. Use only the successfully read evidence below. "
                f"User question: {query}\n{language_instruction} "
                "Return JSON and cite only their source values in source_paths: "
                + json.dumps(evidence, ensure_ascii=False),
                LLMSummaryOutput,
                {
                    "summary": "A concise summary supported by the supplied source.",
                    "source_paths": [item["source"] for item in evidence[:1]],
                },
            )
            assert isinstance(output, LLMSummaryOutput)
            allowed_paths = {item["source"] for item in evidence}
            if not set(output.source_paths).issubset(allowed_paths):
                raise ValueError("LLM cited a source that was not read")
            if answer_language == "zh" and not any(
                "\u4e00" <= character <= "\u9fff" for character in output.summary
            ):
                raise ValueError(
                    "LLM did not honor the requested Chinese answer language"
                )
        except (LLMError, ValidationError, ValueError) as exc:
            self._fallback("summarizer", exc)
            return self._fallback_summarizer.summarize(
                query,
                documents,
                web_sources,
                answer_language=answer_language,
            )
        assert self.last_decision is not None
        self.last_decision = LLMDecision(
            component="summarizer",
            model=self.last_decision.model,
            input_tokens=self.last_decision.input_tokens,
            output_tokens=self.last_decision.output_tokens,
            finish_reason=self.last_decision.finish_reason,
        )
        citations = set(output.source_paths)
        source_lines = [
            f"- {item['title']} — {item['source']}"
            for item in evidence
            if item["source"] in citations
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
