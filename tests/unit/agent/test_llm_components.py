"""Tests for structured LLM agent components without network access."""

import json

from researchflow.agent.llm_components import LLMPlanner, LLMSelector, LLMSummarizer
from researchflow.agent.planner import RulePlanner
from researchflow.agent.selector import StateSelector
from researchflow.agent.summarizer import ExtractiveSummarizer
from researchflow.domain import AgentState, AgentStatus
from researchflow.llm import BaseLLMProvider, LLMRequest, LLMResponse, TokenUsage
from researchflow.tools.offline import ReadDocumentOutput


class FakeProvider(BaseLLMProvider):
    def __init__(self, responses: list[dict[str, object]]) -> None:
        self._responses = iter(responses)

    def complete(self, request: LLMRequest) -> LLMResponse:
        return LLMResponse(
            content=json.dumps(next(self._responses)),
            usage=TokenUsage(input_tokens=3, output_tokens=2),
        )


def test_llm_planner_returns_validated_local_plan() -> None:
    planner = LLMPlanner(
        FakeProvider(
            [
                {
                    "steps": [
                        {
                            "step_id": "search",
                            "description": "Search",
                            "tool_name": "search_documents",
                        },
                        {
                            "step_id": "read",
                            "description": "Read",
                            "tool_name": "read_document",
                        },
                        {"step_id": "summarize", "description": "Summarize"},
                        {
                            "step_id": "save",
                            "description": "Save",
                            "tool_name": "save_note",
                        },
                    ]
                }
            ]
        ),
        RulePlanner(),
        model_name="fake-model",
    )

    plan = planner.create_plan("tool calling")

    assert [step.step_id for step in plan.steps] == [
        "search",
        "read",
        "summarize",
        "save",
    ]
    assert planner.last_decision is not None
    assert planner.last_decision.component == "planner"
    assert planner.last_decision.model == "fake-model"
    assert planner.last_decision.input_tokens == 3


def test_llm_selector_rejects_an_unapproved_tool_and_uses_rule_fallback() -> None:
    selector = LLMSelector(
        FakeProvider(
            [
                {
                    "action_type": "search",
                    "tool_name": "delete_everything",
                    "arguments": {"query": "tool calling", "limit": 5},
                },
                {
                    "action_type": "search",
                    "tool_name": "delete_everything",
                    "arguments": {"query": "tool calling", "limit": 5},
                },
            ]
        ),
        StateSelector(),
        model_name="fake-model",
    )
    state = AgentState(
        run_id="run-1",
        query="tool calling",
        status=AgentStatus.RUNNING,
        plan=RulePlanner().create_plan("tool calling"),
    )

    action = selector.select(state)

    assert action.tool_name == "search_documents"
    assert action.arguments == {"query": "tool calling", "limit": 5}
    assert selector.last_decision is not None
    assert selector.last_decision.fallback is True


def test_llm_summarizer_keeps_only_successfully_read_sources() -> None:
    summarizer = LLMSummarizer(
        FakeProvider(
            [
                {
                    "summary": "The read document explains safe tool calls.",
                    "source_paths": ["good.md"],
                }
            ]
        ),
        ExtractiveSummarizer(),
        model_name="fake-model",
    )
    documents = [
        ReadDocumentOutput(
            path="good.md",
            title="Good",
            content="Safe tool calls are validated.",
            char_count=30,
        )
    ]

    report = summarizer.summarize("tool calling", documents)

    assert "good.md" in report
    assert "unread.md" not in report
    assert "safe tool calls" in report
    assert summarizer.last_decision is not None
    assert summarizer.last_decision.output_tokens == 2


def test_llm_summarizer_falls_back_when_model_cites_an_unread_source() -> None:
    summarizer = LLMSummarizer(
        FakeProvider(
            [
                {
                    "summary": "An untrusted source says otherwise.",
                    "source_paths": ["unread.md"],
                }
            ]
        ),
        ExtractiveSummarizer(),
        model_name="fake-model",
    )
    documents = [
        ReadDocumentOutput(
            path="good.md",
            title="Good",
            content="Validated local content.",
            char_count=24,
        )
    ]

    report = summarizer.summarize("validation", documents)

    assert "good.md" in report
    assert "unread.md" not in report
    assert summarizer.last_decision is not None
    assert summarizer.last_decision.fallback is True
