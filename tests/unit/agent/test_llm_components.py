"""Tests for structured LLM agent components without network access."""

import json
from datetime import UTC, datetime

from researchflow.agent.llm_components import LLMPlanner, LLMSelector, LLMSummarizer
from researchflow.agent.planner import RulePlanner
from researchflow.agent.selector import StateSelector
from researchflow.agent.summarizer import ExtractiveSummarizer
from researchflow.agent.web import WebRulePlanner
from researchflow.domain import AgentState, AgentStatus
from researchflow.llm import (
    BaseLLMProvider,
    LLMRequest,
    LLMResponse,
    LLMStructuredOutputError,
    TokenUsage,
)
from researchflow.tools.offline import ReadDocumentOutput
from researchflow.tools.web import WebSource


class FakeProvider(BaseLLMProvider):
    def __init__(self, responses: list[dict[str, object]]) -> None:
        self._responses = iter(responses)
        self.requests: list[LLMRequest] = []

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        return LLMResponse(
            content=json.dumps(next(self._responses)),
            usage=TokenUsage(input_tokens=3, output_tokens=2),
            finish_reason="stop",
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


def test_llm_components_use_their_configured_token_budgets() -> None:
    provider = FakeProvider(
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
            },
            {
                "action_type": "search",
                "tool_name": "search_documents",
                "arguments": {"query": "tool calling", "limit": 5},
            },
            {"summary": "No sources.", "source_paths": []},
        ]
    )
    planner = LLMPlanner(
        provider, RulePlanner(), model_name="fake-model", max_output_tokens=1024
    )
    plan = planner.create_plan("tool calling")
    selector = LLMSelector(
        provider, StateSelector(), model_name="fake-model", max_output_tokens=1024
    )
    selector.select(
        AgentState(
            run_id="run-1",
            query="tool calling",
            status=AgentStatus.RUNNING,
            plan=plan,
        )
    )
    summarizer = LLMSummarizer(
        provider,
        ExtractiveSummarizer(),
        model_name="fake-model",
        max_output_tokens=2048,
    )
    summarizer.summarize("tool calling", [])

    assert [request.max_output_tokens for request in provider.requests] == [
        1024,
        1024,
        2048,
    ]


def test_planner_retries_field_mismatch_with_its_schema_and_keeps_usage() -> None:
    provider = FakeProvider([{"action": "plan"}, {"action": "plan"}])
    planner = LLMPlanner(provider, RulePlanner(), model_name="fake-model")

    planner.create_plan("tool calling")

    assert planner.last_decision is not None
    assert planner.last_decision.fallback is True
    assert planner.last_decision.input_tokens == 3
    assert planner.last_decision.output_tokens == 2
    assert planner.last_decision.finish_reason == "stop"
    assert len(provider.requests) == 2
    assert '"steps"' in provider.requests[0].user_prompt
    assert '"step_id"' in provider.requests[0].user_prompt
    assert "Validation errors" in provider.requests[1].user_prompt


def test_selector_retries_field_mismatch_with_its_schema_and_keeps_usage() -> None:
    provider = FakeProvider([{"result": "search"}, {"result": "search"}])
    selector = LLMSelector(provider, StateSelector(), model_name="fake-model")
    state = AgentState(
        run_id="run-1",
        query="tool calling",
        status=AgentStatus.RUNNING,
        plan=RulePlanner().create_plan("tool calling"),
    )

    selector.select(state)

    assert selector.last_decision is not None
    assert selector.last_decision.fallback is True
    assert selector.last_decision.input_tokens == 3
    assert selector.last_decision.output_tokens == 2
    assert selector.last_decision.finish_reason == "stop"
    assert len(provider.requests) == 2
    assert '"action_type"' in provider.requests[0].user_prompt
    assert '"arguments"' in provider.requests[0].user_prompt
    assert "Validation errors" in provider.requests[1].user_prompt


def test_summarizer_retries_field_mismatch_with_its_schema_and_keeps_usage() -> None:
    provider = FakeProvider([{"result": "summary"}, {"result": "summary"}])
    summarizer = LLMSummarizer(
        provider, ExtractiveSummarizer(), model_name="fake-model"
    )
    documents = [
        ReadDocumentOutput(
            path="read.md",
            title="Read source",
            content="Verified content.",
            char_count=17,
        )
    ]

    summarizer.summarize("tool calling", documents)

    assert summarizer.last_decision is not None
    assert summarizer.last_decision.fallback is True
    assert summarizer.last_decision.input_tokens == 3
    assert summarizer.last_decision.output_tokens == 2
    assert summarizer.last_decision.finish_reason == "stop"
    assert len(provider.requests) == 2
    assert '"summary"' in provider.requests[0].user_prompt
    assert '"source_paths"' in provider.requests[0].user_prompt
    assert "Validation errors" in provider.requests[1].user_prompt


def test_llm_planner_preserves_the_safe_web_workflow() -> None:
    planner = LLMPlanner(
        FakeProvider(
            [
                {
                    "steps": [
                        {
                            "step_id": "search",
                            "description": "Search local",
                            "tool_name": "search_documents",
                        },
                        {
                            "step_id": "read",
                            "description": "Read",
                            "tool_name": "read_document",
                        },
                        {
                            "step_id": "web_search",
                            "description": "Search web",
                            "tool_name": "web_search",
                        },
                        {
                            "step_id": "fetch_url",
                            "description": "Fetch web",
                            "tool_name": "fetch_url",
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
        WebRulePlanner(),
        model_name="fake-model",
    )

    plan = planner.create_plan("Python")

    assert [step.step_id for step in plan.steps] == [
        "search",
        "read",
        "web_search",
        "fetch_url",
        "summarize",
        "save",
    ]
    assert planner.last_decision is not None
    assert not planner.last_decision.fallback


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


def test_llm_summarizer_can_cite_allowed_fetched_web_evidence() -> None:
    summarizer = LLMSummarizer(
        FakeProvider(
            [
                {
                    "summary": "The official documentation provides the evidence.",
                    "source_paths": ["https://docs.python.org/3/"],
                }
            ]
        ),
        ExtractiveSummarizer(allowed_domains=("docs.python.org",)),
        model_name="fake-model",
        allowed_domains=("docs.python.org",),
    )

    report = summarizer.summarize(
        "Python",
        [],
        [
            WebSource(
                title="Python docs",
                url="https://docs.python.org/3/",
                summary="",
                accessed_at=datetime.now(UTC),
                content="Official evidence.",
            ),
            WebSource(
                title="Untrusted",
                url="https://example.com/python",
                summary="",
                accessed_at=datetime.now(UTC),
                content="Untrusted evidence.",
            ),
        ],
    )

    assert "https://docs.python.org/3/" in report
    assert "example.com" not in report


def test_llm_summarizer_receives_question_and_explicit_answer_language() -> None:
    provider = FakeProvider(
        [
            {
                "summary": "RAG 通过检索外部证据辅助生成。",
                "source_paths": ["paper.md"],
            }
        ]
    )
    summarizer = LLMSummarizer(
        provider,
        ExtractiveSummarizer(),
        model_name="fake-model",
    )

    summarizer.summarize(
        "什么是RAG？帮我查找3篇相关论文",
        [
            ReadDocumentOutput(
                path="paper.md",
                title="RAG paper",
                content="Retrieval augmented generation evidence.",
                char_count=39,
            )
        ],
        answer_language="zh",
    )

    prompt = provider.requests[-1].user_prompt
    assert "什么是RAG？帮我查找3篇相关论文" in prompt
    assert "Simplified Chinese" in prompt
    assert "do not merely summarize" in prompt


def test_llm_summarizer_rejects_english_when_chinese_is_required() -> None:
    summarizer = LLMSummarizer(
        FakeProvider(
            [
                {
                    "summary": "RAG uses retrieval before generation.",
                    "source_paths": ["paper.md"],
                }
            ]
        ),
        ExtractiveSummarizer(),
        model_name="fake-model",
    )

    summarizer.summarize(
        "什么是RAG",
        [
            ReadDocumentOutput(
                path="paper.md",
                title="RAG paper",
                content="中文证据：RAG 先检索再生成。",
                char_count=17,
            )
        ],
        answer_language="zh",
    )

    assert summarizer.last_decision is not None
    assert summarizer.last_decision.fallback is True


def test_llm_fallback_records_safe_reason_without_raw_provider_message() -> None:
    class FailingProvider(BaseLLMProvider):
        def complete(self, request: LLMRequest) -> LLMResponse:
            raise LLMStructuredOutputError("raw response contains secret")

    planner = LLMPlanner(FailingProvider(), RulePlanner(), model_name="fake-model")

    planner.create_plan("tool calling")

    assert planner.last_decision is not None
    assert planner.last_decision.fallback_reason == "structured_output_invalid"
    assert planner.last_decision.error_type == "LLMStructuredOutputError"
