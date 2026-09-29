"""End-to-end LLM agent workflow tests with an in-memory provider."""

import json
from pathlib import Path

from researchflow.agent import (
    AgentRunner,
    LLMPlanner,
    LLMSelector,
    LLMSummarizer,
    RulePlanner,
    StateSelector,
)
from researchflow.agent.summarizer import ExtractiveSummarizer
from researchflow.domain import AgentStatus
from researchflow.execution import JsonlTraceRecorder, ToolExecutor
from researchflow.llm import BaseLLMProvider, LLMRequest, LLMResponse, TokenUsage
from researchflow.tools import ToolContext, ToolRegistry
from researchflow.tools.offline import create_offline_tools


class FakeProvider(BaseLLMProvider):
    def __init__(self, responses: list[dict[str, object]]) -> None:
        self._responses = iter(responses)

    def complete(self, request: LLMRequest) -> LLMResponse:
        return LLMResponse(
            content=json.dumps(next(self._responses)),
            usage=TokenUsage(input_tokens=4, output_tokens=2),
        )


def test_llm_workflow_records_sanitized_decisions_and_saves_only_read_source(
    tmp_path: Path,
) -> None:
    documents = tmp_path / "documents"
    output = tmp_path / "output"
    documents.mkdir()
    output.mkdir()
    (documents / "good.md").write_text(
        "# Good\n\nSafe tool calls need validation.", encoding="utf-8"
    )
    context = ToolContext(
        working_directory=documents,
        output_directory=output,
        run_id="llm-run",
    )
    provider = FakeProvider(
        [
            {
                "steps": [
                    {
                        "step_id": "search",
                        "description": "Search local docs",
                        "tool_name": "search_documents",
                    },
                    {
                        "step_id": "read",
                        "description": "Read matching docs",
                        "tool_name": "read_document",
                    },
                    {"step_id": "summarize", "description": "Summarize reads"},
                    {
                        "step_id": "save",
                        "description": "Save report",
                        "tool_name": "save_note",
                    },
                ]
            },
            {
                "action_type": "search",
                "tool_name": "search_documents",
                "arguments": {"query": "tool calling", "limit": 5},
            },
            {
                "action_type": "read",
                "tool_name": "read_document",
                "arguments": {"path": "good.md"},
            },
            {"action_type": "summarize", "tool_name": None, "arguments": {}},
            {
                "summary": "The read source requires validation for safe calls.",
                "source_paths": ["good.md"],
            },
            {"action_type": "save", "tool_name": "save_note", "arguments": {}},
            {"action_type": "finish", "tool_name": None, "arguments": {}},
        ]
    )
    registry = ToolRegistry()
    for tool in create_offline_tools(context):
        registry.register(tool)
    runner = AgentRunner(
        LLMPlanner(provider, RulePlanner(), model_name="fake-model"),
        LLMSelector(provider, StateSelector(), model_name="fake-model"),
        LLMSummarizer(provider, ExtractiveSummarizer(), model_name="fake-model"),
        ToolExecutor(registry, JsonlTraceRecorder()),
        max_steps=10,
    )

    state = runner.run("tool calling", context)

    assert state.status is AgentStatus.COMPLETED
    assert "good.md" in state.final_answer
    assert "unread.md" not in state.final_answer
    assert len(state.decision_traces) == 7
    assert {trace.model for trace in state.decision_traces} == {"fake-model"}
    assert "secret" not in state.model_dump_json()
    trace_records = [
        json.loads(line)
        for line in (output / "traces" / "llm-run.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    decisions = [
        record for record in trace_records if record.get("event_type") == "llm_decision"
    ]
    assert len(decisions) == 7
    assert all(
        set(record)
        == {
            "event_type",
            "component",
            "model",
            "usage",
            "fallback",
            "fallback_reason",
            "error_type",
        }
        for record in decisions
    )
    assert "secret" not in json.dumps(decisions)
