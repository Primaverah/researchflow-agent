"""End-to-end tests for the rule-driven offline agent."""

import json
from pathlib import Path

from researchflow.agent import (
    AgentRunner,
    ExtractiveSummarizer,
    RulePlanner,
    StateSelector,
)
from researchflow.domain import AgentStatus
from researchflow.execution import JsonlTraceRecorder, ToolExecutor
from researchflow.tools import ToolContext, ToolRegistry
from researchflow.tools.offline import (
    FileSystemDocumentSource,
    FileSystemNoteStore,
    ReadDocumentTool,
    SaveNoteTool,
    SearchDocumentsTool,
    create_offline_tools,
)
from researchflow.tools.offline.interfaces import SearchHit


class MissingFirstSearchBackend:
    def search(self, query: str, limit: int) -> list[SearchHit]:
        return [
            SearchHit("missing.md", "Missing", 100, "missing"),
            SearchHit("good.md", "Good", 90, "真实内容"),
        ][:limit]


def setup_context(tmp_path: Path) -> ToolContext:
    documents = tmp_path / "documents"
    output = tmp_path / "output"
    documents.mkdir()
    output.mkdir()
    return ToolContext(
        working_directory=documents,
        output_directory=output,
        run_id="agent-run",
    )


def build_runner(context: ToolContext, *, max_steps: int = 10) -> AgentRunner:
    registry = ToolRegistry()
    for tool in create_offline_tools(context):
        registry.register(tool)
    return AgentRunner(
        RulePlanner(),
        StateSelector(),
        ExtractiveSummarizer(),
        ToolExecutor(registry, JsonlTraceRecorder()),
        max_steps=max_steps,
    )


def test_complete_workflow_writes_note_and_parseable_traces(tmp_path: Path) -> None:
    context = setup_context(tmp_path)
    (context.working_directory / "agent.md").write_text(
        "# Agent 工具\n\n工具调用需要安全参数校验。", encoding="utf-8"
    )

    state = build_runner(context).run("工具调用", context)

    assert state.status is AgentStatus.COMPLETED
    note = context.output_directory / "notes" / "agent-run.md"
    assert note.read_text(encoding="utf-8") == state.final_answer
    assert "agent.md" in state.final_answer
    assert (context.working_directory / "agent.md").exists()
    trace_lines = (
        (context.output_directory / "traces" / "agent-run.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    )
    records = [json.loads(line) for line in trace_lines]
    assert [record["tool_name"] for record in records] == [
        "search_documents",
        "read_document",
        "save_note",
    ]


def test_no_results_still_saves_report(tmp_path: Path) -> None:
    context = setup_context(tmp_path)

    state = build_runner(context).run("没有结果", context)

    assert state.status is AgentStatus.COMPLETED
    assert "未找到相关文档" in state.final_answer
    assert "- 无" in state.final_answer
    assert (context.output_directory / "notes" / "agent-run.md").exists()


def test_failed_read_continues_to_real_document(tmp_path: Path) -> None:
    context = setup_context(tmp_path)
    (context.working_directory / "good.md").write_text(
        "# Good\n\n真实内容。", encoding="utf-8"
    )
    source = FileSystemDocumentSource(context.working_directory)
    registry = ToolRegistry()
    registry.register(SearchDocumentsTool(MissingFirstSearchBackend()))
    registry.register(ReadDocumentTool(source))
    registry.register(SaveNoteTool(FileSystemNoteStore(context.output_directory)))
    runner = AgentRunner(
        RulePlanner(),
        StateSelector(),
        ExtractiveSummarizer(),
        ToolExecutor(registry, JsonlTraceRecorder()),
    )

    state = runner.run("真实", context)

    assert state.status is AgentStatus.COMPLETED
    assert "good.md" in state.final_answer
    assert "missing.md" not in state.final_answer


def test_max_steps_stops_real_workflow(tmp_path: Path) -> None:
    context = setup_context(tmp_path)
    (context.working_directory / "agent.md").write_text(
        "# Agent\n内容", encoding="utf-8"
    )

    state = build_runner(context, max_steps=1).run("Agent", context)

    assert state.status is AgentStatus.FAILED
    assert "最大步骤" in state.final_answer
    assert len(state.tool_calls) == 1
    assert not (context.output_directory / "notes").exists()
