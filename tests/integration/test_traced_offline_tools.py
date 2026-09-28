"""Integration tests for running the offline tools through ToolExecutor."""

import json
from pathlib import Path

from researchflow.domain import ToolCall
from researchflow.execution import JsonlTraceRecorder, ToolExecutor
from researchflow.tools import ToolContext, ToolRegistry
from researchflow.tools.offline import create_offline_tools


def test_all_offline_tools_run_through_executor(tmp_path: Path) -> None:
    documents = tmp_path / "documents"
    output = tmp_path / "output"
    documents.mkdir()
    output.mkdir()
    (documents / "agent.md").write_text(
        "# Agent 工具调用\n\n安全的本地文档检索。", encoding="utf-8"
    )
    context = ToolContext(
        working_directory=documents,
        output_directory=output,
        run_id="offline-run",
    )
    registry = ToolRegistry()
    for tool in create_offline_tools(context):
        registry.register(tool)
    executor = ToolExecutor(registry, JsonlTraceRecorder())

    search = executor.execute(
        ToolCall(
            call_id="search-1",
            tool_name="search_documents",
            arguments={"query": "工具调用"},
        ),
        context,
    )
    read = executor.execute(
        ToolCall(
            call_id="read-1",
            tool_name="read_document",
            arguments={"path": "agent.md"},
        ),
        context,
    )
    save = executor.execute(
        ToolCall(
            call_id="save-1",
            tool_name="save_note",
            arguments={"path": "研究.md", "content": "中文研究笔记"},
        ),
        context,
    )

    assert search.success and search.output["count"] == 1
    assert read.success and read.output["title"] == "Agent 工具调用"
    assert save.success and (output / "研究.md").read_text() == "中文研究笔记"
    lines = (output / "traces" / "offline-run.jsonl").read_text().splitlines()
    assert len(lines) == 3
    assert [json.loads(line)["call_id"] for line in lines] == [
        "search-1",
        "read-1",
        "save-1",
    ]
