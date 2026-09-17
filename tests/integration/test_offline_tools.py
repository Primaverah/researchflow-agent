"""Integration tests for filesystem-backed offline tools."""

import os
from pathlib import Path

import pytest

from researchflow.domain import ToolCall, ToolResult, ToolResultStatus
from researchflow.tools import ToolContext, ToolRegistry
from researchflow.tools.offline import create_offline_tools


def execute(
    tool,
    context: ToolContext,
    arguments: dict[str, object],
) -> ToolResult:
    return tool.execute(
        ToolCall(call_id="call-1", tool_name=tool.name, arguments=arguments),
        context,
    )


@pytest.fixture
def offline_environment(tmp_path: Path) -> tuple[ToolContext, dict[str, object]]:
    documents = tmp_path / "documents"
    notes = tmp_path / "notes"
    documents.mkdir()
    notes.mkdir()
    context = ToolContext(
        working_directory=documents,
        output_directory=notes,
        run_id="run-1",
    )
    tools = {tool.name: tool for tool in create_offline_tools(context)}
    return context, tools


def test_tools_can_be_registered_and_executed(offline_environment) -> None:
    context, tools = offline_environment
    (context.working_directory / "agent.md").write_text(
        "# Agent 架构\n\n工具调用需要安全校验。", encoding="utf-8"
    )
    registry = ToolRegistry()
    for tool in tools.values():
        registry.register(tool)

    result = execute(registry.get("search_documents"), context, {"query": "工具调用"})

    assert [tool.name for tool in registry.list_tools()] == [
        "read_document",
        "save_note",
        "search_documents",
    ]
    assert result.output["count"] == 1


def test_chinese_phrase_and_multiple_keywords_search(offline_environment) -> None:
    context, tools = offline_environment
    (context.working_directory / "a.md").write_text(
        "# 工具调用安全\n参数校验与文档读取。", encoding="utf-8"
    )
    (context.working_directory / "b.txt").write_text(
        "普通标题\nAgent 使用工具调用完成任务。", encoding="utf-8"
    )

    phrase = execute(tools["search_documents"], context, {"query": "工具调用"})
    keywords = execute(tools["search_documents"], context, {"query": "Agent 文档"})

    assert [hit["path"] for hit in phrase.output["hits"]] == ["a.md", "b.txt"]
    assert {hit["path"] for hit in keywords.output["hits"]} == {"a.md", "b.txt"}


def test_search_scoring_ties_limit_and_empty_results(offline_environment) -> None:
    context, tools = offline_environment
    (context.working_directory / "title.md").write_text(
        "# Agent\n无匹配正文。", encoding="utf-8"
    )
    (context.working_directory / "body.md").write_text(
        "# 普通标题\nAgent", encoding="utf-8"
    )
    for name in ("a.txt", "b.txt"):
        (context.working_directory / name).write_text("关键词", encoding="utf-8")

    weighted = execute(tools["search_documents"], context, {"query": "Agent"})
    tied = execute(tools["search_documents"], context, {"query": "关键词", "limit": 1})
    empty = execute(tools["search_documents"], context, {"query": "不存在"})

    assert weighted.output["hits"][0]["path"] == "title.md"
    assert tied.output["count"] == 1
    assert tied.output["hits"][0]["path"] == "a.txt"
    assert empty.output == {"query": "不存在", "count": 0, "hits": []}


def test_search_returns_bounded_snippet_and_relative_paths(offline_environment) -> None:
    context, tools = offline_environment
    content = "前" * 120 + "工具调用" + "后" * 120
    (context.working_directory / "long.md").write_text(content, encoding="utf-8")

    result = execute(tools["search_documents"], context, {"query": "工具调用"})
    hit = result.output["hits"][0]

    assert hit["path"] == "long.md"
    assert len(hit["snippet"]) <= 160
    assert "工具调用" in hit["snippet"]
    assert str(context.working_directory) not in str(result.output)


def test_read_document_supports_chinese_markdown_and_text(offline_environment) -> None:
    context, tools = offline_environment
    (context.working_directory / "中文.md").write_text(
        "# 中文标题\n\n中文内容。", encoding="utf-8"
    )
    (context.working_directory / "中文.txt").write_text("纯文本内容", encoding="utf-8")

    markdown = execute(tools["read_document"], context, {"path": "中文.md"})
    text = execute(tools["read_document"], context, {"path": "中文.txt"})

    assert markdown.output["title"] == "中文标题"
    assert markdown.output["char_count"] == len("# 中文标题\n\n中文内容。")
    assert text.output["title"] == "中文"
    assert text.output["content"] == "纯文本内容"


@pytest.mark.parametrize(
    ("path", "error_type"),
    [
        ("missing.md", "document_not_found"),
        ("unsupported.pdf", "unsupported_file_type"),
        ("folder.md", "not_a_file"),
        ("invalid.md", "invalid_encoding"),
        ("../escape.md", "unsafe_path"),
        ("nested/../../escape.md", "unsafe_path"),
    ],
)
def test_read_document_returns_stable_failures(
    offline_environment, path: str, error_type: str
) -> None:
    context, tools = offline_environment
    (context.working_directory / "folder.md").mkdir()
    (context.working_directory / "invalid.md").write_bytes(b"\xff\xfe")

    result = execute(tools["read_document"], context, {"path": path})

    assert result.status is ToolResultStatus.FAILED
    assert result.error_type == error_type
    assert str(context.working_directory) not in result.error_message


def test_read_rejects_absolute_and_symlink_escape(
    offline_environment, tmp_path
) -> None:
    context, tools = offline_environment
    outside = tmp_path / "outside.md"
    outside.write_text("秘密", encoding="utf-8")
    link = context.working_directory / "link.md"
    try:
        link.symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"symbolic links are unavailable: {exc}")

    absolute = execute(
        tools["read_document"], context, {"path": str(outside.resolve())}
    )
    symlink = execute(tools["read_document"], context, {"path": "link.md"})

    assert absolute.error_type == "unsafe_path"
    assert symlink.error_type == "unsafe_path"


def test_save_note_supports_chinese_nested_paths_and_overwrite(
    offline_environment,
) -> None:
    context, tools = offline_environment

    created = execute(
        tools["save_note"],
        context,
        {"path": "研究/笔记.md", "content": "  中文笔记  "},
    )
    rejected = execute(
        tools["save_note"],
        context,
        {"path": "研究/笔记.md", "content": "新内容"},
    )
    target = context.output_directory / "研究" / "笔记.md"
    assert target.read_text(encoding="utf-8") == "  中文笔记  "
    overwritten = execute(
        tools["save_note"],
        context,
        {"path": "研究/笔记.md", "content": "新内容", "overwrite": True},
    )

    assert created.output == {"path": "研究/笔记.md", "char_count": 8}
    assert rejected.status is ToolResultStatus.FAILED
    assert rejected.error_type == "note_exists"
    assert overwritten.status is ToolResultStatus.SUCCEEDED
    assert target.read_text(encoding="utf-8") == "新内容"


@pytest.mark.parametrize(
    "path",
    ["../note.md", "nested/../../note.md"],
)
def test_save_rejects_path_traversal(offline_environment, path: str) -> None:
    context, tools = offline_environment

    result = execute(tools["save_note"], context, {"path": path, "content": "x"})

    assert result.status is ToolResultStatus.FAILED
    assert result.error_type == "unsafe_path"


def test_save_rejects_absolute_symlink_directory_and_suffix(
    offline_environment, tmp_path
) -> None:
    context, tools = offline_environment
    outside = tmp_path / "outside"
    outside.mkdir()
    link = context.output_directory / "link"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symbolic links are unavailable: {exc}")
    (context.output_directory / "directory.md").mkdir()

    cases = [
        (str((tmp_path / "absolute.md").resolve()), "unsafe_path"),
        ("link/note.md", "unsafe_path"),
        ("directory.md", "not_a_file"),
        ("note.pdf", "unsupported_file_type"),
    ]
    for path, error_type in cases:
        result = execute(tools["save_note"], context, {"path": path, "content": "内容"})
        assert result.error_type == error_type


def test_failed_atomic_save_removes_temporary_files(
    offline_environment, monkeypatch
) -> None:
    context, tools = offline_environment

    def fail_replace(source, destination) -> None:
        raise OSError("simulated failure")

    monkeypatch.setattr(os, "replace", fail_replace)
    result = execute(
        tools["save_note"], context, {"path": "note.md", "content": "内容"}
    )

    assert result.status is ToolResultStatus.FAILED
    assert result.error_type == "write_failed"
    assert not (context.output_directory / "note.md").exists()
    assert list(context.output_directory.iterdir()) == []
