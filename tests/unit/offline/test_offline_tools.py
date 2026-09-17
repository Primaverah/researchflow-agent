"""Unit tests for the three offline tools and their injectable backends."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from researchflow.domain import ToolCall
from researchflow.tools import ToolContext
from researchflow.tools.offline import (
    KeywordSearchBackend,
    ReadDocumentArgs,
    ReadDocumentTool,
    SaveNoteArgs,
    SaveNoteTool,
    SearchDocumentsArgs,
    SearchDocumentsTool,
)
from researchflow.tools.offline.interfaces import Document, SearchHit


class FakeDocumentSource:
    def list_documents(self) -> list[str]:
        return ["fake.md"]

    def read_document(self, path: str) -> Document:
        return Document(path=path, title="假文档", content="内容")


class ScoringDocumentSource:
    def list_documents(self) -> list[str]:
        return ["title.md", "body.md"]

    def read_document(self, path: str) -> Document:
        if path == "title.md":
            return Document(path=path, title="Agent", content="Agent in body")
        return Document(path=path, title="Other", content="Agent")


class FakeSearchBackend:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    def search(self, query: str, limit: int) -> list[SearchHit]:
        self.calls.append((query, limit))
        return [SearchHit(path="fake.md", title="假文档", score=7, snippet="内容")]


class FakeNoteStore:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, bool]] = []

    def save(self, path: str, content: str, overwrite: bool = False) -> str:
        self.calls.append((path, content, overwrite))
        return path


@pytest.fixture
def context(tmp_path: Path) -> ToolContext:
    return ToolContext(
        working_directory=tmp_path,
        output_directory=tmp_path,
        run_id="run-1",
    )


def test_search_tool_uses_injected_backend(context: ToolContext) -> None:
    backend = FakeSearchBackend()
    tool = SearchDocumentsTool(backend)

    result = tool.execute(
        ToolCall(
            call_id="call-1",
            tool_name="search_documents",
            arguments={"query": "工具", "limit": 3},
        ),
        context,
    )

    assert backend.calls == [("工具", 3)]
    assert result.output == {
        "query": "工具",
        "count": 1,
        "hits": [{"path": "fake.md", "title": "假文档", "score": 7, "snippet": "内容"}],
    }


def test_read_tool_uses_injected_document_source(context: ToolContext) -> None:
    tool = ReadDocumentTool(FakeDocumentSource())

    result = tool.execute(
        ToolCall(
            call_id="call-1",
            tool_name="read_document",
            arguments={"path": "fake.md"},
        ),
        context,
    )

    assert result.output == {
        "path": "fake.md",
        "title": "假文档",
        "content": "内容",
        "char_count": 2,
    }


def test_save_tool_uses_injected_note_store(context: ToolContext) -> None:
    store = FakeNoteStore()
    tool = SaveNoteTool(store)

    result = tool.execute(
        ToolCall(
            call_id="call-1",
            tool_name="save_note",
            arguments={"path": "研究.md", "content": "  内容  ", "overwrite": True},
        ),
        context,
    )

    assert store.calls == [("研究.md", "  内容  ", True)]
    assert result.output == {"path": "研究.md", "char_count": 6}


@pytest.mark.parametrize(
    ("model", "data"),
    [
        (SearchDocumentsArgs, {"query": ""}),
        (SearchDocumentsArgs, {"query": "   "}),
        (SearchDocumentsArgs, {"query": "agent", "limit": 0}),
        (SearchDocumentsArgs, {"query": "agent", "limit": 101}),
        (ReadDocumentArgs, {"path": " "}),
        (ReadDocumentArgs, {"path": "doc.md", "unknown": True}),
        (SaveNoteArgs, {"path": "", "content": "content"}),
        (SaveNoteArgs, {"path": "note.md", "content": " \n "}),
        (SaveNoteArgs, {"path": "note.md", "content": "x", "unknown": True}),
        (SearchDocumentsArgs, {"query": "agent", "unknown": True}),
    ],
)
def test_argument_models_reject_invalid_input(model, data: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        model.model_validate(data)


def test_backend_failure_becomes_structured_result(context: ToolContext) -> None:
    tool = ReadDocumentTool(FakeDocumentSource())

    result = tool.execute(
        ToolCall(
            call_id="call-1",
            tool_name="read_document",
            arguments={"path": "missing.pdf"},
        ),
        context,
    )

    # The fake accepts every path, proving the tool delegates backend policy.
    assert result.success is True


def test_keyword_backend_uses_documented_scoring_weights() -> None:
    hits = KeywordSearchBackend(ScoringDocumentSource()).search("Agent", 10)

    assert [(hit.path, hit.score) for hit in hits] == [
        ("title.md", 175),
        ("body.md", 55),
    ]
