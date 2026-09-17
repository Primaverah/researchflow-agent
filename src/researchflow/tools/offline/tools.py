"""Offline tool adapters backed by small injectable interfaces."""

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator

from researchflow.tools.base import BaseTool
from researchflow.tools.context import ToolContext
from researchflow.tools.offline.interfaces import (
    DocumentSource,
    NoteStore,
    SearchBackend,
)

NonEmptyString = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class SearchDocumentsArgs(BaseModel):
    """Arguments accepted by ``search_documents``."""

    model_config = ConfigDict(extra="forbid")

    query: NonEmptyString
    limit: int = Field(default=10, ge=1, le=100)


class ReadDocumentArgs(BaseModel):
    """Arguments accepted by ``read_document``."""

    model_config = ConfigDict(extra="forbid")

    path: NonEmptyString


class SaveNoteArgs(BaseModel):
    """Arguments accepted by ``save_note``."""

    model_config = ConfigDict(extra="forbid")

    path: NonEmptyString
    content: str
    overwrite: bool = False

    @field_validator("content")
    @classmethod
    def reject_blank_content(cls, value: str) -> str:
        """Reject blank notes without changing the user's content."""
        if not value.strip():
            raise ValueError("content cannot be blank")
        return value


class SearchHitOutput(BaseModel):
    """JSON-safe representation of one search hit."""

    model_config = ConfigDict(extra="forbid")

    path: str
    title: str
    score: int
    snippet: str


class SearchDocumentsOutput(BaseModel):
    """Output returned by ``search_documents``."""

    model_config = ConfigDict(extra="forbid")

    query: str
    count: int
    hits: list[SearchHitOutput]


class ReadDocumentOutput(BaseModel):
    """Output returned by ``read_document``."""

    model_config = ConfigDict(extra="forbid")

    path: str
    title: str
    content: str
    char_count: int


class SaveNoteOutput(BaseModel):
    """Output returned by ``save_note``."""

    model_config = ConfigDict(extra="forbid")

    path: str
    char_count: int


class SearchDocumentsTool(BaseTool):
    """Search documents through an injected search backend."""

    name = "search_documents"
    description = "Search local technical documents using stable keyword scoring."
    args_schema = SearchDocumentsArgs

    def __init__(self, backend: SearchBackend) -> None:
        super().__init__()
        self._backend = backend

    def _execute(self, arguments: BaseModel, context: ToolContext) -> object:
        parsed = SearchDocumentsArgs.model_validate(arguments)
        hits = self._backend.search(parsed.query, parsed.limit)
        serialized_hits = [
            SearchHitOutput(
                path=hit.path,
                title=hit.title,
                score=hit.score,
                snippet=hit.snippet,
            )
            for hit in hits
        ]
        return SearchDocumentsOutput(
            query=parsed.query,
            count=len(hits),
            hits=serialized_hits,
        ).model_dump(mode="json")


class ReadDocumentTool(BaseTool):
    """Read one document through an injected document source."""

    name = "read_document"
    description = "Read one UTF-8 Markdown or text document."
    args_schema = ReadDocumentArgs

    def __init__(self, source: DocumentSource) -> None:
        super().__init__()
        self._source = source

    def _execute(self, arguments: BaseModel, context: ToolContext) -> object:
        parsed = ReadDocumentArgs.model_validate(arguments)
        document = self._source.read_document(parsed.path)
        return ReadDocumentOutput(
            path=document.path,
            title=document.title,
            content=document.content,
            char_count=len(document.content),
        ).model_dump(mode="json")


class SaveNoteTool(BaseTool):
    """Save one research note through an injected note store."""

    name = "save_note"
    description = "Atomically save one UTF-8 Markdown or text research note."
    args_schema = SaveNoteArgs

    def __init__(self, store: NoteStore) -> None:
        super().__init__()
        self._store = store

    def _execute(self, arguments: BaseModel, context: ToolContext) -> object:
        parsed = SaveNoteArgs.model_validate(arguments)
        path = self._store.save(parsed.path, parsed.content, parsed.overwrite)
        return SaveNoteOutput(
            path=path,
            char_count=len(parsed.content),
        ).model_dump(mode="json")
