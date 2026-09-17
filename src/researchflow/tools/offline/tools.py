"""Offline tool adapters backed by small injectable interfaces."""

from pydantic import BaseModel

from researchflow.tools.base import BaseTool
from researchflow.tools.context import ToolContext
from researchflow.tools.offline.interfaces import (
    DocumentSource,
    NoteStore,
    SearchBackend,
)
from researchflow.tools.offline.models import (
    ReadDocumentArgs,
    ReadDocumentOutput,
    SaveNoteArgs,
    SaveNoteOutput,
    SearchDocumentsArgs,
    SearchDocumentsOutput,
    SearchHitOutput,
)


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
