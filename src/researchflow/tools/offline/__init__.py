"""Public API for ResearchFlow's offline tools and local backends."""

from researchflow.tools.offline.backends import (
    FileSystemDocumentSource,
    FileSystemNoteStore,
    KeywordSearchBackend,
)
from researchflow.tools.offline.factory import create_offline_tools
from researchflow.tools.offline.interfaces import (
    DocumentSource,
    NoteStore,
    SearchBackend,
)
from researchflow.tools.offline.tools import (
    ReadDocumentArgs,
    ReadDocumentOutput,
    ReadDocumentTool,
    SaveNoteArgs,
    SaveNoteOutput,
    SaveNoteTool,
    SearchDocumentsArgs,
    SearchDocumentsOutput,
    SearchDocumentsTool,
    SearchHitOutput,
)

__all__ = [
    "DocumentSource",
    "FileSystemDocumentSource",
    "FileSystemNoteStore",
    "KeywordSearchBackend",
    "NoteStore",
    "ReadDocumentArgs",
    "ReadDocumentOutput",
    "ReadDocumentTool",
    "SaveNoteArgs",
    "SaveNoteOutput",
    "SaveNoteTool",
    "SearchBackend",
    "SearchDocumentsArgs",
    "SearchDocumentsOutput",
    "SearchDocumentsTool",
    "SearchHitOutput",
    "create_offline_tools",
]
