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
from researchflow.tools.offline.models import (
    ReadDocumentArgs,
    ReadDocumentOutput,
    SaveNoteArgs,
    SaveNoteOutput,
    SearchDocumentsArgs,
    SearchDocumentsOutput,
    SearchHitOutput,
)
from researchflow.tools.offline.tools import (
    ReadDocumentTool,
    SaveNoteTool,
    SearchDocumentsTool,
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
