"""Factory for the standard local offline tool set."""

from researchflow.tools.base import BaseTool
from researchflow.tools.context import ToolContext
from researchflow.tools.offline.backends import (
    FileSystemDocumentSource,
    FileSystemNoteStore,
    KeywordSearchBackend,
)
from researchflow.tools.offline.tools import (
    ReadDocumentTool,
    SaveNoteTool,
    SearchDocumentsTool,
)


def create_offline_tools(context: ToolContext) -> tuple[BaseTool, ...]:
    """Create the three filesystem-backed offline tools for a run context."""
    source = FileSystemDocumentSource(context.working_directory)
    search_backend = KeywordSearchBackend(source)
    note_store = FileSystemNoteStore(context.output_directory)
    return (
        SearchDocumentsTool(search_backend),
        ReadDocumentTool(source),
        SaveNoteTool(note_store),
    )
