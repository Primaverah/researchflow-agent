"""Small extension interfaces and data objects for offline tools."""

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class Document:
    """A UTF-8 document loaded from a document source."""

    path: str
    title: str
    content: str


@dataclass(frozen=True, slots=True)
class SearchHit:
    """One backend search result."""

    path: str
    title: str
    score: int
    snippet: str


class DocumentSource(Protocol):
    """Source capable of listing and reading documents."""

    def list_documents(self) -> list[str]:
        """Return stable relative paths for available documents."""
        ...

    def read_document(self, path: str) -> Document:
        """Read one document by its relative path."""
        ...


class SearchBackend(Protocol):
    """Backend capable of searching documents."""

    def search(self, query: str, limit: int) -> list[SearchHit]:
        """Return up to ``limit`` stable search hits."""
        ...


class NoteStore(Protocol):
    """Store capable of saving research notes."""

    def save(self, path: str, content: str, overwrite: bool = False) -> str:
        """Save content and return its relative path."""
        ...
