"""Local filesystem and keyword-search backends."""

import os
import tempfile
import unicodedata
from pathlib import Path

from researchflow.tools.errors import ToolFailure, UnsafePathError
from researchflow.tools.offline.interfaces import Document, DocumentSource, SearchHit
from researchflow.tools.paths import resolve_safe_path

SUPPORTED_SUFFIXES = frozenset({".md", ".txt"})
SNIPPET_LENGTH = 160


def _normalize(text: str) -> str:
    return unicodedata.normalize("NFKC", text).casefold()


def _validate_suffix(path: str) -> None:
    if Path(path).suffix.casefold() not in SUPPORTED_SUFFIXES:
        raise ToolFailure(
            "only .md and .txt files are supported",
            error_type="unsupported_file_type",
        )


def _title_for(path: str, content: str) -> str:
    if Path(path).suffix.casefold() == ".md":
        for line in content.splitlines():
            stripped = line.strip()
            if stripped.startswith("# ") and stripped[2:].strip():
                return stripped[2:].strip()
    return Path(path).stem


class FileSystemDocumentSource:
    """Read UTF-8 Markdown and text documents below one root."""

    def __init__(self, root: Path) -> None:
        self._root = root

    def list_documents(self) -> list[str]:
        documents: list[str] = []
        for candidate in self._root.rglob("*"):
            if candidate.suffix.casefold() not in SUPPORTED_SUFFIXES:
                continue
            relative_path = candidate.relative_to(self._root).as_posix()
            try:
                safe_path = resolve_safe_path(
                    self._root, relative_path, must_exist=True
                )
            except UnsafePathError:
                continue
            if safe_path.is_file():
                documents.append(relative_path)
        return sorted(documents)

    def read_document(self, path: str) -> Document:
        try:
            safe_path = resolve_safe_path(self._root, path)
        except UnsafePathError as exc:
            raise ToolFailure(
                "document path is outside the allowed root",
                error_type="unsafe_path",
            ) from exc
        _validate_suffix(path)
        if not safe_path.exists():
            raise ToolFailure(
                "document does not exist", error_type="document_not_found"
            )
        if not safe_path.is_file():
            raise ToolFailure("document path is not a file", error_type="not_a_file")
        try:
            content = safe_path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise ToolFailure(
                "document is not valid UTF-8", error_type="invalid_encoding"
            ) from exc
        except OSError as exc:
            raise ToolFailure(
                "document could not be read", error_type="document_not_found"
            ) from exc
        relative_path = safe_path.relative_to(self._root.resolve()).as_posix()
        return Document(
            path=relative_path,
            title=_title_for(relative_path, content),
            content=content,
        )


class KeywordSearchBackend:
    """Apply deterministic keyword scoring to a document source."""

    def __init__(self, source: DocumentSource) -> None:
        self._source = source

    def search(self, query: str, limit: int) -> list[SearchHit]:
        normalized_query = _normalize(query)
        keywords = tuple(dict.fromkeys(normalized_query.split()))
        hits: list[SearchHit] = []
        for path in self._source.list_documents():
            try:
                document = self._source.read_document(path)
            except ToolFailure:
                continue
            title = _normalize(document.title)
            body = _normalize(document.content)
            score = self._score(normalized_query, keywords, title, body)
            if score <= 0:
                continue
            hits.append(
                SearchHit(
                    path=document.path,
                    title=document.title,
                    score=score,
                    snippet=self._snippet(document.content, normalized_query, keywords),
                )
            )
        hits.sort(key=lambda hit: (-hit.score, hit.path))
        return hits[:limit]

    @staticmethod
    def _score(
        query: str,
        keywords: tuple[str, ...],
        title: str,
        body: str,
    ) -> int:
        score = 0
        if query in title:
            score += 100
        if query in body:
            score += 50
        score += sum(title.count(keyword) * 20 for keyword in keywords)
        score += sum(body.count(keyword) * 5 for keyword in keywords)
        return score

    @staticmethod
    def _snippet(content: str, query: str, keywords: tuple[str, ...]) -> str:
        normalized_content = _normalize(content)
        positions = [
            position
            for term in (query, *keywords)
            if (position := normalized_content.find(term)) >= 0
        ]
        match_position = min(positions, default=0)
        start = max(0, match_position - 60)
        end = min(len(content), start + SNIPPET_LENGTH)
        if end - start < SNIPPET_LENGTH:
            start = max(0, end - SNIPPET_LENGTH)
        return content[start:end].replace("\n", " ")


class FileSystemNoteStore:
    """Atomically save UTF-8 notes below one output root."""

    def __init__(self, root: Path) -> None:
        self._root = root

    def save(self, path: str, content: str, overwrite: bool = False) -> str:
        try:
            target = resolve_safe_path(self._root, path)
        except UnsafePathError as exc:
            raise ToolFailure(
                "note path is outside the allowed root", error_type="unsafe_path"
            ) from exc
        _validate_suffix(path)
        if target.exists() and target.is_dir():
            raise ToolFailure("note path is a directory", error_type="not_a_file")
        if target.exists() and not overwrite:
            raise ToolFailure("note already exists", error_type="note_exists")

        temporary_path: Path | None = None
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            verified_target = resolve_safe_path(self._root, path)
            descriptor, temporary_name = tempfile.mkstemp(
                dir=verified_target.parent,
                prefix=".researchflow-",
                suffix=".tmp",
            )
            temporary_path = Path(temporary_name)
            with os.fdopen(descriptor, "w", encoding="utf-8") as temporary_file:
                temporary_file.write(content)
                temporary_file.flush()
                os.fsync(temporary_file.fileno())
            os.replace(temporary_path, verified_target)
        except UnsafePathError as exc:
            raise ToolFailure(
                "note path is outside the allowed root", error_type="unsafe_path"
            ) from exc
        except OSError as exc:
            raise ToolFailure(
                "note could not be saved", error_type="write_failed"
            ) from exc
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
        return target.relative_to(self._root.resolve()).as_posix()
