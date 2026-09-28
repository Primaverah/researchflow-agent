"""Validated argument and output models for offline tools."""

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator

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
