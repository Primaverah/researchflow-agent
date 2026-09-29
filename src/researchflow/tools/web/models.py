"""Validated models for optional web search and page retrieval."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class WebSearchResult(BaseModel):
    """One candidate returned by a web search provider."""

    model_config = ConfigDict(extra="forbid")

    title: str = Field(min_length=1)
    url: str = Field(min_length=1)
    summary: str = ""


class WebSource(BaseModel):
    """A successfully fetched source eligible for report citation."""

    model_config = ConfigDict(extra="forbid")

    title: str = Field(min_length=1)
    url: str = Field(min_length=1)
    summary: str = ""
    accessed_at: datetime
    content: str = Field(min_length=1)
