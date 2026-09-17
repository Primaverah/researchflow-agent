"""Read-only context supplied to tools at execution time."""

from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from types import MappingProxyType
from typing import Annotated, Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_serializer,
    field_validator,
)

NonEmptyString = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


def _freeze(value: Any) -> Any:
    """Create an immutable copy of JSON-like context data."""
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list | tuple):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, set | frozenset):
        return frozenset(_freeze(item) for item in value)
    return deepcopy(value)


def _thaw(value: Any) -> Any:
    """Convert frozen context data back to JSON-friendly containers."""
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple | frozenset):
        return [_thaw(item) for item in value]
    return value


class ToolContext(BaseModel):
    """Directories and immutable run data available to a tool."""

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    working_directory: Path
    output_directory: Path
    run_id: NonEmptyString
    metadata: Mapping[str, Any] = Field(default_factory=lambda: MappingProxyType({}))

    @field_validator("working_directory", "output_directory")
    @classmethod
    def validate_directory(cls, value: Path) -> Path:
        """Resolve and require an existing directory."""
        resolved = value.expanduser().resolve(strict=False)
        if not resolved.is_dir():
            raise ValueError(f"directory does not exist or is not a directory: {value}")
        return resolved

    @field_validator("metadata", mode="after")
    @classmethod
    def freeze_metadata(cls, value: Mapping[str, Any]) -> Mapping[str, Any]:
        """Prevent callers and tools from mutating contextual metadata."""
        return _freeze(value)

    @field_serializer("metadata")
    def serialize_metadata(self, value: Mapping[str, Any]) -> dict[str, Any]:
        """Serialize immutable metadata using JSON-friendly containers."""
        return _thaw(value)
