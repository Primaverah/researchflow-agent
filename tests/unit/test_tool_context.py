"""Tests for immutable tool execution context."""

import pytest
from pydantic import ValidationError

from researchflow.tools import ToolContext


def test_context_resolves_directories_and_freezes_metadata(tmp_path) -> None:
    metadata = {"source": "test", "options": ["one"]}
    context = ToolContext(
        working_directory=tmp_path,
        output_directory=tmp_path,
        run_id="run-1",
        metadata=metadata,
    )
    metadata["source"] = "changed"

    assert context.working_directory == tmp_path.resolve()
    assert context.metadata["source"] == "test"
    assert context.metadata["options"] == ("one",)
    assert context.model_dump()["metadata"] == {
        "source": "test",
        "options": ["one"],
    }
    with pytest.raises(TypeError):
        context.metadata["source"] = "changed"  # type: ignore[index]
    with pytest.raises(ValidationError):
        context.run_id = "run-2"


def test_context_requires_existing_directories(tmp_path) -> None:
    with pytest.raises(ValidationError, match="directory"):
        ToolContext(
            working_directory=tmp_path / "missing",
            output_directory=tmp_path,
            run_id="run-1",
        )


def test_context_rejects_blank_run_id(tmp_path) -> None:
    with pytest.raises(ValidationError):
        ToolContext(
            working_directory=tmp_path,
            output_directory=tmp_path,
            run_id=" ",
        )
