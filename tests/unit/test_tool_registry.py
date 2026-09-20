"""Tests for explicit tool registration."""

import pytest
from pydantic import BaseModel

from researchflow.tools import (
    BaseTool,
    DuplicateToolError,
    ToolContext,
    ToolNotFoundError,
    ToolRegistry,
)


class NoArgs(BaseModel):
    pass


class AlphaTool(BaseTool):
    name = "alpha"
    description = "Alpha test tool."
    args_schema = NoArgs

    def _execute(self, arguments: BaseModel, context: ToolContext) -> object:
        return None


class BetaTool(AlphaTool):
    name = "beta"


def test_register_get_and_list_tools() -> None:
    registry = ToolRegistry()
    beta = BetaTool()
    alpha = AlphaTool()

    registry.register(beta)
    registry.register(alpha)

    assert registry.get("alpha") is alpha
    assert registry.list_tools() == (alpha, beta)


def test_duplicate_tool_name_is_rejected() -> None:
    registry = ToolRegistry()
    original = AlphaTool()
    registry.register(original)

    with pytest.raises(DuplicateToolError, match="alpha"):
        registry.register(AlphaTool())

    assert registry.get("alpha") is original
    assert registry.list_tools() == (original,)


def test_unknown_tool_is_rejected() -> None:
    registry = ToolRegistry()
    original = AlphaTool()
    registry.register(original)

    with pytest.raises(ToolNotFoundError, match="missing"):
        registry.get("missing")

    assert registry.get("alpha") is original
    assert registry.list_tools() == (original,)


def test_registry_rejects_non_tools() -> None:
    registry = ToolRegistry()

    with pytest.raises(TypeError, match="BaseTool"):
        registry.register(object())  # type: ignore[arg-type]
