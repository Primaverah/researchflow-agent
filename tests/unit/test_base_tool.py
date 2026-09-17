"""Tests for the base tool execution contract."""

import pytest
from pydantic import BaseModel, ConfigDict

from researchflow.domain import ToolCall, ToolResultStatus
from researchflow.tools import (
    BaseTool,
    ToolContext,
    ToolExecutionError,
    ToolValidationError,
    UnsafePathError,
)


class ExampleArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    value: int


class ExampleTool(BaseTool):
    name = "example"
    description = "Return a doubled value."
    args_schema = ExampleArgs

    def _execute(self, arguments: BaseModel, context: ToolContext) -> object:
        parsed = ExampleArgs.model_validate(arguments)
        return {"value": parsed.value * 2, "run_id": context.run_id}


class FailingTool(ExampleTool):
    name = "failing"

    def _execute(self, arguments: BaseModel, context: ToolContext) -> object:
        raise RuntimeError("boom")


class UnsafeTool(ExampleTool):
    name = "unsafe"

    def _execute(self, arguments: BaseModel, context: ToolContext) -> object:
        raise UnsafePathError("unsafe path")


def test_execute_validates_arguments_and_returns_result(tmp_path) -> None:
    context = ToolContext(
        working_directory=tmp_path,
        output_directory=tmp_path,
        run_id="run-1",
    )

    result = ExampleTool().execute(
        ToolCall(call_id="call-1", tool_name="example", arguments={"value": "2"}),
        context,
    )

    assert result.status is ToolResultStatus.SUCCEEDED
    assert result.call_id == "call-1"
    assert result.output == {"value": 4, "run_id": "run-1"}


def test_invalid_arguments_raise_tool_validation_error(tmp_path) -> None:
    context = ToolContext(
        working_directory=tmp_path,
        output_directory=tmp_path,
        run_id="run-1",
    )

    with pytest.raises(ToolValidationError) as exc_info:
        ExampleTool().execute(
            ToolCall(
                call_id="call-1",
                tool_name="example",
                arguments={"value": 1, "extra": True},
            ),
            context,
        )

    assert exc_info.value.tool_name == "example"
    assert exc_info.value.validation_errors
    assert exc_info.value.__cause__ is not None


def test_unexpected_execution_error_is_wrapped(tmp_path) -> None:
    context = ToolContext(
        working_directory=tmp_path,
        output_directory=tmp_path,
        run_id="run-1",
    )

    with pytest.raises(ToolExecutionError) as exc_info:
        FailingTool().execute(
            ToolCall(call_id="call-1", tool_name="failing", arguments={"value": 1}),
            context,
        )

    assert isinstance(exc_info.value.__cause__, RuntimeError)
    assert exc_info.value.call_id == "call-1"


def test_framework_errors_are_not_wrapped(tmp_path) -> None:
    context = ToolContext(
        working_directory=tmp_path,
        output_directory=tmp_path,
        run_id="run-1",
    )

    with pytest.raises(UnsafePathError):
        UnsafeTool().execute(
            ToolCall(call_id="call-1", tool_name="unsafe", arguments={"value": 1}),
            context,
        )


def test_call_name_must_match_tool(tmp_path) -> None:
    context = ToolContext(
        working_directory=tmp_path,
        output_directory=tmp_path,
        run_id="run-1",
    )

    with pytest.raises(ToolExecutionError, match="does not match"):
        ExampleTool().execute(
            ToolCall(call_id="call-1", tool_name="different", arguments={"value": 1}),
            context,
        )
