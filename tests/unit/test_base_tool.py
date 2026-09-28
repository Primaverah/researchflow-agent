"""Tests for the base tool execution contract."""

import pytest
from pydantic import BaseModel, ConfigDict

from researchflow.domain import ToolCall
from researchflow.tools import (
    BaseTool,
    ToolContext,
    ToolExecutionError,
    ToolFailure,
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
        raise ToolFailure("source is unavailable", error_type="source_unavailable")


class BrokenTool(ExampleTool):
    name = "broken"

    def _execute(self, arguments: BaseModel, context: ToolContext) -> object:
        raise RuntimeError("boom")


class UnsafeTool(ExampleTool):
    name = "unsafe"

    def _execute(self, arguments: BaseModel, context: ToolContext) -> object:
        raise UnsafePathError("unsafe path")


@pytest.mark.parametrize("name", ["", " ", "UPPER", "has space", "bang!"])
def test_tool_metadata_rejects_invalid_names(name: str) -> None:
    class InvalidNameTool(ExampleTool):
        pass

    InvalidNameTool.name = name

    with pytest.raises(ValueError, match="tool name"):
        InvalidNameTool()


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

    assert result.success is True
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


def test_expected_execution_failure_returns_structured_result(tmp_path) -> None:
    context = ToolContext(
        working_directory=tmp_path,
        output_directory=tmp_path,
        run_id="run-1",
    )

    result = FailingTool().execute(
        ToolCall(call_id="call-1", tool_name="failing", arguments={"value": 1}),
        context,
    )

    assert result.success is False
    assert result.call_id == "call-1"
    assert result.tool_name == "failing"
    assert result.output is None
    assert result.error_type == "source_unavailable"
    assert result.error_message == "source is unavailable"


def test_programming_errors_are_not_swallowed(tmp_path) -> None:
    context = ToolContext(
        working_directory=tmp_path,
        output_directory=tmp_path,
        run_id="run-1",
    )

    with pytest.raises(RuntimeError, match="boom"):
        BrokenTool().execute(
            ToolCall(call_id="call-1", tool_name="broken", arguments={"value": 1}),
            context,
        )


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
