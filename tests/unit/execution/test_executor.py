"""Tests for the traced tool executor."""

from pathlib import Path

import pytest
from pydantic import BaseModel, ConfigDict

from researchflow.domain import ExecutionStatus, ExecutionTrace, ToolCall
from researchflow.execution import ToolExecutor
from researchflow.tools import BaseTool, ToolContext, ToolFailure, ToolRegistry


class ExampleArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    value: int


class CountingTool(BaseTool):
    name = "counting"
    description = "Count executions and return a value."
    args_schema = ExampleArgs

    def __init__(self) -> None:
        super().__init__()
        self.execution_count = 0

    def _execute(self, arguments: BaseModel, context: ToolContext) -> object:
        self.execution_count += 1
        parsed = ExampleArgs.model_validate(arguments)
        return {"value": parsed.value}


class ExpectedFailureTool(CountingTool):
    name = "expected_failure"

    def _execute(self, arguments: BaseModel, context: ToolContext) -> object:
        self.execution_count += 1
        raise ToolFailure("source unavailable", error_type="source_unavailable")


class BrokenTool(CountingTool):
    name = "broken"

    def _execute(self, arguments: BaseModel, context: ToolContext) -> object:
        self.execution_count += 1
        raise RuntimeError("boom")


class MemoryRecorder:
    def __init__(self) -> None:
        self.traces: list[ExecutionTrace] = []

    def record(self, trace: ExecutionTrace, context: ToolContext) -> None:
        self.traces.append(trace)


@pytest.fixture
def context(tmp_path: Path) -> ToolContext:
    return ToolContext(
        working_directory=tmp_path,
        output_directory=tmp_path,
        run_id="run-1",
    )


def build_executor(tool: BaseTool | None = None) -> tuple[ToolExecutor, MemoryRecorder]:
    registry = ToolRegistry()
    if tool is not None:
        registry.register(tool)
    recorder = MemoryRecorder()
    return ToolExecutor(registry, recorder), recorder


def test_success_returns_result_and_records_one_trace(context: ToolContext) -> None:
    tool = CountingTool()
    executor, recorder = build_executor(tool)
    call = ToolCall(call_id="call-1", tool_name="counting", arguments={"value": 2})

    result = executor.execute(call, context)

    assert result.success is True
    assert result.output == {"value": 2}
    assert tool.execution_count == 1
    assert len(recorder.traces) == 1
    trace = recorder.traces[0]
    assert trace.status is ExecutionStatus.SUCCEEDED
    assert trace.run_id == "run-1"
    assert trace.call_id == "call-1"
    assert trace.tool_name == "counting"
    assert trace.arguments == {"value": 2}
    assert trace.duration_ms >= 0
    assert trace.error_type is None


def test_each_call_has_a_unique_trace_id(context: ToolContext) -> None:
    executor, recorder = build_executor(CountingTool())
    call = ToolCall(call_id="call-1", tool_name="counting", arguments={"value": 2})

    executor.execute(call, context)
    executor.execute(call, context)

    assert len(recorder.traces) == 2
    assert recorder.traces[0].trace_id != recorder.traces[1].trace_id


def test_validation_failure_does_not_execute_tool(context: ToolContext) -> None:
    tool = CountingTool()
    executor, recorder = build_executor(tool)

    result = executor.execute(
        ToolCall(call_id="call-1", tool_name="counting", arguments={"value": "bad"}),
        context,
    )

    assert result.success is False
    assert result.error_type == "tool_validation_error"
    assert tool.execution_count == 0
    assert recorder.traces[0].status is ExecutionStatus.FAILED
    assert recorder.traces[0].error_type == "tool_validation_error"


def test_expected_failure_preserves_error(context: ToolContext) -> None:
    tool = ExpectedFailureTool()
    executor, recorder = build_executor(tool)

    result = executor.execute(
        ToolCall(
            call_id="call-1", tool_name="expected_failure", arguments={"value": 1}
        ),
        context,
    )

    assert result.success is False
    assert result.error_type == "source_unavailable"
    assert recorder.traces[0].error_message == "source unavailable"
    assert tool.execution_count == 1


def test_unexpected_exception_becomes_failure(context: ToolContext) -> None:
    tool = BrokenTool()
    executor, recorder = build_executor(tool)

    result = executor.execute(
        ToolCall(call_id="call-1", tool_name="broken", arguments={"value": 1}),
        context,
    )

    assert result.success is False
    assert result.error_type == "tool_execution_error"
    assert "boom" not in result.error_message
    assert recorder.traces[0].status is ExecutionStatus.FAILED
    assert tool.execution_count == 1


def test_unknown_tool_becomes_failure(context: ToolContext) -> None:
    executor, recorder = build_executor()

    result = executor.execute(
        ToolCall(call_id="call-1", tool_name="missing", arguments={"query": "中文"}),
        context,
    )

    assert result.success is False
    assert result.error_type == "tool_not_found"
    assert recorder.traces[0].arguments == {"query": "中文"}
    assert recorder.traces[0].status is ExecutionStatus.FAILED


def test_keyboard_interrupt_is_not_captured(context: ToolContext) -> None:
    class InterruptingTool(CountingTool):
        name = "interrupting"

        def _execute(self, arguments: BaseModel, context: ToolContext) -> object:
            raise KeyboardInterrupt

    executor, recorder = build_executor(InterruptingTool())

    with pytest.raises(KeyboardInterrupt):
        executor.execute(
            ToolCall(
                call_id="call-1", tool_name="interrupting", arguments={"value": 1}
            ),
            context,
        )
    assert recorder.traces == []
