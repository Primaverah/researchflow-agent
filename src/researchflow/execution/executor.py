"""Unified synchronous entry point for traced tool calls."""

import logging
import time
from copy import deepcopy
from datetime import UTC, datetime
from uuid import uuid4

from researchflow.domain import (
    DecisionTrace,
    ExecutionStatus,
    ExecutionTrace,
    ToolCall,
    ToolResult,
)
from researchflow.execution.recorder import TraceRecorder
from researchflow.tools import ToolContext, ToolError, ToolNotFoundError, ToolRegistry
from researchflow.tools.errors import ToolValidationError

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class ToolExecutor:
    """Execute one registered tool and persist exactly one completed trace."""

    def __init__(self, registry: ToolRegistry, recorder: TraceRecorder) -> None:
        self._registry = registry
        self._recorder = recorder

    def execute(self, call: ToolCall, context: ToolContext) -> ToolResult:
        """Execute a tool call and record its result and duration."""
        result, _ = self.execute_with_trace(call, context)
        return result

    def record_decision(self, decision: DecisionTrace, context: ToolContext) -> None:
        """Persist sanitized agent decision metadata with the run trace."""
        self._recorder.record_decision(decision, context)

    def execute_with_trace(
        self, call: ToolCall, context: ToolContext
    ) -> tuple[ToolResult, ExecutionTrace]:
        """Execute one call and return the exact trace persisted by the recorder."""
        started_at = datetime.now(UTC)
        started_counter = time.perf_counter()
        original_arguments = self._trace_arguments(call)
        try:
            tool = self._registry.get(call.tool_name)
            result = tool.execute(call, context)
        except ToolNotFoundError as exc:
            result = self._failure_result(call, "tool_not_found", str(exc))
        except ToolValidationError as exc:
            result = self._failure_result(call, "tool_validation_error", str(exc))
        except ToolError as exc:
            error_type = getattr(exc, "error_type", "tool_execution_error")
            result = self._failure_result(call, error_type, str(exc))
        except Exception:
            logger.exception(
                "unexpected error while executing tool '%s'", call.tool_name
            )
            result = self._failure_result(
                call,
                "tool_execution_error",
                f"tool '{call.tool_name}' execution failed",
            )

        duration_ms = max(0.0, (time.perf_counter() - started_counter) * 1000)
        trace = self._build_trace(
            call,
            context,
            original_arguments,
            result,
            started_at,
            duration_ms,
        )
        self._recorder.record(trace, context)
        return result, trace

    @staticmethod
    def _trace_arguments(call: ToolCall) -> dict[str, object]:
        """Copy arguments while redacting note bodies from durable traces."""
        arguments = deepcopy(call.arguments)
        content = arguments.get("content")
        if call.tool_name == "save_note" and isinstance(content, str):
            arguments["content"] = {
                "redacted": True,
                "char_count": len(content),
            }
        return arguments

    @staticmethod
    def _failure_result(
        call: ToolCall,
        error_type: str,
        error_message: str,
    ) -> ToolResult:
        return ToolResult(
            call_id=call.call_id,
            tool_name=call.tool_name,
            success=False,
            error_type=error_type,
            error_message=error_message,
        )

    @staticmethod
    def _build_trace(
        call: ToolCall,
        context: ToolContext,
        arguments: dict[str, object],
        result: ToolResult,
        started_at: datetime,
        duration_ms: float,
    ) -> ExecutionTrace:
        trace_fields = {
            "trace_id": str(uuid4()),
            "run_id": context.run_id,
            "call_id": call.call_id,
            "tool_name": call.tool_name,
            "arguments": arguments,
            "started_at": started_at,
            "duration_ms": duration_ms,
        }
        if result.success:
            return ExecutionTrace(
                **trace_fields,
                status=ExecutionStatus.SUCCEEDED,
            )
        return ExecutionTrace(
            **trace_fields,
            status=ExecutionStatus.FAILED,
            error_type=result.error_type,
            error_message=result.error_message,
        )
