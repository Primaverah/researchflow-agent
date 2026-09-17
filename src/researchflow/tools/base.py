"""Abstract contract for synchronous ResearchFlow tools."""

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any, ClassVar

from pydantic import BaseModel, ValidationError

from researchflow.domain import ToolCall, ToolResult, ToolResultStatus
from researchflow.tools.context import ToolContext
from researchflow.tools.errors import (
    ToolError,
    ToolExecutionError,
    ToolFailure,
    ToolValidationError,
)


class BaseTool(ABC):
    """Validate and execute one named tool synchronously."""

    name: ClassVar[str] = ""
    description: ClassVar[str] = ""
    args_schema: ClassVar[type[BaseModel]] = BaseModel

    def __init__(self) -> None:
        """Validate the metadata declared by a concrete tool."""
        if not self.name or not self.name.replace("_", "").isalnum():
            raise ValueError("tool name must contain letters, numbers, or underscores")
        if self.name != self.name.lower():
            raise ValueError("tool name must be lowercase")
        if not self.description.strip():
            raise ValueError("tool description cannot be empty")
        if not isinstance(self.args_schema, type) or not issubclass(
            self.args_schema, BaseModel
        ):
            raise TypeError("args_schema must be a Pydantic BaseModel subclass")

    def validate_arguments(self, arguments: Mapping[str, Any]) -> BaseModel:
        """Validate raw arguments with the tool's declared Pydantic schema."""
        try:
            return self.args_schema.model_validate(dict(arguments))
        except ValidationError as exc:
            raise ToolValidationError(self.name, exc.errors()) from exc

    def execute(self, call: ToolCall, context: ToolContext) -> ToolResult:
        """Validate a call, run the tool, and normalize its successful result."""
        if call.tool_name != self.name:
            raise ToolExecutionError(
                self.name,
                call.call_id,
                f"tool call name '{call.tool_name}' does not match '{self.name}'",
            )
        arguments = self.validate_arguments(call.arguments)
        try:
            output = self._execute(arguments, context)
        except ToolFailure as exc:
            return ToolResult(
                call_id=call.call_id,
                tool_name=self.name,
                status=ToolResultStatus.FAILED,
                error_type=exc.error_type,
                error_message=str(exc),
            )
        except ToolError:
            raise
        except Exception as exc:
            raise ToolExecutionError(
                self.name,
                call.call_id,
                f"tool '{self.name}' execution failed",
            ) from exc
        return ToolResult(
            call_id=call.call_id,
            tool_name=self.name,
            status=ToolResultStatus.SUCCEEDED,
            output=output,
        )

    @abstractmethod
    def _execute(self, arguments: BaseModel, context: ToolContext) -> object:
        """Run the concrete tool with validated arguments."""
