"""Abstract contract for synchronous ResearchFlow tools."""

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any, ClassVar

from pydantic import BaseModel, TypeAdapter, ValidationError

from researchflow.domain import ToolCall, ToolName, ToolResult
from researchflow.tools.context import ToolContext
from researchflow.tools.errors import (
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
        try:
            TypeAdapter(ToolName).validate_python(self.name)
        except ValidationError as exc:
            raise ValueError(
                "tool name must contain only lowercase letters, numbers, or underscores"
            ) from exc
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
                success=False,
                error_type=exc.error_type,
                error_message=str(exc),
            )
        return ToolResult(
            call_id=call.call_id,
            tool_name=self.name,
            success=True,
            output=output,
        )

    @abstractmethod
    def _execute(self, arguments: BaseModel, context: ToolContext) -> object:
        """Run the concrete tool with validated arguments."""
