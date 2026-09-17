"""Tool framework exceptions."""

from typing import Any


class ResearchFlowError(Exception):
    """Base exception for expected ResearchFlow failures."""


class ToolError(ResearchFlowError):
    """Base exception for tool framework failures."""


class ToolValidationError(ToolError):
    """Raised when a tool call does not match its argument schema."""

    def __init__(
        self,
        tool_name: str,
        validation_errors: list[dict[str, Any]],
    ) -> None:
        self.tool_name = tool_name
        self.validation_errors = validation_errors
        super().__init__(f"invalid arguments for tool '{tool_name}'")


class ToolNotFoundError(ToolError):
    """Raised when a registry cannot find a requested tool."""

    def __init__(self, tool_name: str, available_tools: tuple[str, ...] = ()) -> None:
        self.tool_name = tool_name
        self.available_tools = available_tools
        super().__init__(f"tool '{tool_name}' is not registered")


class DuplicateToolError(ToolError):
    """Raised when a tool name is already registered."""

    def __init__(self, tool_name: str) -> None:
        self.tool_name = tool_name
        super().__init__(f"tool '{tool_name}' is already registered")


class ToolExecutionError(ToolError):
    """Raised when a validated tool call fails during execution."""

    def __init__(self, tool_name: str, call_id: str, message: str) -> None:
        self.tool_name = tool_name
        self.call_id = call_id
        super().__init__(message)


class UnsafePathError(ToolError):
    """Raised when a requested path escapes its allowed root."""
