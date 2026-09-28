"""Explicit in-memory registry for tool instances."""

from researchflow.tools.base import BaseTool
from researchflow.tools.errors import DuplicateToolError, ToolNotFoundError


class ToolRegistry:
    """Register and retrieve tools by their stable names."""

    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        """Register one tool without allowing silent replacement."""
        if not isinstance(tool, BaseTool):
            raise TypeError("tool must be an instance of BaseTool")
        if tool.name in self._tools:
            raise DuplicateToolError(tool.name)
        self._tools[tool.name] = tool

    def get(self, name: str) -> BaseTool:
        """Return a registered tool or raise a stable lookup error."""
        try:
            return self._tools[name]
        except KeyError as exc:
            raise ToolNotFoundError(name, tuple(sorted(self._tools))) from exc

    def list_tools(self) -> tuple[BaseTool, ...]:
        """Return registered tools in deterministic name order."""
        return tuple(self._tools[name] for name in sorted(self._tools))
