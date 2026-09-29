"""Factory for opt-in web tools."""

from researchflow.tools.base import BaseTool
from researchflow.tools.web.http import SafeHttpClient
from researchflow.tools.web.provider import WebSearchProvider
from researchflow.tools.web.tools import FetchUrlTool, WebSearchTool


def create_web_tools(
    provider: WebSearchProvider, client: SafeHttpClient | None = None
) -> tuple[BaseTool, ...]:
    return (WebSearchTool(provider), FetchUrlTool(client or SafeHttpClient()))
