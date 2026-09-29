"""Optional web search and safe HTML retrieval tools."""

from researchflow.tools.web.factory import create_web_tools
from researchflow.tools.web.http import HtmlPage, SafeHttpClient, WebFetchError
from researchflow.tools.web.models import WebSearchResult, WebSource
from researchflow.tools.web.provider import (
    TavilySearchProvider,
    WebSearchConfigurationError,
    WebSearchError,
    WebSearchProvider,
)

__all__ = [
    "TavilySearchProvider",
    "SafeHttpClient",
    "HtmlPage",
    "WebFetchError",
    "WebSearchConfigurationError",
    "WebSearchError",
    "WebSearchProvider",
    "WebSearchResult",
    "WebSource",
    "create_web_tools",
]
