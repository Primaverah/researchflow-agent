"""Optional Tavily REST adapter behind a small synchronous protocol."""

import json
import os
from collections.abc import Callable
from typing import Protocol
from urllib.request import Request, urlopen

from researchflow.tools.web.models import WebSearchResult

TAVILY_SEARCH_URL = "https://api.tavily.com/search"


class WebSearchError(RuntimeError):
    """Safe error raised when optional web search cannot complete."""


class WebSearchConfigurationError(WebSearchError):
    """Raised when explicitly requested web search lacks configuration."""


class WebSearchProvider(Protocol):
    """Synchronous provider for normalized web search candidates."""

    def search(self, query: str, limit: int) -> list[WebSearchResult]: ...


class TavilySearchProvider:
    """Minimal Tavily REST provider loaded only by explicit web mode."""

    def __init__(
        self,
        api_key: str,
        *,
        opener: Callable[..., object] = urlopen,
        timeout: float = 10,
    ) -> None:
        if not api_key.strip():
            raise WebSearchConfigurationError(
                "RESEARCHFLOW_SEARCH_API_KEY is required for web search"
            )
        self._api_key = api_key
        self._opener = opener
        self._timeout = timeout

    @classmethod
    def from_environment(cls) -> "TavilySearchProvider":
        """Construct the adapter from the only supported credential location."""
        return cls(os.getenv("RESEARCHFLOW_SEARCH_API_KEY", ""))

    def search(self, query: str, limit: int) -> list[WebSearchResult]:
        """Search Tavily and normalize its result fields without leaking secrets."""
        payload = json.dumps(
            {"api_key": self._api_key, "query": query, "max_results": limit}
        ).encode("utf-8")
        request = Request(
            TAVILY_SEARCH_URL,
            data=payload,
            headers={"Content-Type": "application/json; charset=utf-8"},
            method="POST",
        )
        try:
            with self._opener(request, timeout=self._timeout) as response:
                raw_response = response.read()
            decoded = json.loads(raw_response.decode("utf-8"))
            results = decoded.get("results", [])
            if not isinstance(results, list):
                raise ValueError("results must be a list")
            return [
                WebSearchResult(
                    title=item["title"],
                    url=item["url"],
                    summary=item.get("content", ""),
                )
                for item in results
                if isinstance(item, dict)
                and isinstance(item.get("title"), str)
                and isinstance(item.get("url"), str)
            ]
        except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
            raise WebSearchError("web search request failed") from exc
