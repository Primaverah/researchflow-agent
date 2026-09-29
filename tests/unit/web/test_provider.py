"""Tests for the optional Tavily web search provider."""

import json

import pytest

from researchflow.tools.web.provider import (
    TavilySearchProvider,
    WebSearchConfigurationError,
    WebSearchError,
)


class FakeResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self._payload = payload

    def read(self) -> bytes:
        return json.dumps(self._payload, ensure_ascii=False).encode("utf-8")

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *args: object) -> None:
        return None


def test_tavily_search_posts_environment_key_and_maps_utf8_results() -> None:
    requests = []

    def opener(request, timeout: float):
        requests.append((request, timeout))
        return FakeResponse(
            {
                "results": [
                    {
                        "title": "中文来源",
                        "url": "https://example.com/article",
                        "content": "安全的网页摘要",
                    }
                ]
            }
        )

    provider = TavilySearchProvider("test-key", opener=opener)

    results = provider.search("网页安全", 3)

    assert [(item.title, item.url, item.summary) for item in results] == [
        ("中文来源", "https://example.com/article", "安全的网页摘要")
    ]
    request, timeout = requests[0]
    assert timeout == 10
    assert request.full_url == "https://api.tavily.com/search"
    assert json.loads(request.data.decode("utf-8")) == {
        "api_key": "test-key",
        "query": "网页安全",
        "max_results": 3,
    }


def test_tavily_provider_requires_environment_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("RESEARCHFLOW_SEARCH_API_KEY", raising=False)

    with pytest.raises(WebSearchConfigurationError, match="SEARCH_API_KEY"):
        TavilySearchProvider.from_environment()


def test_tavily_provider_hides_remote_errors() -> None:
    def failing_opener(request, timeout: float):
        raise OSError("key=test-key")

    provider = TavilySearchProvider("test-key", opener=failing_opener)

    with pytest.raises(WebSearchError, match="web search request failed") as error:
        provider.search("web security", 1)

    assert "test-key" not in str(error.value)
