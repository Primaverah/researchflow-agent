"""Tests for deterministic domain checks at web tool boundaries."""

from pathlib import Path

from researchflow.domain import ToolCall
from researchflow.tools import ToolContext
from researchflow.tools.web.http import HtmlPage
from researchflow.tools.web.models import WebSearchResult
from researchflow.tools.web.tools import FetchUrlTool, WebSearchTool


class FakeSearchProvider:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int, tuple[str, ...]]] = []

    def search(
        self, query: str, limit: int, allowed_domains: tuple[str, ...] = ()
    ) -> list[WebSearchResult]:
        self.calls.append((query, limit, allowed_domains))
        return [
            WebSearchResult(title="Official", url="https://docs.python.org/3/"),
            WebSearchResult(title="Other", url="https://example.com/python"),
        ]


class FailingClient:
    def fetch(self, url: str):
        raise AssertionError("blocked domains must not reach the HTTP client")


class OfficialClient:
    def __init__(self) -> None:
        self.urls: list[str] = []

    def fetch(self, url: str) -> HtmlPage:
        self.urls.append(url)
        return HtmlPage(
            title="Python docs",
            content="Official Python documentation evidence.",
            url=url,
        )


class RedirectedClient:
    def fetch(self, url: str) -> HtmlPage:
        return HtmlPage(
            title="Unexpected source",
            content="Untrusted content.",
            url="https://example.com/redirected",
        )


def context(tmp_path: Path) -> ToolContext:
    return ToolContext(
        working_directory=tmp_path,
        output_directory=tmp_path,
        run_id="run-1",
    )


def test_web_search_tool_filters_results_to_allowed_domains(tmp_path: Path) -> None:
    provider = FakeSearchProvider()
    result = WebSearchTool(provider, allowed_domains=("docs.python.org",)).execute(
        ToolCall(
            call_id="search", tool_name="web_search", arguments={"query": "Python"}
        ),
        context(tmp_path),
    )

    assert result.success
    assert result.output["count"] == 1
    assert result.output["results"][0]["url"] == "https://docs.python.org/3/"
    assert provider.calls == [("Python", 5, ("docs.python.org",))]


def test_web_search_retries_once_with_simpler_query_for_allowed_results(
    tmp_path: Path,
) -> None:
    class RetrySearchProvider:
        def __init__(self) -> None:
            self.calls: list[tuple[str, tuple[str, ...]]] = []

        def search(
            self, query: str, limit: int, allowed_domains: tuple[str, ...] = ()
        ) -> list[WebSearchResult]:
            self.calls.append((query, allowed_domains))
            if len(self.calls) == 1:
                return [
                    WebSearchResult(title="Other", url="https://example.com/python")
                ]
            return [
                WebSearchResult(
                    title="Official",
                    url="https://docs.python.org/3.13/whatsnew/3.13.html",
                )
            ]

    provider = RetrySearchProvider()
    result = WebSearchTool(provider, allowed_domains=("docs.python.org",)).execute(
        ToolCall(
            call_id="search",
            tool_name="web_search",
            arguments={"query": "How to use Python 3.13 documentation", "limit": 5},
        ),
        context(tmp_path),
    )

    assert result.success
    assert result.output["results"] == [
        {
            "title": "Official",
            "url": "https://docs.python.org/3.13/whatsnew/3.13.html",
            "summary": "",
        }
    ]
    assert provider.calls == [
        ("How to use Python 3.13 documentation", ("docs.python.org",)),
        ("Python 3.13", ("docs.python.org",)),
    ]


def test_web_search_prefers_version_matched_simplified_chinese_page(
    tmp_path: Path,
) -> None:
    class VersionedSearchProvider:
        def search(
            self, query: str, limit: int, allowed_domains: tuple[str, ...] = ()
        ) -> list[WebSearchResult]:
            return [
                WebSearchResult(
                    title="What’s New In Python 3.12",
                    url="https://docs.python.org/3.12/whatsnew/3.12.html",
                ),
                WebSearchResult(
                    title="Python 3.13 documentation",
                    url="https://docs.python.org/3.13/",
                ),
                WebSearchResult(
                    title="What’s New In Python 3.13",
                    url="https://docs.python.org/3.13/whatsnew/3.13.html",
                ),
                WebSearchResult(
                    title="Python 3.13 新特性",
                    url="https://docs.python.org/zh-cn/3.13/whatsnew/3.13.html",
                ),
                WebSearchResult(
                    title="What’s New In Python 3.14",
                    url="https://docs.python.org/3.14/whatsnew/3.14.html",
                ),
            ]

    result = WebSearchTool(
        VersionedSearchProvider(), allowed_domains=("docs.python.org",)
    ).execute(
        ToolCall(
            call_id="search",
            tool_name="web_search",
            arguments={"query": "Python 3.13", "limit": 5},
        ),
        context(tmp_path),
    )

    assert result.success
    assert result.output["results"] == [
        {
            "title": "Python 3.13 新特性",
            "url": "https://docs.python.org/zh-cn/3.13/whatsnew/3.13.html",
            "summary": "",
        }
    ]


def test_fetch_tool_rejects_disallowed_domain_before_http(tmp_path: Path) -> None:
    result = FetchUrlTool(
        FailingClient(), allowed_domains=("docs.python.org",)
    ).execute(
        ToolCall(
            call_id="fetch",
            tool_name="fetch_url",
            arguments={"url": "https://example.com/python", "title": "Other"},
        ),
        context(tmp_path),
    )

    assert not result.success
    assert result.error_type == "web_domain_not_allowed"


def test_fetch_tool_reads_whitelisted_official_result(tmp_path: Path) -> None:
    client = OfficialClient()
    result = FetchUrlTool(client, allowed_domains=("docs.python.org",)).execute(
        ToolCall(
            call_id="fetch",
            tool_name="fetch_url",
            arguments={
                "url": "https://docs.python.org/3/",
                "title": "Python docs",
            },
        ),
        context(tmp_path),
    )

    assert result.success
    assert client.urls == ["https://docs.python.org/3/"]
    assert result.output["url"] == "https://docs.python.org/3/"


def test_fetch_tool_rechecks_final_page_domain(tmp_path: Path) -> None:
    result = FetchUrlTool(
        RedirectedClient(), allowed_domains=("docs.python.org",)
    ).execute(
        ToolCall(
            call_id="fetch",
            tool_name="fetch_url",
            arguments={
                "url": "https://docs.python.org/3/",
                "title": "Python docs",
            },
        ),
        context(tmp_path),
    )

    assert not result.success
    assert result.error_type == "web_domain_not_allowed"
