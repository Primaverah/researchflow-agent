"""Tests for deterministic domain checks at web tool boundaries."""

from pathlib import Path

from researchflow.domain import ToolCall
from researchflow.tools import ToolContext
from researchflow.tools.web.models import WebSearchResult
from researchflow.tools.web.tools import FetchUrlTool, WebSearchTool


class FakeSearchProvider:
    def search(self, query: str, limit: int) -> list[WebSearchResult]:
        return [
            WebSearchResult(title="Official", url="https://docs.python.org/3/"),
            WebSearchResult(title="Other", url="https://example.com/python"),
        ]


class FailingClient:
    def fetch(self, url: str):
        raise AssertionError("blocked domains must not reach the HTTP client")


def context(tmp_path: Path) -> ToolContext:
    return ToolContext(
        working_directory=tmp_path,
        output_directory=tmp_path,
        run_id="run-1",
    )


def test_web_search_tool_filters_results_to_allowed_domains(tmp_path: Path) -> None:
    result = WebSearchTool(
        FakeSearchProvider(), allowed_domains=("docs.python.org",)
    ).execute(
        ToolCall(
            call_id="search", tool_name="web_search", arguments={"query": "Python"}
        ),
        context(tmp_path),
    )

    assert result.success
    assert result.output["count"] == 1
    assert result.output["results"][0]["url"] == "https://docs.python.org/3/"


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
