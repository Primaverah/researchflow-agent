"""Tool adapters for optional search and safe HTML retrieval."""

import re
from datetime import UTC, datetime
from urllib.parse import urlparse

from pydantic import BaseModel

from researchflow.tools import BaseTool, ToolContext, ToolFailure
from researchflow.tools.web.domains import is_allowed_domain
from researchflow.tools.web.http import SafeHttpClient, WebFetchError
from researchflow.tools.web.models import (
    FetchUrlArgs,
    WebSearchArgs,
    WebSearchResult,
    WebSource,
)
from researchflow.tools.web.provider import WebSearchError, WebSearchProvider

_QUERY_STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "documentation",
    "docs",
    "do",
    "for",
    "how",
    "i",
    "is",
    "of",
    "official",
    "the",
    "to",
    "use",
    "what",
}


class WebSearchTool(BaseTool):
    name = "web_search"
    description = "Search public web sources through the configured provider."
    args_schema = WebSearchArgs

    def __init__(
        self, provider: WebSearchProvider, allowed_domains: tuple[str, ...] = ()
    ) -> None:
        super().__init__()
        self._provider = provider
        self._allowed_domains = allowed_domains

    def _execute(self, arguments: BaseModel, context: ToolContext) -> object:
        parsed = WebSearchArgs.model_validate(arguments)
        try:
            results = self._provider.search(
                parsed.query, parsed.limit, self._allowed_domains
            )
        except WebSearchError as exc:
            raise ToolFailure(
                "web search failed", error_type="web_search_failed"
            ) from exc
        results = [
            item
            for item in results
            if is_allowed_domain(item.url, self._allowed_domains)
        ]
        results = self._filter_relevant_results(results, parsed.query)
        simpler_query = self._simpler_query(parsed.query)
        if self._allowed_domains and not results and simpler_query is not None:
            try:
                results = self._provider.search(
                    simpler_query, parsed.limit, self._allowed_domains
                )
            except WebSearchError as exc:
                raise ToolFailure(
                    "web search failed", error_type="web_search_failed"
                ) from exc
            results = [
                item
                for item in results
                if is_allowed_domain(item.url, self._allowed_domains)
            ]
            results = self._filter_relevant_results(results, parsed.query)
        return {
            "query": parsed.query,
            "count": len(results),
            "results": [item.model_dump(mode="json") for item in results],
        }

    @staticmethod
    def _simpler_query(query: str) -> str | None:
        keywords = [
            token
            for token in re.findall(
                r"[A-Za-z0-9][A-Za-z0-9._-]*|[\u4e00-\u9fff]+", query
            )
            if token.casefold() not in _QUERY_STOP_WORDS
        ]
        simpler = " ".join(keywords[:4])
        return simpler if simpler and simpler != query else None

    @staticmethod
    def _filter_relevant_results(
        results: list[WebSearchResult], query: str
    ) -> list[WebSearchResult]:
        match = re.search(r"\bpython\s*(3\.\d+)\b", query, re.IGNORECASE)
        version = match.group(1) if match else None
        if version is not None:
            results = [
                item
                for item in results
                if WebSearchTool._matches_python_version(item, version)
            ]

        selected: dict[str, WebSearchResult] = {}
        for item in results:
            key = WebSearchTool._content_key(item.url)
            current = selected.get(key)
            if current is None or WebSearchTool._preference(
                item
            ) < WebSearchTool._preference(current):
                selected[key] = item
        return list(selected.values())

    @staticmethod
    def _matches_python_version(item: WebSearchResult, version: str) -> bool:
        parsed = urlparse(item.url)
        path = parsed.path.rstrip("/")
        path_versions = set(re.findall(r"3\.\d+", path))
        if path_versions and path_versions != {version}:
            return False
        if path in {"", f"/{version}", f"/zh-cn/{version}"}:
            return False
        return version in f"{item.title} {item.url}"

    @staticmethod
    def _content_key(url: str) -> str:
        parsed = urlparse(url)
        parts = [part for part in parsed.path.rstrip("/").split("/") if part]
        if parts and parts[0].casefold() in {"zh-cn", "zh-tw"}:
            parts.pop(0)
        return f"{parsed.hostname or ''}/{'/'.join(parts)}"

    @staticmethod
    def _preference(item: WebSearchResult) -> int:
        parts = [part.casefold() for part in urlparse(item.url).path.split("/")]
        return 0 if "zh-cn" in parts else 1


class FetchUrlTool(BaseTool):
    name = "fetch_url"
    description = "Fetch one safe public HTML page."
    args_schema = FetchUrlArgs

    def __init__(
        self, client: SafeHttpClient, allowed_domains: tuple[str, ...] = ()
    ) -> None:
        super().__init__()
        self._client = client
        self._allowed_domains = allowed_domains

    def _execute(self, arguments: BaseModel, context: ToolContext) -> object:
        parsed = FetchUrlArgs.model_validate(arguments)
        if not is_allowed_domain(parsed.url, self._allowed_domains):
            raise ToolFailure(
                "web source domain is not allowed", error_type="web_domain_not_allowed"
            )
        try:
            page = self._client.fetch(parsed.url)
        except WebFetchError as exc:
            raise ToolFailure(str(exc), error_type=exc.error_type) from exc
        if not is_allowed_domain(page.url, self._allowed_domains):
            raise ToolFailure(
                "web source domain is not allowed", error_type="web_domain_not_allowed"
            )
        return WebSource(
            title=page.title or parsed.title,
            url=page.url,
            summary=parsed.summary,
            accessed_at=datetime.now(UTC),
            content=page.content,
        ).model_dump(mode="json")
