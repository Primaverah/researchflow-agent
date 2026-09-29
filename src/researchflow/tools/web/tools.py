"""Tool adapters for optional search and safe HTML retrieval."""

from datetime import UTC, datetime

from pydantic import BaseModel

from researchflow.tools import BaseTool, ToolContext, ToolFailure
from researchflow.tools.web.http import SafeHttpClient, WebFetchError
from researchflow.tools.web.models import FetchUrlArgs, WebSearchArgs, WebSource
from researchflow.tools.web.provider import WebSearchError, WebSearchProvider


class WebSearchTool(BaseTool):
    name = "web_search"
    description = "Search public web sources through the configured provider."
    args_schema = WebSearchArgs

    def __init__(self, provider: WebSearchProvider) -> None:
        super().__init__()
        self._provider = provider

    def _execute(self, arguments: BaseModel, context: ToolContext) -> object:
        parsed = WebSearchArgs.model_validate(arguments)
        try:
            results = self._provider.search(parsed.query, parsed.limit)
        except WebSearchError as exc:
            raise ToolFailure(
                "web search failed", error_type="web_search_failed"
            ) from exc
        return {
            "query": parsed.query,
            "count": len(results),
            "results": [item.model_dump(mode="json") for item in results],
        }


class FetchUrlTool(BaseTool):
    name = "fetch_url"
    description = "Fetch one safe public HTML page."
    args_schema = FetchUrlArgs

    def __init__(self, client: SafeHttpClient) -> None:
        super().__init__()
        self._client = client

    def _execute(self, arguments: BaseModel, context: ToolContext) -> object:
        parsed = FetchUrlArgs.model_validate(arguments)
        try:
            page = self._client.fetch(parsed.url)
        except WebFetchError as exc:
            raise ToolFailure(
                "web page fetch failed", error_type="web_fetch_failed"
            ) from exc
        return WebSource(
            title=page.title or parsed.title,
            url=page.url,
            summary=parsed.summary,
            accessed_at=datetime.now(UTC),
            content=page.content,
        ).model_dump(mode="json")
