"""Deterministic extractive summarization for local documents."""

import re
import unicodedata

from researchflow.tools.offline import ReadDocumentOutput
from researchflow.tools.web import WebSource

MAX_EXTRACTS_PER_DOCUMENT = 2
MAX_EXTRACT_LENGTH = 240


def _normalize(text: str) -> str:
    return unicodedata.normalize("NFKC", text).casefold()


class ExtractiveSummarizer:
    """Build a Markdown report using only text from read documents."""

    def summarize(
        self,
        query: str,
        documents: list[ReadDocumentOutput],
        web_sources: list[WebSource] | None = None,
    ) -> str:
        """Return a stable report with extracts and verified local sources."""
        extracts: list[str] = []
        sources: list[tuple[str, str]] = []
        seen_sources: set[str] = set()
        terms = tuple(dict.fromkeys(_normalize(query).split()))

        for document in documents:
            if document.path in seen_sources:
                continue
            seen_sources.add(document.path)
            sources.append((document.title, document.path))
            lines = self._meaningful_lines(document.content)
            matched = [
                line
                for line in lines
                if terms and any(term in _normalize(line) for term in terms)
            ]
            selected = (matched or lines)[:MAX_EXTRACTS_PER_DOCUMENT]
            extracts.extend(line[:MAX_EXTRACT_LENGTH] for line in selected)

        summary_lines = (
            [f"- {extract}" for extract in extracts]
            if extracts
            else ["- 未找到相关文档。"]
        )
        web_sources = web_sources or []
        source_lines = [
            f"[{index}] {title} — {path}"
            for index, (title, path) in enumerate(sources, start=1)
        ]
        source_lines.extend(
            f"[{index}] {source.title} — {source.url}"
            for index, source in enumerate(web_sources, start=len(source_lines) + 1)
        )
        if not source_lines:
            source_lines = ["- 无"]
        return "\n".join(
            [
                "# 研究报告",
                "",
                "## 问题",
                "",
                query,
                "",
                "## 汇总",
                "",
                *summary_lines,
                "",
                "## 来源",
                "",
                *source_lines,
            ]
        )

    @staticmethod
    def _meaningful_lines(content: str) -> list[str]:
        lines: list[str] = []
        for raw_line in content.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = re.split(r"(?<=[。！？.!?])\s*", line)
            lines.extend(part.strip() for part in parts if part.strip())
        return lines
