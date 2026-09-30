"""Deterministic extractive summarization for local documents."""

import re
import unicodedata

from researchflow.tools.offline import ReadDocumentOutput
from researchflow.tools.web import WebSource
from researchflow.tools.web.domains import is_allowed_domain

MAX_EXTRACTS_PER_DOCUMENT = 2
MAX_EXTRACT_LENGTH = 240


def _normalize(text: str) -> str:
    return unicodedata.normalize("NFKC", text).casefold()


class ExtractiveSummarizer:
    """Build a Markdown report using only text from read documents."""

    def __init__(self, allowed_domains: tuple[str, ...] = ()) -> None:
        self._allowed_domains = allowed_domains

    def summarize(
        self,
        query: str,
        documents: list[ReadDocumentOutput],
        web_sources: list[WebSource] | None = None,
    ) -> str:
        """Return a stable report with source-supported thematic conclusions."""
        evidence: list[tuple[str, int]] = []
        sources: list[tuple[str, str]] = []
        seen_sources: set[str] = set()
        terms = tuple(dict.fromkeys(_normalize(query).split()))

        for document in documents:
            if document.path in seen_sources:
                continue
            seen_sources.add(document.path)
            lines = self._meaningful_lines(document.content)
            matched = [
                line
                for line in lines
                if terms and any(term in _normalize(line) for term in terms)
            ]
            selected = (matched or lines)[:MAX_EXTRACTS_PER_DOCUMENT]
            if selected:
                sources.append((document.title, document.path))
                source_index = len(sources)
                evidence.extend(
                    (line[:MAX_EXTRACT_LENGTH], source_index) for line in selected
                )

        for source in web_sources or []:
            if not is_allowed_domain(source.url, self._allowed_domains):
                continue
            lines = self._meaningful_lines(source.content)
            matched = [
                line
                for line in lines
                if terms and any(term in _normalize(line) for term in terms)
            ]
            selected = (matched or lines)[:MAX_EXTRACTS_PER_DOCUMENT]
            if selected:
                sources.append((source.title, source.url))
                source_index = len(sources)
                evidence.extend(
                    (line[:MAX_EXTRACT_LENGTH], source_index) for line in selected
                )

        summary_lines = self._thematic_summary(query, evidence)
        source_lines = [
            f"[{index}] {title} — {path}"
            for index, (title, path) in enumerate(sources, start=1)
        ]
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
            parts = re.split(r"(?<=[。！？!?])\s*|(?<!\d)\.(?:\s+|$)", line)
            lines.extend(part.strip() for part in parts if part.strip())
        return lines

    @staticmethod
    def _thematic_summary(query: str, evidence: list[tuple[str, int]]) -> list[str]:
        if not evidence:
            return ["- 未找到相关文档。"]

        facts: list[str] = []
        source_indexes: list[int] = []
        seen_facts: set[str] = set()
        for text, source_index in evidence:
            fact = text.strip()
            if not fact or _normalize(fact) in seen_facts:
                continue
            seen_facts.add(_normalize(fact))
            facts.append(fact)
            if source_index not in source_indexes:
                source_indexes.append(source_index)

        citations = ", ".join(str(index) for index in source_indexes)
        return [f"- {query}：{'；'.join(facts)} [{citations}]"]
