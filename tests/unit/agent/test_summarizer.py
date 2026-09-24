"""Tests for deterministic extractive summarization."""

from researchflow.agent import ExtractiveSummarizer
from researchflow.tools.offline import ReadDocumentOutput


def document(path: str, title: str, content: str) -> ReadDocumentOutput:
    return ReadDocumentOutput(
        path=path,
        title=title,
        content=content,
        char_count=len(content),
    )


def test_extracts_relevant_chinese_text_and_real_sources() -> None:
    documents = [
        document("a.md", "工具安全", "无关说明。\n工具调用需要参数校验。\n其他内容。"),
        document("b.md", "Agent", "Agent 使用本地文档完成研究。"),
    ]

    report = ExtractiveSummarizer().summarize("工具调用", documents)

    assert "工具调用需要参数校验。" in report
    assert "工具安全 — a.md" in report
    assert "Agent — b.md" in report
    assert report.index("a.md") < report.index("b.md")
    assert "http" not in report


def test_deduplicates_sources_and_uses_fallback_text() -> None:
    first = document("a.md", "文档", "# 标题\n第一个有意义的段落。\n第二段。")

    report = ExtractiveSummarizer().summarize("没有匹配", [first, first])

    assert report.count("文档 — a.md") == 1
    assert "第一个有意义的段落。" in report


def test_no_documents_has_no_sources() -> None:
    report = ExtractiveSummarizer().summarize("问题", [])

    assert "未找到相关文档" in report
    assert "## 来源\n\n- 无" in report
