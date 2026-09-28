"""Tests for the bilingual retrieval evaluation baseline."""

from researchflow.evaluation import (
    EmbeddingRetriever,
    EvaluationCase,
    evaluate_retriever,
    summarize_rankings,
)
from researchflow.evaluation.hybrid import rrf_fuse
from researchflow.tools.offline import Bm25SearchBackend
from researchflow.tools.offline.interfaces import Document


class StaticSource:
    def __init__(self, documents: tuple[Document, ...]) -> None:
        self.documents = {document.path: document for document in documents}

    def list_documents(self) -> list[str]:
        return sorted(self.documents)

    def read_document(self, path: str) -> Document:
        return self.documents[path]


class FakeEmbeddingProvider:
    def embed(self, texts):
        return [
            [1.0, 0.0] if "工具" in text or "tool" in text else [0.0, 1.0]
            for text in texts
        ]


def test_summarize_rankings_reports_language_and_overall_metrics() -> None:
    cases = (
        EvaluationCase("zh-1", "中文查询", "zh", ("zh-a.md",)),
        EvaluationCase("en-1", "english query", "en", ("en-a.md",)),
    )
    rankings = {
        "zh-1": ("zh-a.md", "zh-b.md"),
        "en-1": ("en-b.md", "en-a.md"),
    }

    result = summarize_rankings(cases, rankings)

    assert result["zh"] == {
        "queries": 1,
        "recall_at_1": 1.0,
        "recall_at_3": 1.0,
        "recall_at_5": 1.0,
        "mrr": 1.0,
    }
    assert result["en"]["recall_at_1"] == 0.0
    assert result["en"]["mrr"] == 0.5
    assert result["overall"]["recall_at_1"] == 0.5
    assert result["overall"]["mrr"] == 0.75


def test_summarize_rankings_counts_missing_results_as_zero() -> None:
    case = EvaluationCase("zh-1", "中文查询", "zh", ("zh-a.md",))

    result = summarize_rankings((case,), {"zh-1": ()})

    assert result["zh"] == {
        "queries": 1,
        "recall_at_1": 0.0,
        "recall_at_3": 0.0,
        "recall_at_5": 0.0,
        "mrr": 0.0,
    }


def test_bm25_ranks_documents_by_query_term_frequency() -> None:
    source = StaticSource(
        (
            Document("low.md", "Low", "tool safety"),
            Document("high.md", "High", "tool calling tool calling safety"),
        )
    )

    hits = Bm25SearchBackend(source).search("tool calling", limit=2)

    assert [hit.path for hit in hits] == ["high.md", "low.md"]


def test_evaluator_reports_all_same_language_metrics_with_fake_provider() -> None:
    result = evaluate_retriever("all", provider=FakeEmbeddingProvider())

    assert set(result) == {"keyword", "bm25", "embedding", "hybrid"}
    for summary in result.values():
        assert summary["zh"]["queries"] == 2
        assert summary["en"]["queries"] == 2
        assert summary["overall"]["queries"] == 4


def test_embedding_retriever_handles_zero_vectors_and_rrf_is_stable() -> None:
    documents = (
        Document("b.md", "B", "other"),
        Document("a.md", "A", "tool"),
    )
    hits = EmbeddingRetriever(documents, FakeEmbeddingProvider()).search("tool", 2)

    assert [hit.path for hit in hits] == ["a.md", "b.md"]
    assert [hit.path for hit in rrf_fuse(hits, list(reversed(hits)))] == [
        "a.md",
        "b.md",
    ]
