"""Run retriever baselines over the fixed bilingual evaluation set."""

from researchflow.evaluation.dataset import (
    EVALUATION_CASES,
    EVALUATION_DOCUMENTS,
)
from researchflow.evaluation.embeddings import (
    EmbeddingProvider,
    EmbeddingRetriever,
    SentenceTransformerProvider,
)
from researchflow.evaluation.hybrid import rrf_fuse
from researchflow.evaluation.metrics import summarize_rankings
from researchflow.tools.offline import Bm25SearchBackend, KeywordSearchBackend
from researchflow.tools.offline.interfaces import Document


class _LanguageDocumentSource:
    def __init__(self, language: str) -> None:
        self._documents = {
            document.path: Document(document.path, document.title, document.content)
            for document in EVALUATION_DOCUMENTS
            if document.language == language
        }

    def list_documents(self) -> list[str]:
        return sorted(self._documents)

    def read_document(self, path: str) -> Document:
        return self._documents[path]


def evaluate_retriever(
    retriever: str,
    provider: EmbeddingProvider | None = None,
    model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
) -> dict[str, dict[str, dict[str, float | int]]]:
    """Evaluate one retriever or both retrievers by language."""
    names = (
        ("keyword", "bm25", "embedding", "hybrid")
        if retriever == "all"
        else (retriever,)
    )
    if not set(names) <= {"keyword", "bm25", "embedding", "hybrid"}:
        raise ValueError("retriever must be keyword, bm25, embedding, hybrid, or all")
    embedding_provider = (
        provider or SentenceTransformerProvider(model_name)
        if {"embedding", "hybrid"} & set(names)
        else None
    )
    result = {}
    for name in names:
        rankings = {}
        for case in EVALUATION_CASES:
            source = _LanguageDocumentSource(case.language)
            documents = [source.read_document(path) for path in source.list_documents()]
            if name == "keyword":
                hits = KeywordSearchBackend(source).search(case.query, 5)
            elif name == "bm25":
                hits = Bm25SearchBackend(source).search(case.query, 5)
            else:
                embedding = EmbeddingRetriever(
                    documents, embedding_provider
                ).search(case.query, 5)
                hits = (
                    embedding
                    if name == "embedding"
                    else rrf_fuse(
                        Bm25SearchBackend(source).search(case.query, 5), embedding
                    )
                )
            rankings[case.case_id] = tuple(hit.path for hit in hits)
        result[name] = summarize_rankings(EVALUATION_CASES, rankings)
    return result
