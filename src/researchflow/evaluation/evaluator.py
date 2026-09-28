"""Run retriever baselines over the fixed bilingual evaluation set."""

from researchflow.evaluation.dataset import EVALUATION_CASES, EVALUATION_DOCUMENTS, EvaluationDocument
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


def evaluate_retriever(retriever: str) -> dict[str, dict[str, dict[str, float | int]]]:
    """Evaluate one retriever or both retrievers by language."""
    names = ("keyword", "bm25") if retriever == "all" else (retriever,)
    if not set(names) <= {"keyword", "bm25"}:
        raise ValueError("retriever must be keyword, bm25, or all")
    result = {}
    for name in names:
        rankings = {}
        for case in EVALUATION_CASES:
            source = _LanguageDocumentSource(case.language)
            backend = KeywordSearchBackend(source) if name == "keyword" else Bm25SearchBackend(source)
            rankings[case.case_id] = tuple(hit.path for hit in backend.search(case.query, 5))
        result[name] = summarize_rankings(EVALUATION_CASES, rankings)
    return result
