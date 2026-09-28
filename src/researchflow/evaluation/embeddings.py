"""Lazy multilingual embedding providers and cosine retrieval."""

import math
from collections.abc import Sequence
from typing import Protocol

from researchflow.tools.offline.interfaces import Document, SearchHit


class EmbeddingProvider(Protocol):
    def embed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]: ...


class EmbeddingUnavailableError(RuntimeError):
    """Raised when the optional local embedding runtime is unavailable."""


class SentenceTransformerProvider:
    def __init__(
        self, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"
    ) -> None:
        self._model_name = model_name
        self._model = None

    def embed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise EmbeddingUnavailableError(
                    "embedding retrieval requires the optional "
                    "sentence-transformers dependency"
                ) from exc
            try:
                self._model = SentenceTransformer(self._model_name)
            except Exception as exc:
                raise EmbeddingUnavailableError(
                    f"embedding model '{self._model_name}' could not be loaded"
                ) from exc
        return self._model.encode(list(texts), normalize_embeddings=False)


class EmbeddingRetriever:
    def __init__(
        self, documents: Sequence[Document], provider: EmbeddingProvider
    ) -> None:
        self._documents = tuple(documents)
        self._provider = provider

    def search(self, query: str, limit: int) -> list[SearchHit]:
        vectors = self._provider.embed(
            [query, *(doc.content for doc in self._documents)]
        )
        query_vector = vectors[0]
        hits = [
            SearchHit(
                doc.path, doc.title, _cosine(query_vector, vector), doc.content[:160]
            )
            for doc, vector in zip(self._documents, vectors[1:], strict=True)
        ]
        hits.sort(key=lambda hit: (-hit.score, hit.path))
        return hits[:limit]


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    denominator = math.sqrt(sum(value * value for value in left)) * math.sqrt(
        sum(value * value for value in right)
    )
    return (
        sum(a * b for a, b in zip(left, right, strict=True)) / denominator
        if denominator
        else 0.0
    )
