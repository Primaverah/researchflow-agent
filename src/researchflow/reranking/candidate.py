"""Bounded lexical/vector RRF ranking for already-safe web candidates."""

import math
import re
from collections import Counter
from collections.abc import Sequence
from typing import Protocol

from researchflow.reranking.models import (
    CandidateRanking,
    RankedSource,
    RankingCandidate,
)


class TextEmbeddingProvider(Protocol):
    """Small adapter so model loading stays outside the ranker."""

    def embed(self, texts: list[str]) -> Sequence[Sequence[float]]: ...


class CandidateRanker:
    """Fuse local BM25 and optional vector rankings without creating evidence."""

    def __init__(
        self, embedding: TextEmbeddingProvider | None = None, *, rrf_constant: int = 60
    ) -> None:
        if rrf_constant < 1:
            raise ValueError("rrf_constant must be positive")
        self._embedding = embedding
        self._rrf_constant = rrf_constant

    def rank(
        self, query: str, candidates: Sequence[RankingCandidate]
    ) -> CandidateRanking:
        unique = self._unique(candidates)
        if not unique:
            return CandidateRanking(source_ids=[], records=[])
        lexical_scores = self._bm25_scores(query, unique)
        lexical_order = self._ordered(lexical_scores, unique)
        fallback_reason: str | None = None
        embedding_scores: dict[str, float] = {}
        try:
            if self._embedding is None:
                raise RuntimeError("embedding unavailable")
            embedding_scores = self._embedding_scores(query, unique)
            embedding_order = self._ordered(embedding_scores, unique)
            rrf_scores = self._rrf(lexical_order, embedding_order)
            ordered = self._ordered(rrf_scores, unique)
            reason = "bm25_embedding_rrf"
        except (RuntimeError, ValueError, ArithmeticError):
            ordered = lexical_order
            rrf_scores = {}
            fallback_reason = "embedding_unavailable"
            reason = "bm25_fallback"
        records = [
            RankedSource(
                source_id=candidate.source_id,
                position=index,
                scores={
                    "bm25": lexical_scores[candidate.source_id],
                    **(
                        {"embedding": embedding_scores[candidate.source_id]}
                        if candidate.source_id in embedding_scores
                        else {}
                    ),
                    **(
                        {"rrf": rrf_scores[candidate.source_id]}
                        if candidate.source_id in rrf_scores
                        else {}
                    ),
                },
                reason=reason,
                fallback_reason=fallback_reason,
            )
            for index, candidate in enumerate(ordered, start=1)
        ]
        return CandidateRanking(
            source_ids=[candidate.source_id for candidate in ordered], records=records
        )

    @staticmethod
    def _unique(candidates: Sequence[RankingCandidate]) -> list[RankingCandidate]:
        unique: dict[str, RankingCandidate] = {}
        for candidate in candidates:
            unique.setdefault(candidate.source_id, candidate)
        return list(unique.values())

    @staticmethod
    def _tokens(value: str) -> list[str]:
        return re.findall(r"[a-z0-9+.#_-]+|[\u4e00-\u9fff]", value.casefold())

    def _bm25_scores(
        self, query: str, candidates: Sequence[RankingCandidate]
    ) -> dict[str, float]:
        terms = self._tokens(query)
        documents = [self._tokens(candidate.text) for candidate in candidates]
        average_length = sum(map(len, documents)) / len(documents) or 1.0
        document_frequency = Counter(
            term for document in documents for term in set(document)
        )
        scores: dict[str, float] = {}
        total = len(documents)
        for candidate, document in zip(candidates, documents, strict=True):
            frequencies = Counter(document)
            score = 0.0
            for term in terms:
                frequency = frequencies[term]
                if not frequency:
                    continue
                inverse_frequency = math.log(
                    1
                    + (total - document_frequency[term] + 0.5)
                    / (document_frequency[term] + 0.5)
                )
                denominator = frequency + 1.2 * (
                    1 - 0.75 + 0.75 * len(document) / average_length
                )
                score += inverse_frequency * frequency * 2.2 / denominator
            scores[candidate.source_id] = score
        return scores

    def _embedding_scores(
        self, query: str, candidates: Sequence[RankingCandidate]
    ) -> dict[str, float]:
        assert self._embedding is not None
        vectors = self._embedding.embed(
            [query, *(candidate.text for candidate in candidates)]
        )
        if len(vectors) != len(candidates) + 1:
            raise ValueError("embedding provider returned an unexpected vector count")
        query_vector = vectors[0]
        scores: dict[str, float] = {}
        for candidate, vector in zip(candidates, vectors[1:], strict=True):
            score = self._cosine(query_vector, vector)
            if not math.isfinite(score):
                raise ValueError("embedding provider returned a non-finite similarity")
            scores[candidate.source_id] = score
        return scores

    @staticmethod
    def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
        if len(left) != len(right):
            raise ValueError("embedding vectors must have matching dimensions")
        denominator = math.sqrt(sum(value * value for value in left)) * math.sqrt(
            sum(value * value for value in right)
        )
        if not denominator:
            raise ValueError("embedding vectors must not be zero")
        # SentenceTransformer may yield numpy scalar elements. Convert at the
        # ranking boundary so trace/event JSON remains portable.
        return float(sum(a * b for a, b in zip(left, right, strict=True)) / denominator)

    def _rrf(
        self,
        lexical_order: Sequence[RankingCandidate],
        embedding_order: Sequence[RankingCandidate],
    ) -> dict[str, float]:
        scores: dict[str, float] = {}
        for ranking in (lexical_order, embedding_order):
            for rank, candidate in enumerate(ranking, start=1):
                scores[candidate.source_id] = scores.get(
                    candidate.source_id, 0.0
                ) + 1 / (self._rrf_constant + rank)
        return scores

    @staticmethod
    def _ordered(
        scores: dict[str, float], candidates: Sequence[RankingCandidate]
    ) -> list[RankingCandidate]:
        return sorted(
            candidates,
            key=lambda candidate: (
                -scores[candidate.source_id],
                candidate.input_position,
            ),
        )
