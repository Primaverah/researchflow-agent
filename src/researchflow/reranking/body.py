"""Bounded reranking of successfully read web bodies."""

import math
from collections.abc import Sequence
from typing import Protocol

from researchflow.reranking.candidate import CandidateRanker, TextEmbeddingProvider
from researchflow.reranking.models import BodyRanking, RankedSource, RankingCandidate
from researchflow.tools.web.models import WebSource


class CrossEncoder(Protocol):
    """Small pair-scoring contract; implementations load models lazily."""

    def score(self, pairs: list[tuple[str, str]]) -> Sequence[float]: ...


class BodyReranker:
    """Rank read source bodies; never receives candidates that failed fetch."""

    def __init__(
        self,
        embedding: TextEmbeddingProvider | None = None,
        cross_encoder: CrossEncoder | None = None,
        *,
        cross_encoder_top_n: int = 5,
        body_relevance_threshold: float = 0.0,
    ) -> None:
        if cross_encoder_top_n < 1:
            raise ValueError("cross_encoder_top_n must be positive")
        self._embedding = embedding
        self._cross_encoder = cross_encoder
        self._cross_encoder_top_n = cross_encoder_top_n
        self._threshold = body_relevance_threshold

    def rank(self, query: str, sources: Sequence[WebSource]) -> BodyRanking:
        if not sources:
            return BodyRanking([], [], {}, [])
        candidates = [
            RankingCandidate(source.url, source.title, source.content, index)
            for index, source in enumerate(sources)
        ]
        embedding_scores = self._embedding_scores(query, candidates)
        ordered = self._order(candidates, embedding_scores)
        cross_scores: dict[str, float] = {}
        fallback_reason: str | None = None
        if self._cross_encoder is not None:
            try:
                top = ordered[: self._cross_encoder_top_n]
                values = self._cross_encoder.score(
                    [(query, candidate.text) for candidate in top]
                )
                if len(values) != len(top):
                    raise ValueError("cross encoder returned an unexpected score count")
                cross_scores = {
                    candidate.source_id: self._finite(value)
                    for candidate, value in zip(top, values, strict=True)
                }
                ordered = [
                    *sorted(
                        top,
                        key=lambda candidate: (
                            -cross_scores[candidate.source_id],
                            candidate.input_position,
                        ),
                    ),
                    *ordered[self._cross_encoder_top_n :],
                ]
            except (RuntimeError, ValueError, ArithmeticError):
                fallback_reason = "cross_encoder_unavailable"
        accepted: list[str] = []
        rejected: dict[str, str] = {}
        records: list[RankedSource] = []
        for position, candidate in enumerate(ordered, start=1):
            score = cross_scores.get(
                candidate.source_id, embedding_scores.get(candidate.source_id)
            )
            if score is None or score >= self._threshold:
                accepted.append(candidate.source_id)
            else:
                rejected[candidate.source_id] = "body_relevance_below_threshold"
            scores = {}
            if candidate.source_id in embedding_scores:
                scores["embedding"] = embedding_scores[candidate.source_id]
            if candidate.source_id in cross_scores:
                scores["cross_encoder"] = cross_scores[candidate.source_id]
            records.append(
                RankedSource(
                    source_id=candidate.source_id,
                    position=position,
                    scores=scores,
                    reason=("embedding_cross_encoder" if cross_scores else "embedding"),
                    fallback_reason=fallback_reason,
                )
            )
        return BodyRanking(
            source_ids=[candidate.source_id for candidate in ordered],
            accepted_source_ids=accepted,
            rejected_reasons=rejected,
            records=records,
        )

    def _embedding_scores(
        self, query: str, candidates: Sequence[RankingCandidate]
    ) -> dict[str, float]:
        if self._embedding is None:
            return {}
        try:
            vectors = self._embedding.embed(
                [query, *(item.text for item in candidates)]
            )
            if len(vectors) != len(candidates) + 1:
                raise ValueError(
                    "embedding provider returned an unexpected vector count"
                )
            return {
                candidate.source_id: self._cosine(vectors[0], vector)
                for candidate, vector in zip(candidates, vectors[1:], strict=True)
            }
        except (RuntimeError, ValueError, ArithmeticError):
            return {}

    @staticmethod
    def _order(
        candidates: Sequence[RankingCandidate], scores: dict[str, float]
    ) -> list[RankingCandidate]:
        return sorted(
            candidates,
            key=lambda candidate: (
                -scores.get(candidate.source_id, 0.0),
                candidate.input_position,
            ),
        )

    @staticmethod
    def _finite(value: float) -> float:
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError("reranker score must be finite")
        return float(value)

    @classmethod
    def _cosine(cls, left: Sequence[float], right: Sequence[float]) -> float:
        return cls._finite(CandidateRanker._cosine(left, right))
