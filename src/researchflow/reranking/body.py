"""Bounded reranking of successfully read web bodies."""

import json
import math
from collections.abc import Sequence
from typing import Protocol

from pydantic import BaseModel, Field

from researchflow.llm import BaseLLMProvider, LLMRequest
from researchflow.reranking.candidate import CandidateRanker, TextEmbeddingProvider
from researchflow.reranking.models import BodyRanking, RankedSource, RankingCandidate
from researchflow.tools.web.models import WebSource


class CrossEncoder(Protocol):
    """Small pair-scoring contract; implementations load models lazily."""

    def score(self, pairs: list[tuple[str, str]]) -> Sequence[float]: ...


class LLMRelevanceDecision(BaseModel):
    """A source-bound LLM relevance judgment for one already-read body."""

    source_id: str = Field(min_length=1)
    score: float = Field(ge=0.0, le=1.0)
    reason: str = Field(min_length=1, max_length=240)


class LLMRelevanceOutput(BaseModel):
    decisions: list[LLMRelevanceDecision] = Field(min_length=1, max_length=5)


class LLMBodyReranker:
    """Use a structured provider only to order a supplied set of read sources."""

    def __init__(
        self,
        provider: BaseLLMProvider,
        *,
        model_name: str,
        max_output_tokens: int = 400,
        max_body_chars: int = 2000,
    ) -> None:
        self._provider = provider
        self._model_name = model_name
        self._max_output_tokens = max_output_tokens
        self._max_body_chars = max_body_chars

    def score(self, query: str, sources: Sequence[WebSource]) -> dict[str, float]:
        source_ids = {source.url for source in sources}
        payload = [
            {
                "source_id": source.url,
                "title": source.title,
                "content": source.content[: self._max_body_chars],
            }
            for source in sources
        ]
        output, _ = self._provider.complete_structured(
            LLMRequest(
                user_prompt=(
                    "Rank only the supplied already-read sources for relevance to the "
                    "user question. Return a 0..1 score and a brief reason for every "
                    "source. Do not add sources or facts. Question: "
                    + query
                    + "\nSources: "
                    + json.dumps(payload, ensure_ascii=False)
                ),
                max_output_tokens=self._max_output_tokens,
            ),
            LLMRelevanceOutput,
        )
        scores: dict[str, float] = {}
        for decision in output.decisions:
            if decision.source_id not in source_ids:
                raise ValueError("llm_invalid_source_id")
            if decision.source_id in scores:
                raise ValueError("llm_duplicate_source_id")
            scores[decision.source_id] = decision.score
        if set(scores) != source_ids:
            raise ValueError("llm_incomplete_source_set")
        return scores


class BodyReranker:
    """Rank read source bodies; never receives candidates that failed fetch."""

    def __init__(
        self,
        embedding: TextEmbeddingProvider | None = None,
        cross_encoder: CrossEncoder | None = None,
        llm_reranker: LLMBodyReranker | None = None,
        *,
        cross_encoder_top_n: int = 5,
        body_relevance_threshold: float = 0.0,
        llm_top_n: int = 3,
        llm_margin: float = 0.08,
    ) -> None:
        if cross_encoder_top_n < 1:
            raise ValueError("cross_encoder_top_n must be positive")
        if llm_top_n < 1:
            raise ValueError("llm_top_n must be positive")
        if llm_margin < 0:
            raise ValueError("llm_margin cannot be negative")
        self._embedding = embedding
        self._cross_encoder = cross_encoder
        self._cross_encoder_top_n = cross_encoder_top_n
        self._threshold = body_relevance_threshold
        self._llm_reranker = llm_reranker
        self._llm_top_n = llm_top_n
        self._llm_margin = llm_margin

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
        llm_scores: dict[str, float] = {}
        if self._should_use_llm(ordered, cross_scores, embedding_scores):
            top_sources = [
                source
                for source in sources
                if source.url in {item.source_id for item in ordered[: self._llm_top_n]}
            ]
            by_source = {source.url: source for source in top_sources}
            ordered_sources = [
                by_source[item.source_id] for item in ordered[: self._llm_top_n]
            ]
            try:
                assert self._llm_reranker is not None
                llm_scores = self._llm_reranker.score(query, ordered_sources)
                top = sorted(
                    ordered[: self._llm_top_n],
                    key=lambda candidate: (
                        -llm_scores[candidate.source_id],
                        candidate.input_position,
                    ),
                )
                ordered = [*top, *ordered[self._llm_top_n :]]
            except (RuntimeError, ValueError, ArithmeticError):
                fallback_reason = "llm_invalid_source_id"
        accepted: list[str] = []
        rejected: dict[str, str] = {}
        records: list[RankedSource] = []
        for position, candidate in enumerate(ordered, start=1):
            score = cross_scores.get(
                candidate.source_id,
                llm_scores.get(
                    candidate.source_id, embedding_scores.get(candidate.source_id)
                ),
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
            if candidate.source_id in llm_scores:
                scores["llm"] = llm_scores[candidate.source_id]
            records.append(
                RankedSource(
                    source_id=candidate.source_id,
                    position=position,
                    scores=scores,
                    reason=(
                        "embedding_cross_encoder_llm"
                        if llm_scores
                        else "embedding_cross_encoder"
                        if cross_scores
                        else "embedding"
                    ),
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

    def _should_use_llm(
        self,
        ordered: Sequence[RankingCandidate],
        cross_scores: dict[str, float],
        embedding_scores: dict[str, float],
    ) -> bool:
        if self._llm_reranker is None or len(ordered) < 2:
            return False
        scores = cross_scores or embedding_scores
        if not scores:
            return True
        return (
            abs(
                scores.get(ordered[0].source_id, 0.0)
                - scores.get(ordered[1].source_id, 0.0)
            )
            <= self._llm_margin
        )

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
