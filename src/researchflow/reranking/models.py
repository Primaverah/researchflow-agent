"""JSON-safe value objects shared by bounded reranking stages."""

from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RankingCandidate:
    """Search-result text that may rank a read attempt but is never evidence."""

    source_id: str
    title: str
    summary: str
    input_position: int

    @property
    def text(self) -> str:
        return f"{self.title}\n{self.summary}".strip()


@dataclass(frozen=True, slots=True)
class RankedSource:
    """One source's explainable ordering data without page content."""

    source_id: str
    position: int
    scores: Mapping[str, float]
    reason: str
    fallback_reason: str | None = None


@dataclass(frozen=True, slots=True)
class CandidateRanking:
    """Stable candidate source order and metadata for trace projection."""

    source_ids: list[str]
    records: list[RankedSource]


@dataclass(frozen=True, slots=True)
class BodyRanking:
    """Ordered read sources and any explicit relevance-threshold rejections."""

    source_ids: list[str]
    accepted_source_ids: list[str]
    rejected_reasons: Mapping[str, str]
    records: list[RankedSource]
