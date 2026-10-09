"""Pure evidence thresholds shared by both research graph executors."""

from dataclasses import dataclass
from datetime import date
from typing import Any, Literal
from urllib.parse import urlparse


@dataclass(frozen=True, slots=True)
class EvidencePolicy:
    """The evidence threshold appropriate for one research question."""

    name: Literal["standard", "current_complete_list"]
    required_source_count: int
    official_domains: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class EvidenceAssessment:
    """A JSON-safe-at-the-boundary summary of an evidence policy decision."""

    policy: EvidencePolicy
    sufficient: bool
    accepted_source_count: int
    required_source_count: int
    gaps: tuple[str, ...] = ()
    official_complete_source_id: str | None = None


_CURRENT_TERMS = ("今年", "最新", "本年度", "current", "latest", "this year")
_LIST_TERMS = (
    "获奖者",
    "获奖名单",
    "名单",
    "完整列表",
    "完整名单",
    "winners",
    "laureates",
    "list",
)
_COMPLETE_LIST_TERMS = (
    "完整",
    "全部",
    "全体",
    "名单",
    "获奖者",
    "winner",
    "laureate",
    "list",
)


def classify_evidence_policy(
    query: str, *, official_domains: tuple[str, ...] = ()
) -> EvidencePolicy:
    """Classify only the narrow, freshness-sensitive complete-list intent."""
    normalized = query.casefold()
    if any(term in normalized for term in _CURRENT_TERMS) and any(
        term in normalized for term in _LIST_TERMS
    ):
        return EvidencePolicy(
            name="current_complete_list",
            required_source_count=2,
            official_domains=official_domains,
        )
    return EvidencePolicy(
        name="standard", required_source_count=1, official_domains=official_domains
    )


def evaluate_evidence_policy(
    query: str,
    sources: list[Any],
    *,
    official_domains: tuple[str, ...] = (),
) -> EvidenceAssessment:
    """Determine whether accepted source objects meet the query's threshold.

    The caller is responsible for passing only successfully-read, relevant and
    text-quality-approved sources.  This function only applies source-count and
    configured-official complete-list policy.
    """
    policy = classify_evidence_policy(query, official_domains=official_domains)
    unique_sources = _unique_sources(sources)
    source_count = len(unique_sources)
    if policy.name == "standard":
        return EvidenceAssessment(
            policy=policy,
            sufficient=source_count >= policy.required_source_count,
            accepted_source_count=source_count,
            required_source_count=policy.required_source_count,
            gaps=() if source_count else ("no_successful_relevant_source",),
        )

    official_id = next(
        (
            _source_id(source)
            for source in unique_sources
            if _is_official_complete_list(source, policy.official_domains)
        ),
        None,
    )
    sufficient = source_count >= policy.required_source_count or official_id is not None
    return EvidenceAssessment(
        policy=policy,
        sufficient=sufficient,
        accepted_source_count=source_count,
        required_source_count=policy.required_source_count,
        gaps=() if sufficient else ("insufficient_distinct_sources",),
        official_complete_source_id=official_id,
    )


def _unique_sources(sources: list[Any]) -> list[Any]:
    unique: dict[str, Any] = {}
    for source in sources:
        source_id = _source_id(source)
        if source_id:
            unique.setdefault(source_id, source)
    return list(unique.values())


def _source_id(source: Any) -> str:
    return str(getattr(source, "url", getattr(source, "path", ""))).strip()


def _is_official_complete_list(source: Any, official_domains: tuple[str, ...]) -> bool:
    source_url = str(getattr(source, "url", ""))
    parsed = urlparse(source_url)
    if parsed.scheme != "https" or not parsed.hostname:
        return False
    configured = {domain.casefold().strip() for domain in official_domains}
    if parsed.hostname.casefold() not in configured:
        return False

    text = (
        f"{getattr(source, 'title', '')}\n{getattr(source, 'content', '')}"
    ).casefold()
    has_year = str(date.today().year) in text or any(
        marker in text for marker in ("今年", "本年度", "current", "latest")
    )
    return has_year and any(term in text for term in _COMPLETE_LIST_TERMS)
