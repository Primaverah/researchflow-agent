"""Pure evidence thresholds shared by both research graph executors."""

import re
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


@dataclass(frozen=True, slots=True)
class FreshnessIntent:
    """Time/list constraints used only for retrieval and evidence admission."""

    target_year: int | None = None
    requires_current_evidence: bool = False
    requires_list_evidence: bool = False


_CURRENT_TERMS = (
    "今年",
    "本年度",
    "当前",
    "最新",
    "current",
    "currently",
    "latest",
    "this year",
)
_LIST_TERMS = (
    "获奖者",
    "获奖名单",
    "名单",
    "完整列表",
    "完整名单",
    "哪些",
    "winners",
    "laureates",
    "list",
)


def classify_freshness_intent(
    query: str, *, today: date | None = None
) -> FreshnessIntent:
    """Resolve explicit or relative years without changing the answer target."""
    normalized = query.casefold()
    explicit = re.search(r"(?:19|20)\d{2}", normalized)
    target_year = int(explicit.group(0)) if explicit else None
    is_current = target_year is not None or any(
        term in normalized for term in _CURRENT_TERMS
    )
    if target_year is None and is_current:
        target_year = (today or date.today()).year
    return FreshnessIntent(
        target_year=target_year,
        requires_current_evidence=is_current,
        requires_list_evidence=any(term in normalized for term in _LIST_TERMS),
    )


def build_retrieval_query(query: str, intent: FreshnessIntent) -> str:
    """Add the resolved year once for search while preserving the original query."""
    if intent.target_year is None or str(intent.target_year) in query:
        return query
    return f"{query} {intent.target_year}"
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
    intent = classify_freshness_intent(query)
    if intent.requires_current_evidence and intent.requires_list_evidence:
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
