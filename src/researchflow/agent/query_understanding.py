"""Deterministic, JSON-safe understanding of one research question."""

import re
from datetime import date
from typing import Literal

from pydantic import Field

from researchflow.agent.evidence_policy import (
    FreshnessIntent,
    build_retrieval_query,
    classify_freshness_intent,
)
from researchflow.domain.models import DomainModel


class QueryUnderstanding(DomainModel):
    """The explicit boundary between conversational context and retrieval."""

    current_input: str
    resolved_subject: str = ""
    standalone_query: str
    intent: str
    required_facets: list[str] = Field(default_factory=list)
    search_queries: list[str] = Field(min_length=1, max_length=3)
    freshness: FreshnessIntent = Field(default_factory=FreshnessIntent)
    decision_source: Literal["deterministic", "llm", "deterministic_fallback"]
    fallback_reason: str | None = None


_REFERENCE_TOKENS = (
    "它们",
    "他们",
    "她们",
    "两者",
    "该项目",
    "这个游戏",
    "它",
    "他",
    "她",
)


def build_deterministic_understanding(
    current_input: str,
    *,
    prior_subject: str,
    prior_query: str,
    prior_answer_target: str,
    answer_language: str,
    today: date | None = None,
) -> QueryUnderstanding:
    """Resolve a turn without treating historical wording as query text."""
    latest = current_input.strip()
    subject = prior_subject.strip() or explicit_subject(prior_query)
    intent = classify_intent(latest)
    requested_language = requested_answer_language(latest)
    language_only = requested_language is not None and is_language_only_request(latest)
    prior_target = prior_answer_target.strip() or prior_query.strip()
    explicit = (
        ""
        if language_only or intent == "continuation" or has_reference(latest)
        else explicit_subject(latest)
    )
    if explicit:
        subject = explicit

    if language_only and prior_target:
        standalone = prior_target
    elif has_reference(latest) and subject:
        standalone = replace_reference(latest, subject)
    else:
        standalone = latest

    facets: list[str] = []
    if intent == "compiler" and subject:
        standalone = f"{subject}都使用哪些常见编译器"
        facets = ["common_compilers"]
    elif intent == "representative_works" and subject:
        standalone = f"{subject}的代表电影作品有哪些"
        facets = ["representative_works"]
    elif intent == "age" and subject:
        standalone = f"{subject}的出生日期及截至当前日期的年龄"
        facets = ["birth_date", "age"]
    elif intent == "latest_update" and subject:
        standalone = f"{subject}最近什么时候更新"
        facets = ["latest_update"]
    elif intent == "history" and subject:
        facets = [*subject_terms(subject), "history"]
    elif intent == "continuation" and subject:
        standalone = f"{subject}更多信息"

    freshness = classify_freshness_intent(standalone, today=today)
    if intent == "list_current" and "诺贝尔" in standalone:
        facets = ["nobel_prize", "winners", "target_year"]
    query = build_retrieval_query(standalone, freshness)
    return QueryUnderstanding(
        current_input=current_input,
        resolved_subject=subject,
        standalone_query=standalone,
        intent=intent,
        required_facets=facets,
        search_queries=[query],
        freshness=freshness,
        decision_source="deterministic",
        fallback_reason=None,
    )


def explicit_subject(query: str) -> str:
    definition = re.search(
        r"(?:什么是|what\s+is)\s*([A-Za-z][A-Za-z0-9+.#_-]*)",
        query,
        flags=re.IGNORECASE,
    )
    if definition:
        return definition.group(1)
    for suffix in ("有什么区别", "是谁", "是个什么游戏", "是什么游戏"):
        if query.endswith(suffix):
            return query.removesuffix(suffix).strip(" ，。？?")
    if query and not any(
        token in query for token in ("什么", "哪些", "怎么", "吗", "？", "?")
    ):
        return query
    return ""


def classify_intent(query: str) -> str:
    normalized = query.casefold()
    if "编译器" in query or "compiler" in normalized:
        return "compiler"
    if "代表作" in query:
        return "representative_works"
    if "多大" in query or "年龄" in query:
        return "age"
    if "更新" in query:
        return "latest_update"
    if "发展历史" in query or "history" in normalized:
        return "history"
    freshness = classify_freshness_intent(query)
    if freshness.requires_current_evidence and freshness.requires_list_evidence:
        return "list_current"
    if "区别" in query:
        return "comparison"
    if normalized in {"more", "continue", "继续", "更多"}:
        return "continuation"
    return "overview"


def requested_answer_language(query: str) -> str | None:
    normalized = query.casefold().replace(" ", "")
    if any(token in normalized for token in ("中文回答", "汉语回答", "简体中文")):
        return "zh"
    if any(token in normalized for token in ("english", "英文回答", "英语回答")):
        return "en"
    return None


def is_language_only_request(query: str) -> bool:
    normalized = re.sub(r"[，。！？?！\s]", "", query.casefold())
    return normalized in {
        "用中文回答",
        "请用中文回答",
        "中文回答",
        "请用简体中文回答",
        "answerinchinese",
        "pleaseanswerinchinese",
        "answerinenglish",
        "pleaseanswerinenglish",
        "用英文回答",
        "请用英文回答",
    }


def has_reference(query: str) -> bool:
    return any(token in query for token in _REFERENCE_TOKENS)


def replace_reference(query: str, subject: str) -> str:
    for token in _REFERENCE_TOKENS:
        if token in query:
            return query.replace(token, subject)
    return query


def subject_terms(subject: str) -> list[str]:
    """Return stable individual terms when a resolved subject is a pair."""
    terms = [
        part.strip()
        for part in re.split(r"(?:和|与|及|/|、|,|，)", subject)
        if part.strip()
    ]
    return terms or [subject]
