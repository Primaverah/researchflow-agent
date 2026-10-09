from datetime import UTC, date, datetime

from researchflow.agent.evidence_policy import (
    build_retrieval_query,
    classify_freshness_intent,
    evaluate_evidence_policy,
)
from researchflow.tools.web import WebSource


def _source(url: str, content: str = "2026 年完整获奖者名单") -> WebSource:
    return WebSource(
        title="2026 年获奖者完整名单",
        url=url,
        content=content,
        accessed_at=datetime.now(UTC),
    )


def test_current_complete_list_requires_two_distinct_sources() -> None:
    assessment = evaluate_evidence_policy(
        "今年诺贝尔奖获奖者名单", [_source("https://news.example/one")]
    )

    assert assessment.policy.name == "current_complete_list"
    assert assessment.sufficient is False
    assert assessment.required_source_count == 2
    assert assessment.gaps == ("insufficient_distinct_sources",)


def test_configured_official_complete_list_allows_one_source() -> None:
    assessment = evaluate_evidence_policy(
        "今年诺贝尔奖获奖者名单",
        [_source("https://official.example/winners")],
        official_domains=("official.example",),
    )

    assert assessment.sufficient is True
    assert assessment.required_source_count == 2
    assert assessment.official_complete_source_id == "https://official.example/winners"


def test_same_url_never_counts_twice() -> None:
    source = _source("https://news.example/winners")

    assessment = evaluate_evidence_policy(
        "今年获奖名单", [source, source.model_copy(update={"title": "duplicate"})]
    )

    assert assessment.sufficient is False
    assert assessment.accepted_source_count == 1


def test_non_official_or_incomplete_page_does_not_get_single_source_exception() -> None:
    assessment = evaluate_evidence_policy(
        "今年诺贝尔奖获奖者名单",
        [_source("http://official.example/winners", content="2026 年获奖新闻")],
        official_domains=("official.example",),
    )

    assert assessment.sufficient is False
    assert assessment.official_complete_source_id is None


def test_current_list_intent_adds_execution_year_to_retrieval_query() -> None:
    intent = classify_freshness_intent(
        "今年的诺贝尔奖目前出炉了哪些", today=date(2026, 10, 9)
    )

    assert intent.target_year == 2026
    assert intent.requires_current_evidence is True
    assert intent.requires_list_evidence is True
    assert build_retrieval_query("今年的诺贝尔奖目前出炉了哪些", intent).endswith(
        "2026"
    )


def test_explicit_year_wins_over_execution_year() -> None:
    intent = classify_freshness_intent("2025年诺贝尔奖名单", today=date(2026, 10, 9))

    assert intent.target_year == 2025
