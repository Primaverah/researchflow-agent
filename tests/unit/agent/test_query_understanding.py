from datetime import date

from researchflow.agent.query_understanding import build_deterministic_understanding


def test_c_cpp_history_keeps_subjects_and_history_facet() -> None:
    result = build_deterministic_understanding(
        "它们的发展历史",
        prior_subject="C和C++",
        prior_query="C和C++有什么区别",
        prior_answer_target="C和C++有什么区别",
        answer_language="zh",
    )

    assert result.current_input == "它们的发展历史"
    assert result.resolved_subject == "C和C++"
    assert result.standalone_query == "C和C++的发展历史"
    assert result.intent == "history"
    assert result.required_facets == ["C", "C++", "history"]
    assert result.search_queries == ["C和C++的发展历史"]
    assert result.decision_source == "deterministic"


def test_current_nobel_list_has_year_and_required_facets() -> None:
    result = build_deterministic_understanding(
        "今年的诺贝尔奖获得者有哪些",
        prior_subject="",
        prior_query="",
        prior_answer_target="",
        answer_language="zh",
        today=date(2026, 10, 10),
    )

    assert result.standalone_query == "今年的诺贝尔奖获得者有哪些"
    assert result.intent == "list_current"
    assert result.required_facets == ["nobel_prize", "winners", "target_year"]
    assert result.search_queries == ["今年的诺贝尔奖获得者有哪些 2026"]
    assert result.freshness.target_year == 2026
