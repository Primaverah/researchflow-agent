"""Tests for the bilingual retrieval evaluation baseline."""

from researchflow.evaluation import EvaluationCase, summarize_rankings


def test_summarize_rankings_reports_language_and_overall_metrics() -> None:
    cases = (
        EvaluationCase("zh-1", "中文查询", "zh", ("zh-a.md",)),
        EvaluationCase("en-1", "english query", "en", ("en-a.md",)),
    )
    rankings = {
        "zh-1": ("zh-a.md", "zh-b.md"),
        "en-1": ("en-b.md", "en-a.md"),
    }

    result = summarize_rankings(cases, rankings)

    assert result["zh"] == {
        "queries": 1,
        "recall_at_1": 1.0,
        "recall_at_3": 1.0,
        "recall_at_5": 1.0,
        "mrr": 1.0,
    }
    assert result["en"]["recall_at_1"] == 0.0
    assert result["en"]["mrr"] == 0.5
    assert result["overall"]["recall_at_1"] == 0.5
    assert result["overall"]["mrr"] == 0.75


def test_summarize_rankings_counts_missing_results_as_zero() -> None:
    case = EvaluationCase("zh-1", "中文查询", "zh", ("zh-a.md",))

    result = summarize_rankings((case,), {"zh-1": ()})

    assert result["zh"] == {
        "queries": 1,
        "recall_at_1": 0.0,
        "recall_at_3": 0.0,
        "recall_at_5": 0.0,
        "mrr": 0.0,
    }
