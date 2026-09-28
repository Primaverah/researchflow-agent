"""Ranking metrics for the fixed bilingual evaluation set."""

from collections.abc import Mapping, Sequence

from researchflow.evaluation.dataset import EvaluationCase

_CUTOFFS = (1, 3, 5)


def summarize_rankings(
    cases: Sequence[EvaluationCase], rankings: Mapping[str, Sequence[str]]
) -> dict[str, dict[str, float | int]]:
    """Return per-language and overall Recall@k and MRR summaries."""
    languages = sorted({case.language for case in cases})
    result = {
        language: _summarize(
            [case for case in cases if case.language == language], rankings
        )
        for language in languages
    }
    result["overall"] = _summarize(cases, rankings)
    return result


def _summarize(
    cases: Sequence[EvaluationCase], rankings: Mapping[str, Sequence[str]]
) -> dict[str, float | int]:
    query_count = len(cases)
    totals = {cutoff: 0 for cutoff in _CUTOFFS}
    reciprocal_rank_total = 0.0
    for case in cases:
        ranking = rankings.get(case.case_id, ())
        relevant = set(case.relevant_paths)
        for cutoff in _CUTOFFS:
            totals[cutoff] += any(path in relevant for path in ranking[:cutoff])
        for index, path in enumerate(ranking, start=1):
            if path in relevant:
                reciprocal_rank_total += 1 / index
                break
    if query_count == 0:
        return {
            "queries": 0,
            **{f"recall_at_{cutoff}": 0.0 for cutoff in _CUTOFFS},
            "mrr": 0.0,
        }
    return {
        "queries": query_count,
        **{f"recall_at_{cutoff}": totals[cutoff] / query_count for cutoff in _CUTOFFS},
        "mrr": reciprocal_rank_total / query_count,
    }
