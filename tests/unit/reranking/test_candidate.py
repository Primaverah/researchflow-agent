import json

import pytest

from researchflow.reranking.candidate import CandidateRanker
from researchflow.reranking.models import RankingCandidate


class FixedEmbedding:
    def __init__(self, vectors: list[list[float]]) -> None:
        self._vectors = vectors

    def embed(self, texts: list[str]) -> list[list[float]]:
        assert len(texts) == len(self._vectors)
        return self._vectors


class FailingEmbedding:
    def embed(self, texts: list[str]) -> list[list[float]]:
        raise RuntimeError("model unavailable")


def _candidates() -> list[RankingCandidate]:
    return [
        RankingCandidate(
            source_id="https://example.test/gcc",
            title="C++ compiler guide",
            summary="GCC and Clang support C++.",
            input_position=0,
        ),
        RankingCandidate(
            source_id="https://example.test/clang",
            title="C++ compiler guide",
            summary="Clang works with C++.",
            input_position=1,
        ),
    ]


def test_rrf_fuses_bm25_and_embedding_and_breaks_ties_by_input_order() -> None:
    ranking = CandidateRanker(
        embedding=FixedEmbedding(
            [
                [1.0, 0.0],
                [1.0, 0.0],
                [1.0, 0.0],
            ]
        )
    ).rank("C++ compiler", _candidates())

    assert ranking.source_ids == [
        "https://example.test/gcc",
        "https://example.test/clang",
    ]
    assert ranking.records[0].scores["bm25"] > 0
    assert ranking.records[0].scores["embedding"] == 1.0
    assert ranking.records[0].scores["rrf"] > 0


def test_embedding_failure_preserves_stable_bm25_order() -> None:
    ranking = CandidateRanker(embedding=FailingEmbedding()).rank(
        "C++ compiler", _candidates()
    )

    assert ranking.source_ids == [
        "https://example.test/clang",
        "https://example.test/gcc",
    ]
    assert ranking.records[0].fallback_reason == "embedding_unavailable"
    assert "embedding" not in ranking.records[0].scores


def test_empty_candidates_produce_an_empty_ranking() -> None:
    ranking = CandidateRanker().rank("C++ compiler", [])

    assert ranking.source_ids == []
    assert ranking.records == []


def test_embedding_scores_are_json_serializable_python_floats() -> None:
    numpy = pytest.importorskip("numpy")

    class NumpyEmbedding:
        def embed(self, texts: list[str]):
            assert len(texts) == 3
            return [
                [numpy.float32(1.0), numpy.float32(0.0)],
                [numpy.float32(1.0), numpy.float32(0.0)],
                [numpy.float32(0.0), numpy.float32(1.0)],
            ]

    ranking = CandidateRanker(embedding=NumpyEmbedding()).rank(
        "C++ compiler", _candidates()
    )

    assert isinstance(ranking.records[0].scores["embedding"], float)
    json.dumps(ranking.records[0].scores)
