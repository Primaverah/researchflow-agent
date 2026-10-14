from datetime import UTC, datetime

from researchflow.reranking.body import BodyReranker
from researchflow.tools.web.models import WebSource


class FixedEmbedding:
    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]


class FixedCrossEncoder:
    def __init__(self) -> None:
        self.pairs: list[tuple[str, str]] = []

    def score(self, pairs: list[tuple[str, str]]) -> list[float]:
        self.pairs = pairs
        return [0.9, 0.2]


def _sources() -> list[WebSource]:
    return [
        WebSource(
            title="Unrelated page",
            url="https://example.test/unrelated",
            accessed_at=datetime(2026, 10, 14, tzinfo=UTC),
            content="A page without compiler details.",
        ),
        WebSource(
            title="C++ compiler guide",
            url="https://example.test/compiler",
            accessed_at=datetime(2026, 10, 14, tzinfo=UTC),
            content="GCC, Clang, and MSVC are common C++ compilers.",
        ),
    ]


def test_cross_encoder_only_scores_successfully_supplied_sources() -> None:
    cross_encoder = FixedCrossEncoder()
    ranking = BodyReranker(
        embedding=FixedEmbedding(), cross_encoder=cross_encoder, cross_encoder_top_n=2
    ).rank("C++ compiler", _sources())

    assert cross_encoder.pairs == [
        (
            "C++ compiler",
            "C++ compiler guide\nGCC, Clang, and MSVC are common C++ compilers.",
        ),
        ("C++ compiler", "Unrelated page\nA page without compiler details."),
    ]
    assert ranking.source_ids == [
        "https://example.test/compiler",
        "https://example.test/unrelated",
    ]
    assert ranking.records[0].scores["cross_encoder"] == 0.9


def test_body_relevance_threshold_rejects_low_scoring_sources() -> None:
    ranking = BodyReranker(
        embedding=FixedEmbedding(), body_relevance_threshold=0.5
    ).rank("C++ compiler", _sources())

    assert ranking.accepted_source_ids == ["https://example.test/compiler"]
    assert ranking.rejected_reasons == {
        "https://example.test/unrelated": "body_relevance_below_threshold"
    }
