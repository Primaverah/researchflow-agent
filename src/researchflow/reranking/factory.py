"""Lazy optional model construction for web reranking."""

from dataclasses import dataclass, field

from researchflow.evaluation.embeddings import SentenceTransformerProvider
from researchflow.reranking.body import BodyReranker, CrossEncoder
from researchflow.reranking.candidate import CandidateRanker


class SentenceTransformerCrossEncoder:
    """Load a sentence-transformers CrossEncoder only when scored."""

    def __init__(self, model_name: str) -> None:
        self._model_name = model_name
        self._model = None

    def score(self, pairs: list[tuple[str, str]]) -> list[float]:
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as exc:
                raise RuntimeError("sentence-transformers is required") from exc
            self._model = CrossEncoder(self._model_name)
        return [float(value) for value in self._model.predict(pairs)]


@dataclass(frozen=True, slots=True)
class RerankingConfig:
    embedding_enabled: bool = False
    embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2"
    cross_encoder_enabled: bool = False
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    cross_encoder_top_n: int = 5
    llm_enabled: bool = False
    llm_top_n: int = 3
    llm_margin: float = 0.08
    body_relevance_threshold: float = 0.0


@dataclass(frozen=True, slots=True)
class RerankerBundle:
    candidate_ranker: CandidateRanker
    body_reranker: BodyReranker
    fallback_reasons: list[str] = field(default_factory=list)


def build_rerankers(config: RerankingConfig) -> RerankerBundle:
    embedding = (
        SentenceTransformerProvider(config.embedding_model)
        if config.embedding_enabled
        else None
    )
    cross_encoder: CrossEncoder | None = (
        SentenceTransformerCrossEncoder(config.cross_encoder_model)
        if config.cross_encoder_enabled
        else None
    )
    return RerankerBundle(
        candidate_ranker=CandidateRanker(embedding),
        body_reranker=BodyReranker(
            embedding,
            cross_encoder,
            cross_encoder_top_n=config.cross_encoder_top_n,
            body_relevance_threshold=config.body_relevance_threshold,
        ),
    )
