"""Lazy optional model construction for web reranking."""

from dataclasses import dataclass, field
from typing import Any

from researchflow.config import value
from researchflow.evaluation.embeddings import SentenceTransformerProvider
from researchflow.llm import BaseLLMProvider
from researchflow.reranking.body import BodyReranker, CrossEncoder, LLMBodyReranker
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


def build_rerankers(
    config: RerankingConfig,
    *,
    llm_provider: BaseLLMProvider | None = None,
    llm_model_name: str = "configured-llm",
) -> RerankerBundle:
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
    llm_reranker = (
        LLMBodyReranker(llm_provider, model_name=llm_model_name)
        if config.llm_enabled and llm_provider is not None
        else None
    )
    return RerankerBundle(
        candidate_ranker=CandidateRanker(embedding),
        body_reranker=BodyReranker(
            embedding,
            cross_encoder,
            llm_reranker,
            cross_encoder_top_n=config.cross_encoder_top_n,
            body_relevance_threshold=config.body_relevance_threshold,
            llm_top_n=config.llm_top_n,
            llm_margin=config.llm_margin,
        ),
    )


def resolve_reranking_config(config: dict[str, Any]) -> RerankingConfig:
    """Read non-secret reranking settings with the project's env precedence."""
    return RerankingConfig(
        embedding_enabled=_bool(value(config, "reranking", "embedding_enabled", True)),
        embedding_model=str(
            value(
                config,
                "reranking",
                "embedding_model",
                "paraphrase-multilingual-MiniLM-L12-v2",
            )
        ),
        cross_encoder_enabled=_bool(
            value(config, "reranking", "cross_encoder_enabled", False)
        ),
        cross_encoder_model=str(
            value(
                config,
                "reranking",
                "cross_encoder_model",
                "cross-encoder/ms-marco-MiniLM-L-6-v2",
            )
        ),
        cross_encoder_top_n=_positive_int(
            value(config, "reranking", "cross_encoder_top_n", 5),
            "cross_encoder_top_n",
        ),
        llm_enabled=_bool(value(config, "reranking", "llm_enabled", False)),
        llm_top_n=_positive_int(
            value(config, "reranking", "llm_top_n", 3), "llm_top_n"
        ),
        llm_margin=_nonnegative_float(
            value(config, "reranking", "llm_margin", 0.08), "llm_margin"
        ),
        body_relevance_threshold=_nonnegative_float(
            value(
                config,
                "reranking",
                "body_relevance_threshold",
                0.0,
            ),
            "body_relevance_threshold",
        ),
    )


def _bool(raw: object) -> bool:
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, str) and raw.casefold() in {"true", "1", "yes", "on"}:
        return True
    if isinstance(raw, str) and raw.casefold() in {"false", "0", "no", "off"}:
        return False
    raise ValueError("reranking boolean configuration is invalid")


def _positive_int(raw: object, name: str) -> int:
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"reranking {name} must be an integer") from exc
    if value < 1:
        raise ValueError(f"reranking {name} must be positive")
    return value


def _nonnegative_float(raw: object, name: str) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"reranking {name} must be a number") from exc
    if value < 0:
        raise ValueError(f"reranking {name} must not be negative")
    return value
