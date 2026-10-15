from researchflow.reranking.factory import (
    RerankingConfig,
    build_rerankers,
    resolve_reranking_config,
)


def test_factory_does_not_import_models_when_embedding_is_disabled() -> None:
    bundle = build_rerankers(RerankingConfig(embedding_enabled=False))

    assert bundle.candidate_ranker is not None
    assert bundle.body_reranker is not None
    assert bundle.fallback_reasons == []


def test_config_uses_web_embedding_but_keeps_expensive_rerankers_opt_in() -> None:
    config = resolve_reranking_config({"reranking": {}})

    assert config.embedding_enabled is True
    assert config.cross_encoder_enabled is False
    assert config.llm_enabled is False
