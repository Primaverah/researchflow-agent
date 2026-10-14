from researchflow.reranking.factory import RerankingConfig, build_rerankers


def test_factory_does_not_import_models_when_embedding_is_disabled() -> None:
    bundle = build_rerankers(RerankingConfig(embedding_enabled=False))

    assert bundle.candidate_ranker is not None
    assert bundle.body_reranker is not None
    assert bundle.fallback_reasons == []
