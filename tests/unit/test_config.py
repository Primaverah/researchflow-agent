"""Tests for local secret configuration normalization."""

import os

from researchflow.config import load_local_secrets


def test_load_local_secrets_loads_llm_runtime_settings_and_unquotes_values(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.delenv("RESEARCHFLOW_SEARCH_API_KEY", raising=False)
    monkeypatch.delenv("RESEARCHFLOW_LLM_API_KEY", raising=False)
    monkeypatch.delenv("RESEARCHFLOW_LLM_BASE_URL", raising=False)
    monkeypatch.delenv("RESEARCHFLOW_LLM_MODEL", raising=False)
    monkeypatch.delenv("RESEARCHFLOW_LLM_TIMEOUT", raising=False)
    (tmp_path / ".env.local").write_text(
        "\n".join(
            (
                'RESEARCHFLOW_SEARCH_API_KEY="quoted-search-key"',
                'RESEARCHFLOW_LLM_API_KEY="quoted-llm-key"',
                "RESEARCHFLOW_LLM_BASE_URL='https://llm.example/v1'",
                "RESEARCHFLOW_LLM_MODEL=example-model",
                "RESEARCHFLOW_LLM_TIMEOUT=45",
            )
        )
        + "\n",
        encoding="utf-8",
    )

    load_local_secrets(tmp_path)

    assert os.getenv("RESEARCHFLOW_SEARCH_API_KEY") == "quoted-search-key"
    assert os.getenv("RESEARCHFLOW_LLM_API_KEY") == "quoted-llm-key"
    assert os.getenv("RESEARCHFLOW_LLM_BASE_URL") == "https://llm.example/v1"
    assert os.getenv("RESEARCHFLOW_LLM_MODEL") == "example-model"
    assert os.getenv("RESEARCHFLOW_LLM_TIMEOUT") == "45"
