"""Tests for local secret configuration normalization."""

import os

from researchflow.config import load_local_secrets


def test_load_local_secrets_removes_matching_outer_quotes(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.delenv("RESEARCHFLOW_SEARCH_API_KEY", raising=False)
    (tmp_path / ".env.local").write_text(
        'RESEARCHFLOW_SEARCH_API_KEY="quoted-search-key"\n', encoding="utf-8"
    )

    load_local_secrets(tmp_path)

    assert os.getenv("RESEARCHFLOW_SEARCH_API_KEY") == "quoted-search-key"
