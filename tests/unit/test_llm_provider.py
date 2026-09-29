"""Tests for optional structured LLM integration without network access."""

import json

import pytest
from pydantic import BaseModel

from researchflow.llm import (
    BaseLLMProvider,
    LLMConfig,
    LLMRequest,
    LLMResponse,
    TokenUsage,
)


class CheckResult(BaseModel):
    status: str


class FakeProvider(BaseLLMProvider):
    def __init__(self, contents: list[str]) -> None:
        self.contents = contents
        self.calls = 0

    def complete(self, request: LLMRequest) -> LLMResponse:
        content = self.contents[self.calls]
        self.calls += 1
        return LLMResponse(
            content=content, usage=TokenUsage(input_tokens=2, output_tokens=1)
        )


def test_config_reads_required_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RESEARCHFLOW_LLM_API_KEY", "secret")
    monkeypatch.setenv("RESEARCHFLOW_LLM_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setenv("RESEARCHFLOW_LLM_MODEL", "model")
    monkeypatch.setenv("RESEARCHFLOW_LLM_TIMEOUT", "12")

    config = LLMConfig.from_environment()

    assert config.model == "model"
    assert config.timeout == 12


def test_structured_response_retries_invalid_json_once() -> None:
    provider = FakeProvider(["not-json", json.dumps({"status": "ok"})])

    result, usage = provider.complete_structured(
        LLMRequest(user_prompt="check"), CheckResult
    )

    assert result == CheckResult(status="ok")
    assert usage.output_tokens == 1
    assert provider.calls == 2
