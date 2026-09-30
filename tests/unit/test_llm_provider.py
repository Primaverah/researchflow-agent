"""Tests for optional structured LLM integration without network access."""

import json

import pytest
from pydantic import BaseModel

from researchflow.llm import (
    BaseLLMProvider,
    LLMConfig,
    LLMError,
    LLMRequest,
    LLMResponse,
    OpenAICompatibleProvider,
    TokenUsage,
)


class CheckResult(BaseModel):
    status: str


class FakeProvider(BaseLLMProvider):
    def __init__(self, contents: list[str]) -> None:
        self.contents = contents
        self.calls = 0
        self.requests: list[LLMRequest] = []

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
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


def test_config_resolves_cli_over_environment_over_toml(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "researchflow.llm.config.load_project_config",
        lambda: {
            "llm": {
                "base_url": "https://toml.invalid/v1",
                "model": "toml-model",
                "response_format": "json_object",
                "thinking": True,
                "timeout": 10,
                "retries": 1,
                "planner_max_tokens": 100,
                "selector_max_tokens": 200,
                "summarizer_max_tokens": 300,
            }
        },
    )
    monkeypatch.setenv("RESEARCHFLOW_LLM_API_KEY", "secret")
    monkeypatch.setenv("RESEARCHFLOW_LLM_MODEL", "environment-model")
    monkeypatch.setenv("RESEARCHFLOW_LLM_TIMEOUT", "20")
    monkeypatch.setenv("RESEARCHFLOW_LLM_THINKING", "false")

    config = LLMConfig.resolve(
        cli={
            "base_url": "https://cli.invalid/v1",
            "model": "cli-model",
            "response_format": "json_schema",
            "thinking": False,
            "timeout": 30,
            "retries": 2,
            "planner_max_tokens": 1024,
            "selector_max_tokens": 1024,
            "summarizer_max_tokens": 2048,
        }
    )

    assert config.base_url == "https://cli.invalid/v1"
    assert config.model == "cli-model"
    assert config.response_format == "json_schema"
    assert config.thinking is False
    assert config.timeout == 30
    assert config.retries == 2
    assert config.planner_max_tokens == 1024
    assert config.selector_max_tokens == 1024
    assert config.summarizer_max_tokens == 2048


def test_provider_forwards_configured_request_settings() -> None:
    class Completions:
        def __init__(self) -> None:
            self.kwargs: dict[str, object] | None = None

        def create(self, **kwargs):
            self.kwargs = kwargs
            return type(
                "Response",
                (),
                {
                    "choices": [
                        type(
                            "Choice",
                            (),
                            {
                                "message": type("Message", (), {"content": "{}"})(),
                                "finish_reason": "stop",
                            },
                        )()
                    ],
                    "usage": type(
                        "Usage", (), {"prompt_tokens": 3, "completion_tokens": 2}
                    )(),
                },
            )()

    completions = Completions()
    provider = OpenAICompatibleProvider(
        LLMConfig(
            api_key="secret",
            base_url="https://example.invalid/v1",
            model="model",
            response_format="json_object",
            thinking=False,
            timeout=12,
            retries=2,
            planner_max_tokens=1024,
            selector_max_tokens=1024,
            summarizer_max_tokens=2048,
        )
    )
    provider._client = type(  # type: ignore[attr-defined]
        "Client", (), {"chat": type("Chat", (), {"completions": completions})()}
    )()

    provider.complete(
        LLMRequest(
            user_prompt="check",
            max_output_tokens=1024,
            response_format="json_object",
            thinking=False,
        )
    )

    assert completions.kwargs == {
        "model": "model",
        "messages": [
            {"role": "system", "content": "Return only valid JSON."},
            {"role": "user", "content": "check"},
        ],
        "response_format": {"type": "json_object"},
        "max_tokens": 1024,
        "extra_body": {"thinking": {"type": "disabled"}},
    }


def test_provider_error_exposes_only_safe_exception_metadata() -> None:
    class RequestError(Exception):
        status_code = 401

    class Completions:
        def create(self, **_kwargs):
            raise RequestError("Bearer sk-secret-value")

    provider = OpenAICompatibleProvider(LLMConfig(api_key="secret", model="model"))
    provider._client = type(  # type: ignore[attr-defined]
        "Client", (), {"chat": type("Chat", (), {"completions": Completions()})()}
    )()

    with pytest.raises(LLMError) as exc_info:
        provider.complete(LLMRequest(user_prompt="check", max_output_tokens=1))

    assert exc_info.value.diagnostic == {
        "error_type": "RequestError",
        "http_status": 401,
    }


def test_structured_response_retries_invalid_json_once() -> None:
    provider = FakeProvider(["not-json", json.dumps({"status": "ok"})])

    result, usage = provider.complete_structured(
        LLMRequest(user_prompt="check", max_output_tokens=1), CheckResult
    )

    assert result == CheckResult(status="ok")
    assert usage.output_tokens == 1
    assert provider.calls == 2
    assert "Validation errors" in provider.requests[1].user_prompt
    assert '"status"' in provider.requests[1].user_prompt


def test_structured_response_accepts_json_fenced_by_compatible_provider() -> None:
    provider = FakeProvider(['```json\n{"status": "ok"}\n```'])

    result, usage = provider.complete_structured(
        LLMRequest(user_prompt="check", max_output_tokens=1), CheckResult
    )

    assert result == CheckResult(status="ok")
    assert usage.output_tokens == 1
    assert provider.calls == 1
