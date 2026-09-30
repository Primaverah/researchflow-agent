"""OpenAI-compatible provider with validated JSON output."""

import json
import re
from abc import ABC, abstractmethod
from typing import Protocol, TypeVar

from pydantic import BaseModel, ValidationError

from researchflow.llm.config import LLMConfig
from researchflow.llm.errors import LLMError, LLMStructuredOutputError
from researchflow.llm.models import LLMRequest, LLMResponse, TokenUsage

T = TypeVar("T", bound=BaseModel)


class LLMProvider(Protocol):
    def complete(self, request: LLMRequest) -> LLMResponse: ...


class BaseLLMProvider(ABC):
    @abstractmethod
    def complete(self, request: LLMRequest) -> LLMResponse: ...

    def complete_structured(
        self, request: LLMRequest, schema: type[T]
    ) -> tuple[T, TokenUsage]:
        current_request = request
        for attempt in range(2):
            response = self.complete(current_request)
            self._last_finish_reason = response.finish_reason
            self._last_usage = response.usage
            try:
                return self._validate_structured(
                    response.content, schema
                ), response.usage
            except (ValidationError, ValueError) as exc:
                if attempt:
                    diagnostic = self._diagnostic(response, exc)
                    self._last_diagnostic = diagnostic
                    raise LLMStructuredOutputError(
                        "LLM returned invalid structured output",
                        diagnostic,
                        input_tokens=response.usage.input_tokens,
                        output_tokens=response.usage.output_tokens,
                        finish_reason=response.finish_reason,
                    ) from exc
                current_request = self._repair_request(request, schema, exc)
        raise AssertionError("unreachable")

    @staticmethod
    def _repair_request(
        request: LLMRequest, schema: type[T], error: Exception
    ) -> LLMRequest:
        validation_errors = (
            error.errors() if isinstance(error, ValidationError) else [str(error)]
        )
        repair_prompt = (
            f"{request.user_prompt}\n\nThe previous response failed schema validation. "
            "Return corrected JSON only. Validation errors: "
            + json.dumps(validation_errors, ensure_ascii=False)
            + "\nJSON Schema: "
            + json.dumps(schema.model_json_schema(), ensure_ascii=False)
        )
        return request.model_copy(update={"user_prompt": repair_prompt})

    @staticmethod
    def _diagnostic(response: LLMResponse, error: Exception) -> dict[str, object]:
        preview = response.content[:240]
        preview = re.sub(r"(?i)(api[_ -]?key\s*[:=]\s*)\S+", r"\1[REDACTED]", preview)
        preview = re.sub(r"sk-[A-Za-z0-9_-]+|Bearer\s+\S+", "[REDACTED]", preview)
        validation_errors = (
            [
                {"loc": list(item["loc"]), "msg": item["msg"], "type": item["type"]}
                for item in error.errors()
            ]
            if isinstance(error, ValidationError)
            else [{"type": type(error).__name__, "msg": str(error)[:160]}]
        )
        return {
            "http_status": response.http_status,
            "finish_reason": response.finish_reason,
            "content_preview": preview,
            "validation_errors": validation_errors,
            "usage": response.usage.model_dump(),
        }

    @staticmethod
    def _validate_structured(content: str, schema: type[T]) -> T:
        """Validate JSON, tolerating a Markdown fence from compatible APIs."""
        try:
            return schema.model_validate_json(content)
        except (ValidationError, ValueError):
            pass

        fenced = content.strip()
        if fenced.startswith("```") and fenced.endswith("```"):
            fenced = fenced.split("\n", 1)[1].rsplit("\n", 1)[0]
        decoded, end = json.JSONDecoder().raw_decode(fenced.lstrip())
        if fenced.lstrip()[end:].strip():
            raise ValueError("structured response contains trailing text")
        return schema.model_validate(decoded)


class OpenAICompatibleProvider(BaseLLMProvider):
    def __init__(self, config: LLMConfig) -> None:
        self._config = config
        self._client = None

    @property
    def model_name(self) -> str:
        """Return the configured model name without exposing credentials."""
        return self._config.model

    def complete(self, request: LLMRequest) -> LLMResponse:
        try:
            if self._client is None:
                from openai import OpenAI

                self._client = OpenAI(
                    api_key=self._config.api_key,
                    base_url=self._config.base_url,
                    timeout=self._config.timeout,
                    max_retries=self._config.retries,
                )
            response = self._client.chat.completions.create(
                model=self._config.model,
                messages=[
                    {"role": "system", "content": request.system_prompt},
                    {"role": "user", "content": request.user_prompt},
                ],
                response_format={"type": request.response_format},
                max_tokens=request.max_output_tokens,
                extra_body={
                    "thinking": {"type": "enabled" if request.thinking else "disabled"}
                },
            )
        except ImportError as exc:
            raise LLMError(
                "LLM support requires the optional openai dependency"
            ) from exc
        except Exception as exc:
            raise LLMError(
                "LLM request failed; check configuration, authentication, "
                "rate limits, and network",
                {
                    "error_type": type(exc).__name__,
                    "http_status": getattr(exc, "status_code", None),
                },
            ) from exc
        content = response.choices[0].message.content or ""
        usage = response.usage
        return LLMResponse(
            content=content,
            usage=TokenUsage(
                input_tokens=usage.prompt_tokens or 0,
                output_tokens=usage.completion_tokens or 0,
            ),
            finish_reason=response.choices[0].finish_reason or "unknown",
            http_status=getattr(
                getattr(response, "_response", None), "status_code", None
            ),
        )
