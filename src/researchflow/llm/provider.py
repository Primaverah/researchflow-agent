"""OpenAI-compatible provider with validated JSON output."""

import json
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
        for attempt in range(2):
            response = self.complete(request)
            try:
                return self._validate_structured(
                    response.content, schema
                ), response.usage
            except (ValidationError, ValueError) as exc:
                if attempt:
                    raise LLMStructuredOutputError(
                        "LLM returned invalid structured output"
                    ) from exc
        raise AssertionError("unreachable")

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
                )
            response = self._client.chat.completions.create(
                model=self._config.model,
                messages=[
                    {"role": "system", "content": request.system_prompt},
                    {"role": "user", "content": request.user_prompt},
                ],
                response_format={"type": "json_object"},
                max_tokens=request.max_output_tokens,
            )
        except ImportError as exc:
            raise LLMError(
                "LLM support requires the optional openai dependency"
            ) from exc
        except Exception as exc:
            raise LLMError(
                "LLM request failed; check configuration, authentication, "
                "rate limits, and network"
            ) from exc
        content = response.choices[0].message.content or ""
        usage = response.usage
        return LLMResponse(
            content=content,
            usage=TokenUsage(
                input_tokens=usage.prompt_tokens or 0,
                output_tokens=usage.completion_tokens or 0,
            ),
        )
