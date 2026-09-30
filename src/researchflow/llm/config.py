"""Precedence-resolved LLM configuration."""

import os
from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, Field

from researchflow.config import load_project_config
from researchflow.llm.errors import LLMConfigurationError


class LLMConfig(BaseModel):
    api_key: str = Field(min_length=1, repr=False)
    base_url: str = "https://api.openai.com/v1"
    model: str = Field(min_length=1)
    response_format: str = Field(default="json_object", min_length=1)
    thinking: bool = False
    timeout: float = Field(default=30, gt=0)
    retries: int = Field(default=0, ge=0)
    planner_max_tokens: int = Field(default=1024, ge=1)
    selector_max_tokens: int = Field(default=1024, ge=1)
    summarizer_max_tokens: int = Field(default=2048, ge=1)

    @classmethod
    def resolve(cls, *, cli: Mapping[str, object] | None = None) -> "LLMConfig":
        config = load_project_config()
        section = config.get("llm", {})
        if not isinstance(section, dict):
            section = {}
        overrides = cli or {}

        def setting(name: str, default: Any) -> Any:
            cli_value = overrides.get(name)
            if cli_value is not None:
                return cli_value
            environment_value = os.getenv(f"RESEARCHFLOW_LLM_{name.upper()}")
            if environment_value is not None:
                return environment_value
            return section.get(name, default)

        values = {
            "api_key": os.getenv("RESEARCHFLOW_LLM_API_KEY", ""),
            "base_url": setting("base_url", "https://api.openai.com/v1"),
            "model": setting("model", ""),
            "response_format": setting("response_format", "json_object"),
            "thinking": setting("thinking", False),
            "timeout": setting("timeout", 30),
            "retries": setting("retries", 0),
            "planner_max_tokens": setting("planner_max_tokens", 1024),
            "selector_max_tokens": setting("selector_max_tokens", 1024),
            "summarizer_max_tokens": setting("summarizer_max_tokens", 2048),
        }
        if not values["api_key"] or not values["model"]:
            raise LLMConfigurationError(
                "RESEARCHFLOW_LLM_API_KEY and RESEARCHFLOW_LLM_MODEL are required"
            )
        try:
            return cls(**values)
        except ValueError as exc:
            raise LLMConfigurationError("LLM configuration is invalid") from exc

    @classmethod
    def from_environment(cls) -> "LLMConfig":
        """Compatibility alias for callers without CLI overrides."""
        return cls.resolve()
