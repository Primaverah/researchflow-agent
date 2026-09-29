"""Environment-only LLM configuration."""

import os

from pydantic import BaseModel, Field

from researchflow.llm.errors import LLMConfigurationError


class LLMConfig(BaseModel):
    api_key: str = Field(min_length=1, repr=False)
    base_url: str = "https://api.openai.com/v1"
    model: str = Field(min_length=1)
    timeout: float = Field(default=30, gt=0)

    @classmethod
    def from_environment(cls) -> "LLMConfig":
        values = {
            "api_key": os.getenv("RESEARCHFLOW_LLM_API_KEY", ""),
            "base_url": os.getenv(
                "RESEARCHFLOW_LLM_BASE_URL", "https://api.openai.com/v1"
            ),
            "model": os.getenv("RESEARCHFLOW_LLM_MODEL", ""),
            "timeout": os.getenv("RESEARCHFLOW_LLM_TIMEOUT", "30"),
        }
        if not values["api_key"] or not values["model"]:
            raise LLMConfigurationError(
                "RESEARCHFLOW_LLM_API_KEY and RESEARCHFLOW_LLM_MODEL are required"
            )
        try:
            return cls(**values)
        except ValueError as exc:
            raise LLMConfigurationError(
                "LLM environment configuration is invalid"
            ) from exc
