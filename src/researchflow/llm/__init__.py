"""Optional structured LLM integration."""

from researchflow.llm.config import LLMConfig
from researchflow.llm.errors import (
    LLMConfigurationError,
    LLMDependencyError,
    LLMError,
    LLMStructuredOutputError,
)
from researchflow.llm.models import LLMRequest, LLMResponse, TokenUsage
from researchflow.llm.provider import (
    BaseLLMProvider,
    LLMProvider,
    OpenAICompatibleProvider,
    require_openai_dependency,
)

__all__ = [
    "BaseLLMProvider",
    "LLMConfig",
    "LLMConfigurationError",
    "LLMDependencyError",
    "LLMError",
    "LLMProvider",
    "LLMRequest",
    "LLMResponse",
    "LLMStructuredOutputError",
    "OpenAICompatibleProvider",
    "require_openai_dependency",
    "TokenUsage",
]
