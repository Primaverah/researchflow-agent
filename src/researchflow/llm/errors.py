"""Safe LLM error types that never contain credentials."""


class LLMError(RuntimeError):
    pass


class LLMConfigurationError(LLMError):
    pass


class LLMStructuredOutputError(LLMError):
    pass
