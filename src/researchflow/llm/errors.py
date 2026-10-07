"""Safe LLM error types that never contain credentials."""


class LLMError(RuntimeError):
    def __init__(
        self, message: str, diagnostic: dict[str, object] | None = None
    ) -> None:
        super().__init__(message)
        self.diagnostic = diagnostic or {}


class LLMConfigurationError(LLMError):
    pass


class LLMDependencyError(LLMError):
    """Raised when the selected LLM mode lacks its optional client package."""


class LLMStructuredOutputError(LLMError):
    def __init__(
        self,
        message: str,
        diagnostic: dict[str, object] | None = None,
        *,
        input_tokens: int = 0,
        output_tokens: int = 0,
        finish_reason: str = "unknown",
    ) -> None:
        super().__init__(message, diagnostic)
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.finish_reason = finish_reason
