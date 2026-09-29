"""Typed LLM request, response, and usage models."""

from pydantic import BaseModel, Field


class TokenUsage(BaseModel):
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)


class LLMRequest(BaseModel):
    user_prompt: str = Field(min_length=1)
    system_prompt: str = "Return only valid JSON."
    max_output_tokens: int = Field(default=256, ge=1)


class LLMResponse(BaseModel):
    content: str
    usage: TokenUsage
