# LLM Provider and Structured Output Design

## Goal

Provide an optional OpenAI-compatible LLM integration for configuration checks
and validated structured responses without changing the offline rule agent.

## Configuration

`RESEARCHFLOW_LLM_API_KEY`, `RESEARCHFLOW_LLM_BASE_URL`,
`RESEARCHFLOW_LLM_MODEL`, and `RESEARCHFLOW_LLM_TIMEOUT` configure the provider.
The API key is read only at runtime, never included in models, logs, traces,
errors, fixtures, or example values.

## Architecture

The `llm` package defines Pydantic request/response/token models and an
`LLMProvider` protocol. `OpenAICompatibleProvider` lazily imports the optional
OpenAI client and calls a compatible chat-completions endpoint with JSON output.
It parses JSON into the requested Pydantic model and retries one time only for
invalid structured output.

Configuration, authentication, timeout, rate-limit, connection, and structure
failures map to dedicated safe errors. The CLI receives only a sanitized message.

## CLI and tests

`researchflow llm-check` sends a fixed non-sensitive prompt, validates a small
structured response, and prints the result with token usage. Tests inject a fake
provider and cover retry/error behavior without network access. `openai` is an
optional `llm` extra; offline commands do not import it.
