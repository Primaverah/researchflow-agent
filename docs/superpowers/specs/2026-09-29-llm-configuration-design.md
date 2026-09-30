# LLM Configuration Injection Design

## Goal

Make every LLM entry point resolve one effective configuration with the
precedence CLI > environment > `researchflow.toml` > defaults, and ensure that
the resolved values drive all LLM calls and their traces.

## Scope

Only LLM configuration and observability change. Existing Web/Tavily changes,
including their files and tests, are out of scope. `.env.local` is never
inspected directly by task work; the existing secret loader remains the only
runtime mechanism that may load its approved key names.

## Configuration model

`LLMConfig` becomes the single effective configuration object. It contains:
`api_key`, `base_url`, `model`, `response_format`, `thinking`, `timeout`,
`retries`, and role-specific `planner_max_tokens`, `selector_max_tokens`, and
`summarizer_max_tokens`.

`LLMConfig.resolve()` accepts optional CLI overrides. For every non-secret
field, it resolves CLI values first, then `RESEARCHFLOW_LLM_*` environment
values, then `[llm]` TOML values, then a fixed default. The API key is loaded
only through the existing secret-loading mechanism and is never serialized to
traces or command output.

## Request and Provider flow

The shared Provider factory receives the resolved config and is used by both
`llm-check` and AgentRunner. Role components create `LLMRequest` values with
their configured output-token budgets. The Provider forwards base URL, model,
response format, thinking, timeout, and retries to the OpenAI-compatible
client/request path. The request model no longer silently defaults output
tokens to 256.

## Observability and validation

Each component decision records the response's actual token usage and finish
reason, plus explicit success and fallback state. Trace diagnostics remain
sanitized and must not expose keys. `llm-check` emits structured per-component
status sufficient to verify Planner, Selector, and Summarizer each completed
with `success=true`, `fallback=false`, `finish_reason=stop`, and nonzero usage.
It exits nonzero if any condition is not met.

## Testing and acceptance

Unit tests cover precedence, request construction, Provider forwarding, trace
fields/sanitization, and a shared factory used by both entry points. The full
pytest suite and Ruff must pass before up to three `uv run --env-file
.env.local researchflow llm-check` attempts. No credentials are printed.
