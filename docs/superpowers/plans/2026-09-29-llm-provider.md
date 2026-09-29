# LLM Provider and Structured Output Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add optional OpenAI-compatible structured LLM checks while keeping offline workflows independent.

**Architecture:** The LLM package owns typed contracts, environment configuration, safe errors, and a lazy compatible provider. The CLI delegates only the fixed check request to this package; tests inject a fake provider.

**Tech Stack:** Python 3.11, Pydantic, OpenAI Python SDK, Typer, pytest, Ruff.

**Spec:** `docs/superpowers/specs/2026-09-29-llm-provider-design.md`

## Global Constraints

- API keys must never enter source, output, logs, traces, errors, or fixtures.
- The `llm` dependency extra must not affect default offline commands.
- JSON validation retries once at most; no network tests.
- Do not modify the rule Agent or add planner/tool-selection behavior.

## Review Focus

- Missing configuration must fail before client construction.
- Authentication, timeout, rate-limit, and network errors must not leak keys.
- Invalid JSON followed by valid JSON must consume only one retry.
- Invalid JSON after retry must raise a structured-output error.
- CLI output must expose token usage but not request headers or secrets.

---

### Task 1: Provider contracts and safe configuration

**Files:**
- Create: `src/researchflow/llm/models.py`, `config.py`, `errors.py`, `provider.py`, `__init__.py`
- Test: `tests/unit/test_llm_provider.py`
- Modify: `pyproject.toml`, `uv.lock`

**Interfaces:**
- Produces: `LLMProvider`, request/response/usage models, `LLMConfig.from_environment()`, and safe errors.

- [ ] Write failing environment/configuration and fake-provider structured-response tests.
- [ ] Implement models, config, errors, lazy OpenAI-compatible client, JSON validation, and one retry.
- [ ] Run targeted tests and commit `feat: add structured LLM provider`.

### Task 2: CLI, examples, and acceptance tests

**Files:**
- Modify: `src/researchflow/cli.py`, `tests/test_cli.py`, `README.md`, `.gitignore`
- Create: `.env.example`

**Interfaces:**
- Consumes: Task 1 provider contracts.
- Produces: `researchflow llm-check`.

- [ ] Write failing Fake Provider CLI tests for success and safe error output.
- [ ] Implement fixed structured check command and dependency-injection seam.
- [ ] Add environment variable documentation/examples and run full pytest, Ruff, format, and offline CLI checks.
- [ ] Commit `feat: add LLM configuration check CLI`.
