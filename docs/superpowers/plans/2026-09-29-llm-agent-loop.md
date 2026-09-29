# LLM Agent Decision Loop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add optional LLM planner, selector, and summarizer components to the bounded local research workflow.

**Architecture:** LLM components use typed provider responses and existing state/tool contracts. The runner composes either rule or LLM components, retaining maximum-step protection and rule fallback for all configuration or output failures.

**Tech Stack:** Python 3.11, Pydantic, existing LLM Provider, pytest, Ruff.

**Spec:** `docs/superpowers/specs/2026-09-29-llm-agent-loop-design.md`

## Global Constraints

- `rule` mode remains default and behavior-compatible.
- LLM decisions can invoke only the existing three local tools.
- Retry invalid structured output once, then fall back safely.
- Never record API keys, raw prompts, raw model output, or credentials in traces.
- Do not add network search, MCP, memory, databases, or LLM tools.

## Review Focus

- A malicious selected tool name must be rejected before execution.
- A selector cannot exceed the existing maximum-step guard.
- Empty search results must finish safely without a fabricated source.
- A summary must omit failed or unread documents.
- Missing LLM configuration must select the rule fallback without changing output safety.

---

### Task 1: Structured LLM Agent components

**Files:**
- Create: `src/researchflow/agent/llm_components.py`
- Modify: `src/researchflow/agent/__init__.py`
- Test: `tests/unit/agent/test_llm_components.py`

**Interfaces:**
- Produces: `LLMPlanner`, `LLMSelector`, `LLMSummarizer`, and typed decision models.

- [ ] Write Fake Provider tests for valid plan/action/summary, invalid tool rejection, and source filtering.
- [ ] Implement typed component calls with one structured retry and rule-component fallback.
- [ ] Run targeted tests and commit `feat: add structured LLM agent components`.

### Task 2: Runner modes and safe decision tracing

**Files:**
- Modify: `src/researchflow/agent/runner.py`, `src/researchflow/domain/models.py`
- Test: `tests/unit/agent/test_runner.py`, `tests/integration/test_agent_workflow.py`

**Interfaces:**
- Consumes: Task 1 components.
- Produces: bounded LLM execution with sanitized decision metadata.

- [ ] Write failing end-to-end Fake Provider workflow, max-step, and fallback tests.
- [ ] Implement component composition and sanitized decision event recording.
- [ ] Run targeted tests and commit `feat: add LLM agent execution mode`.

### Task 3: CLI mode and regression validation

**Files:**
- Modify: `src/researchflow/cli.py`, `tests/test_cli.py`, `README.md`

**Interfaces:**
- Produces: `--agent-mode rule|llm` defaulting to `rule`.

- [ ] Write failing CLI tests for mode selection and unavailable LLM fallback.
- [ ] Implement mode option, provider assembly, and concise documentation.
- [ ] Run full pytest, Ruff, format, and offline CLI smoke checks; commit `feat: expose LLM agent mode`.
