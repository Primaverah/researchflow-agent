# LLM Configuration Injection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Inject one precedence-resolved LLM configuration into every LLM call and produce verifiable, secret-safe decisions.

**Architecture:** Extend `LLMConfig` as the sole configuration resolver and let a shared CLI factory construct the Provider from it. Components receive role-specific request settings from that config; Provider responses feed exact decision-trace fields.

**Tech Stack:** Python 3.11, Pydantic, Typer, OpenAI SDK, pytest, Ruff.

**Spec:** `docs/superpowers/specs/2026-09-29-llm-configuration-design.md`

## Global Constraints

- Precedence is CLI > environment > TOML > defaults for non-secret LLM configuration.
- Scope is LLM only; do not change Web/Tavily behavior.
- Do not read, print, trace, or otherwise expose `.env.local` secrets.
- Planner, Selector, and Summarizer budgets are exactly 1024, 1024, and 2048 when configured as in `researchflow.toml`.
- Do not push changes.

## Review Focus

- A CLI value of `False` or `0` must override TOML rather than be treated as absent.
- Environment configuration must override TOML even when TOML is present.
- Provider request arguments must not fall back to 256 output tokens.
- Retries must apply to transport calls and structured-output retries without creating unbounded attempts.
- Trace serialization must not contain an API key even when a provider error includes one.

---

### Task 1: Resolve effective LLM configuration

**Files:**
- Modify: `src/researchflow/llm/config.py`
- Modify: `src/researchflow/config.py`
- Test: `tests/unit/test_llm_config.py`

**Interfaces:**
- Produces: `LLMConfig.resolve(*, cli: Mapping[str, object] | None = None) -> LLMConfig`.
- Produces: role budget and request/provider settings as typed fields on `LLMConfig`.

- [ ] **Step 1: Write failing precedence and default tests**

Test TOML, environment, and explicit CLI values for every requested field,
including false `thinking` and the three role budgets.

- [ ] **Step 2: Run the configuration tests to verify failure**

Run: `uv run pytest tests/unit/test_llm_config.py -v`
Expected: FAIL because the resolver and fields do not exist.

- [ ] **Step 3: Implement typed resolution in `LLMConfig`**

Load permitted secrets through the existing loader; resolve each non-secret
field in the required order without logging the key.

- [ ] **Step 4: Run the configuration tests to verify they pass**

Run: `uv run pytest tests/unit/test_llm_config.py -v`
Expected: PASS.

### Task 2: Carry configuration into requests and Provider calls

**Files:**
- Modify: `src/researchflow/llm/models.py`
- Modify: `src/researchflow/llm/provider.py`
- Modify: `src/researchflow/agent/llm_components.py`
- Test: `tests/unit/test_llm_provider.py`
- Test: `tests/unit/agent/test_llm_components.py`

**Interfaces:**
- Consumes: `LLMConfig` from Task 1.
- Produces: configured `LLMRequest` values and OpenAI-compatible completion arguments.

- [ ] **Step 1: Write failing tests for role budgets and Provider forwarding**

Assert the three components create requests with their configured budgets, and
assert Provider calls receive model, response format, thinking, timeout, and
bounded retries from config.

- [ ] **Step 2: Run focused tests to verify failure**

Run: `uv run pytest tests/unit/test_llm_provider.py tests/unit/agent/test_llm_components.py -v`
Expected: FAIL because settings are hard-coded or absent.

- [ ] **Step 3: Implement request settings and Provider forwarding**

Remove the `256` default path; use required per-role request budgets and
forward supported configuration exactly once per completion attempt.

- [ ] **Step 4: Run focused tests to verify they pass**

Run: `uv run pytest tests/unit/test_llm_provider.py tests/unit/agent/test_llm_components.py -v`
Expected: PASS.

### Task 3: Unify entry points and expose verified decision results

**Files:**
- Modify: `src/researchflow/cli.py`
- Modify: `src/researchflow/agent/runner.py`
- Modify: `src/researchflow/execution/recorder.py`
- Modify: `src/researchflow/domain/models.py`
- Test: `tests/test_cli.py`
- Test: `tests/unit/execution/test_recorder.py`

**Interfaces:**
- Consumes: shared Provider factory and configured components from Tasks 1–2.
- Produces: `llm-check` output containing component, usage, finish reason,
  success, and fallback, with a nonzero exit on acceptance failure.

- [ ] **Step 1: Write failing tests for shared construction, check output, and secret-safe traces**

Verify `llm-check` and AgentRunner call the same factory, traces capture exact
decision fields, and serialized records omit secret strings.

- [ ] **Step 2: Run CLI and recorder tests to verify failure**

Run: `uv run pytest tests/test_cli.py tests/unit/execution/test_recorder.py -v`
Expected: FAIL because the output and factory contract are incomplete.

- [ ] **Step 3: Implement shared construction and acceptance output**

Construct configured components through one factory and emit one sanitized,
machine-readable status record per role. Preserve existing fallback behavior.

- [ ] **Step 4: Run CLI and recorder tests to verify they pass**

Run: `uv run pytest tests/test_cli.py tests/unit/execution/test_recorder.py -v`
Expected: PASS.

### Task 4: Full offline and live verification

**Files:**
- Modify: only files required by failed tests in Tasks 1–3.

**Interfaces:**
- Consumes: all previous tasks.
- Produces: recorded verification evidence.

- [ ] **Step 1: Run the full offline suite**

Run: `uv run pytest`
Expected: PASS.

- [ ] **Step 2: Run Ruff**

Run: `uv run ruff check .`
Expected: PASS.

- [ ] **Step 3: Run up to three real checks**

Run: `uv run --env-file .env.local researchflow llm-check`
Expected: Planner, Selector, and Summarizer each report nonzero usage,
`success=true`, `fallback=false`, and `finish_reason=stop`.
