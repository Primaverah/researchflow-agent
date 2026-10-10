# Session Evidence and Console UI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make completed, session-backed research runs expose their verified evidence in the Web console and give the console a concise, readable interface.

**Architecture:** A session research callback will collect the research graph's safe lifecycle events and return a typed, display-safe result containing its answer and evidence projection. The application service will project that result into the existing `RunSnapshot`, preserving the single persisted run record consumed by the API. The React console will render only populated evidence groups, distinguish candidates from read evidence, and use responsive semantic layout without rendering source HTML.

**Tech Stack:** Python 3.11, Pydantic, LangGraph/SQLite, pytest, React 19, TypeScript, Vitest, CSS.

**Spec:** User-approved conversation design, 2026-10-09.

## Global Constraints

- Keep raw page bodies and search snippets out of application snapshots and browser UI.
- Preserve the current session ID / LangGraph `thread_id` mapping and checkpoint isolation.
- Do not change evidence acceptance thresholds or report Markdown semantics.
- Existing direct session callbacks returning strings remain supported.
- Do not push or merge this branch.

## Review Focus

- A completed session response with evidence must persist the same source lists after a `ResearchService` restart.
- A legacy string-only session callback must still complete normally with empty evidence.
- A session interruption must not invent evidence before its research callback runs.
- Empty evidence must use one compact empty state rather than three misleading sections.
- URLs must remain ordinary safe external links and the answer remains text, not injected HTML.

---

### Task 1: Return and persist session research evidence

**Files:**
- Modify: `src/researchflow/agent/session.py`
- Modify: `src/researchflow/application/service.py`
- Modify: `src/researchflow/cli.py`
- Test: `tests/application/test_service.py`

**Interfaces:**
- Produces: `SessionResearchResult(response: str, candidates, read_sources, rejected_sources, evidence_status)` and `ChatResult.research_result`.
- Consumes: research graph lifecycle events to project only display-safe source metadata.

- [ ] **Step 1: Write failing service tests**

Cover a typed session result with one candidate and one read web source, asserting the stored snapshot has those sources plus evidence status/policy; cover a string-only result retaining empty evidence.

- [ ] **Step 2: Run the targeted pytest cases and verify they fail because session results are not projected.**

- [ ] **Step 3: Add the typed callback result and normalize legacy string callbacks in `session.py`.**

- [ ] **Step 4: Have the CLI session callback retain the final `AgentGraphState`, and have `ResearchService` project it into the terminal snapshot.**

- [ ] **Step 5: Run targeted tests, then `uv run pytest tests/application/test_service.py tests/test_cli.py`.**

### Task 2: Render concise evidence and refresh console styling

**Files:**
- Modify: `frontend/src/App.tsx`
- Modify: `frontend/src/App.css`
- Test: `frontend/src/App.test.tsx`

**Interfaces:**
- Consumes: existing `RunSnapshot.evidence` lists.
- Produces: an accessible `研究证据` side panel with only populated groups or a single empty state.

- [ ] **Step 1: Write failing UI tests**

Assert an empty run exposes one “暂无可展示的研究证据” state and no three empty category headings; assert populated evidence has the candidate/read labels and source link.

- [ ] **Step 2: Run the focused Vitest tests and verify they fail against the current three-`无` UI.**

- [ ] **Step 3: Implement the semantic evidence panel and responsive visual redesign.**

- [ ] **Step 4: Run the focused tests and `npm test -- --run`, then `npm run build`.**

### Task 3: Verify the integrated behavior

**Files:**
- Test: `tests/integration/test_web_console_acceptance.py` (only if an existing test seam permits a session-backed check)

- [ ] **Step 1: Add or extend an integration test that fetches a saved session and observes projected evidence metadata.**

- [ ] **Step 2: Run the test to verify it fails before the final integration adjustment, if one is needed.**

- [ ] **Step 3: Make only the minimal integration adjustment required.**

- [ ] **Step 4: Run `uv run pytest`, `uv run ruff check .`, `uv run ruff format --check .`, `npm test -- --run`, and `npm run build`.**

