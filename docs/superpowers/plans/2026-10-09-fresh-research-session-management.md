# Fresh Research and Session Management Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ensure time-sensitive research reaches a truthful terminal state and add readable, renameable, deletable local sessions without changing LangGraph thread identity.

**Architecture:** Preserve `session_id` as the shared SQLite/LangGraph `thread_id`, and add separate application metadata for its human-facing display name. Carry graph execution metadata through event projection into persisted run snapshots so the API/UI cannot label a budget stop as completed. Add a small freshness-intent helper used by query generation, stable candidate ordering, and evidence admission before the existing grounded-answer pipeline.

**Tech Stack:** Python 3.11+, Pydantic, LangGraph, SQLite, FastAPI, Typer, React, TypeScript, Vitest, pytest, Ruff.

**Spec:** `docs/superpowers/specs/2026-10-09-fresh-research-and-session-management-design.md`

## Global Constraints

- `session_id` remains the only LangGraph `thread_id`; display names never alter checkpointing, history, or resume behavior.
- Keep raw search snippets and page bodies out of application snapshots and browser UI.
- Admit evidence only after a successful read, relevance review, and the policy for the active question.
- Retain bounded execution, SSRF protections, and one shared persistent SQLite database across sessions.
- Delete only the explicitly named session and remove both application-owned and exact-thread checkpoint rows.
- A terminal run is `completed` only when graph `end_reason` is `completed`.
- Do not generate factual current-event answers from stale, failed, or absent evidence.

## Review Focus

- A user-supplied lower `--max-steps` value must still stop safely and be visibly non-completed; cover it in Task 1.
- An explicit year such as `2025` must not be rewritten to the execution year; cover it in Task 2.
- A current-page source without a year must not silently satisfy a current-list request; cover it in Task 2.
- Deleting one session must leave a second session's runs and checkpoint rows intact; cover it in Task 3.
- Rename/delete network failures must preserve the selected session and expose an accessible error; cover it in Task 5.

---

## File Structure

| Path | Responsibility |
| --- | --- |
| `src/researchflow/agent/evidence_policy.py` | Detect freshness/list intent, construct retrieval query, rank/reject stale candidates, and apply evidence thresholds. |
| `src/researchflow/agent/graph.py` | Preserve graph budget metadata in state and finish events. |
| `src/researchflow/agent/research_graph.py` | Resolve the default node budget and retain it across LangGraph nodes. |
| `src/researchflow/agent/session.py` | Project graph terminal metadata into `SessionResearchResult`; manage exact checkpoint/catalog deletion. |
| `src/researchflow/application/models.py` | Define safe session-list and run-execution metadata models. |
| `src/researchflow/application/store.py` | Persist session display metadata and exact-session application-row deletion. |
| `src/researchflow/application/service.py` | Project terminal graph state honestly and expose session rename/delete operations. |
| `src/researchflow/api/app.py` | Serve session records plus PATCH/DELETE endpoints. |
| `src/researchflow/cli.py`, `researchflow.toml` | Resolve one configured node-budget default for direct and session runners. |
| `frontend/src/api.ts`, `frontend/src/App.tsx`, `frontend/src/styles.css` | Render session names and implement rename/delete UI with error states. |
| `tests/`, `frontend/src/*.test.tsx` | Pin graph, policy, store/API, and browser regressions before implementation. |

### Task 1: Preserve execution-budget truth from graph to run status

**Files:**
- Modify: `researchflow.toml`
- Modify: `src/researchflow/agent/graph.py`
- Modify: `src/researchflow/agent/research_graph.py`
- Modify: `src/researchflow/agent/session.py`
- Modify: `src/researchflow/application/models.py`
- Modify: `src/researchflow/application/service.py`
- Modify: `src/researchflow/cli.py`
- Test: `tests/agent/test_graph.py`
- Test: `tests/agent/test_research_graph.py`
- Test: `tests/application/test_service.py`
- Test: `tests/test_cli.py`

**Interfaces:**
- Produces `resolve_max_steps(agent_config: Mapping[str, object], max_replans: int) -> int`, returning configured `agent.max_steps` or `10 + (3 * max_replans)`.
- Produces `AgentGraphState.max_steps: int`, `SessionResearchResult.end_reason: str`, `SessionResearchResult.node_steps: int`, and `SessionResearchResult.max_steps: int`.
- Produces `RunSnapshot.end_reason: str | None`, `RunSnapshot.node_steps: int | None`, and `RunSnapshot.max_steps: int | None`.
- Produces a `RunStatus.FAILED` snapshot when the final `end_reason` is `max_steps`; `completed` is reserved for `end_reason == completed`.

- [ ] **Step 1: Write failing graph and projection tests**

```python
def test_default_budget_allows_one_replan_to_reach_finish() -> None:
    result = runner_with_one_failed_retrieval.run("今年的诺贝尔奖目前出炉了哪些")
    assert result.end_reason.value == "completed"
    assert result.node_steps <= result.max_steps == 13

def test_lower_budget_is_persisted_as_non_completed_terminal_state() -> None:
    snapshot = service_with_max_steps(1).start_turn(session_id="budget", question="问题")
    assert snapshot.status.value == "failed"
    assert snapshot.end_reason == "max_steps"
    assert snapshot.node_steps == 1
```

- [ ] **Step 2: Run the focused tests to verify they fail**

Run: `uv run pytest tests/agent/test_graph.py tests/agent/test_research_graph.py tests/application/test_service.py tests/test_cli.py -q`

Expected: FAIL because graph budget fields and non-completed projection are absent or session execution still hard-codes `10`.

- [ ] **Step 3: Implement the resolved budget and safe terminal metadata**

Set `[agent].max_steps` in `researchflow.toml` to `13`. Make direct CLI and session workflow construction call `resolve_max_steps`; an explicit CLI override remains authoritative. Add `max_steps` to every initial/updated `AgentGraphState`, include `end_reason`, `node_steps`, and `max_steps` in the `run_completed` event, parse those fields in `SessionResearchResult.from_events`, and persist them in `RunSnapshot`. When projection observes `max_steps`, persist a failed terminal run with the existing human-readable budget explanation instead of overwriting it from evidence status.

- [ ] **Step 4: Run the focused tests to verify they pass**

Run: `uv run pytest tests/agent/test_graph.py tests/agent/test_research_graph.py tests/application/test_service.py tests/test_cli.py -q`

Expected: PASS.

- [ ] **Step 5: Commit the graph-budget change**

```powershell
git add researchflow.toml src/researchflow/agent/graph.py src/researchflow/agent/research_graph.py src/researchflow/agent/session.py src/researchflow/application/models.py src/researchflow/application/service.py src/researchflow/cli.py tests/agent/test_graph.py tests/agent/test_research_graph.py tests/application/test_service.py tests/test_cli.py
git commit -m "fix: report graph budget exhaustion honestly"
```

### Task 2: Add freshness-aware query, ranking, and evidence admission

**Files:**
- Modify: `src/researchflow/agent/evidence_policy.py`
- Modify: `src/researchflow/agent/graph.py`
- Modify: `src/researchflow/agent/research_graph.py`
- Test: `tests/agent/test_evidence_policy.py`
- Test: `tests/agent/test_graph.py`
- Test: `tests/agent/test_research_graph.py`

**Interfaces:**
- Produces immutable `FreshnessIntent(target_year: int | None, requires_current_evidence: bool, requires_list_evidence: bool)`.
- Produces `classify_freshness_intent(query: str, *, today: date) -> FreshnessIntent` and `build_retrieval_query(query: str, intent: FreshnessIntent) -> str`.
- Produces `rank_candidates_for_query(candidates: Sequence[Source], query: str, intent: FreshnessIntent) -> list[Source]`, which preserves provider order for equal scores.
- Extends `evidence_rejection_reason(...) -> str | None` with `stale_for_current_query` and preserves the existing relevance checks.

- [ ] **Step 1: Write failing policy and graph tests**

```python
def test_current_list_query_is_year_qualified_and_requires_two_sources() -> None:
    intent = classify_freshness_intent("今年的诺贝尔奖目前出炉了哪些", today=date(2026, 10, 9))
    assert build_retrieval_query("今年的诺贝尔奖目前出炉了哪些", intent).endswith("2026")
    assert intent.requires_list_evidence is True
    assert policy.required_source_count(intent) == 2

def test_current_list_ranks_2026_before_2022_and_rejects_stale_body() -> None:
    ranked = rank_candidates_for_query([source_2022, source_2026], QUERY, intent_2026)
    assert [item.url for item in ranked] == [source_2026.url, source_2022.url]
    assert evidence_rejection_reason(source_2022, QUERY, intent_2026) == "stale_for_current_query"

def test_explicit_year_is_not_replaced_by_execution_year() -> None:
    assert classify_freshness_intent("2025年诺贝尔奖名单", today=date(2026, 10, 9)).target_year == 2025
```

- [ ] **Step 2: Run the focused tests to verify they fail**

Run: `uv run pytest tests/agent/test_evidence_policy.py tests/agent/test_graph.py tests/agent/test_research_graph.py -q`

Expected: FAIL because `哪些` is not list intent, retrieval does not include 2026, and old pages can satisfy generic evidence.

- [ ] **Step 3: Implement `FreshnessIntent` and integrate it at the retrieval boundary**

Detect explicit years first; otherwise resolve `今年`, `本年度`, `当前`, `最新`, `currently`, and `latest` with the execution date. Keep the user-facing standalone query unchanged, place the target year only in the retrieval query/trace, and use it to stably order candidates before the bounded read batch. Treat `哪些` as list intent. For current-list evidence, require a target-year marker in accepted title/content and reject other successfully-read pages with `stale_for_current_query`; retain the two-distinct-source threshold except for existing configured official complete-list logic.

- [ ] **Step 4: Run the focused tests to verify they pass**

Run: `uv run pytest tests/agent/test_evidence_policy.py tests/agent/test_graph.py tests/agent/test_research_graph.py -q`

Expected: PASS.

- [ ] **Step 5: Commit the freshness-policy change**

```powershell
git add src/researchflow/agent/evidence_policy.py src/researchflow/agent/graph.py src/researchflow/agent/research_graph.py tests/agent/test_evidence_policy.py tests/agent/test_graph.py tests/agent/test_research_graph.py
git commit -m "fix: require current evidence for fresh list questions"
```

### Task 3: Persist session display metadata and exact-session deletion

**Files:**
- Modify: `src/researchflow/application/models.py`
- Modify: `src/researchflow/application/store.py`
- Modify: `src/researchflow/agent/session.py`
- Modify: `src/researchflow/application/service.py`
- Modify: `src/researchflow/cli.py`
- Test: `tests/application/test_store.py`
- Test: `tests/application/test_service.py`
- Test: `tests/agent/test_session.py`
- Test: `tests/test_cli.py`

**Interfaces:**
- Produces `SessionRecord(session_id: str, display_name: str, updated_at: datetime)` and `SessionRename(display_name: str)` models.
- Produces `SqliteRunStore.list_sessions() -> list[SessionRecord]`, `rename_session(session_id: str, display_name: str) -> SessionRecord`, and `delete_session(session_id: str) -> bool`.
- Produces `SessionCatalog.delete_exact(session_id: str) -> bool`, deleting only matching `session_catalog`, `checkpoints.thread_id`, and `writes.thread_id` entries.
- Produces `ResearchService.rename_session(session_id: str, display_name: str) -> SessionRecord` and `ResearchService.delete_session(session_id: str) -> None`.

- [ ] **Step 1: Write failing persistence and isolation tests**

```python
def test_first_turn_creates_display_name_from_first_question() -> None:
    service.start_turn(session_id="alpha", question="村上春树是谁")
    assert store.list_sessions()[0].display_name == "村上春树是谁"

def test_rename_changes_metadata_only() -> None:
    assert service.rename_session("alpha", "作者资料").display_name == "作者资料"
    assert checkpoint_thread_ids(connection) == {"alpha"}

def test_delete_removes_only_target_application_and_checkpoint_rows() -> None:
    service.delete_session("alpha")
    assert no_rows_for_session(connection, "alpha")
    assert rows_for_session(connection, "beta")
```

- [ ] **Step 2: Run the focused tests to verify they fail**

Run: `uv run pytest tests/application/test_store.py tests/application/test_service.py tests/agent/test_session.py tests/test_cli.py -q`

Expected: FAIL because session metadata and coordinated exact-session deletion do not exist.

- [ ] **Step 3: Implement the metadata schema and coordinated deletion**

Create `researchflow_session_metadata(session_id TEXT PRIMARY KEY, display_name TEXT NOT NULL, updated_at TEXT NOT NULL)` during store initialization. Upsert a fallback display name derived from the first persisted question without replacing manual names; reject trimmed-empty rename values. Delete all exact-session application rows (`researchflow_runs`, events, start/resume idempotency rows, metadata) and then remove the exact catalog/checkpoint/write rows in one SQLite transaction boundary appropriate to the existing connection model. Make the existing CLI session deletion path call the same coordinated service, not only `SessionCatalog.delete`.

- [ ] **Step 4: Run the focused tests to verify they pass**

Run: `uv run pytest tests/application/test_store.py tests/application/test_service.py tests/agent/test_session.py tests/test_cli.py -q`

Expected: PASS.

- [ ] **Step 5: Commit the session persistence change**

```powershell
git add src/researchflow/application/models.py src/researchflow/application/store.py src/researchflow/agent/session.py src/researchflow/application/service.py src/researchflow/cli.py tests/application/test_store.py tests/application/test_service.py tests/agent/test_session.py tests/test_cli.py
git commit -m "feat: persist and delete named sessions safely"
```

### Task 4: Expose named-session management through the API

**Files:**
- Modify: `src/researchflow/api/app.py`
- Modify: `src/researchflow/application/models.py`
- Test: `tests/api/test_app.py`

**Interfaces:**
- `GET /api/sessions -> list[SessionRecord]`, newest first.
- `PATCH /api/sessions/{session_id}` consumes `SessionRename` and returns `SessionRecord`.
- `DELETE /api/sessions/{session_id}` returns HTTP 204, while unknown/deleted IDs return HTTP 404.
- `GET /api/sessions/{session_id}` retains its historical-runs response and returns HTTP 404 after deletion.

- [ ] **Step 1: Write failing API contract tests**

```python
def test_session_api_lists_and_renames_records(client: TestClient) -> None:
    assert client.get("/api/sessions").json() == [{"session_id": "alpha", "display_name": "第一问"}]
    response = client.patch("/api/sessions/alpha", json={"display_name": "资料研究"})
    assert response.status_code == 200
    assert response.json()["display_name"] == "资料研究"

def test_delete_session_returns_204_then_404(client: TestClient) -> None:
    assert client.delete("/api/sessions/alpha").status_code == 204
    assert client.get("/api/sessions/alpha").status_code == 404
```

- [ ] **Step 2: Run the API tests to verify they fail**

Run: `uv run pytest tests/api/test_app.py -q`

Expected: FAIL because session listing returns strings and PATCH/DELETE routes are absent.

- [ ] **Step 3: Implement session record, rename, and delete routes**

Use the application service methods from Task 3. Validate `display_name` through the Pydantic request model, translate missing sessions to 404, and return no JSON body for a successful deletion. Do not expose snippets, bodies, keys, or checkpoint payloads.

- [ ] **Step 4: Run the API tests to verify they pass**

Run: `uv run pytest tests/api/test_app.py -q`

Expected: PASS.

- [ ] **Step 5: Commit the API change**

```powershell
git add src/researchflow/api/app.py src/researchflow/application/models.py tests/api/test_app.py
git commit -m "feat: add session rename and delete api"
```

### Task 5: Add session naming and deletion controls to the browser UI

**Files:**
- Modify: `frontend/src/api.ts`
- Modify: `frontend/src/App.tsx`
- Modify: `frontend/src/styles.css`
- Test: `frontend/src/api.test.ts`
- Test: `frontend/src/App.test.tsx`

**Interfaces:**
- Produces `SessionListItem = { session_id: string; display_name: string }`, `renameSession(sessionId: string, displayName: string)`, and `deleteSession(sessionId: string)` in `frontend/src/api.ts`.
- `App` owns `sessions: SessionListItem[]`, `sessionError: string | null`, and an inline active-session rename state.
- Selecting a deleted active session creates a fresh UUID, clears loaded report/history, and refreshes the session list after a successful API response.

- [ ] **Step 1: Write failing frontend tests**

```tsx
it("shows a display name and immutable ID subtitle", async () => {
  mockListSessions.mockResolvedValue([{ session_id: "alpha", display_name: "诺奖资料" }]);
  render(<App />);
  expect(await screen.findByText("诺奖资料")).toBeVisible();
  expect(screen.getByLabelText("会话 ID：alpha")).toBeVisible();
});

it("renames and deletes the active session", async () => {
  // rename invokes PATCH; confirmed deletion invokes DELETE then clears the report.
});

it("keeps the session selected and announces an error when rename/delete fails", async () => {
  // rejected request leaves the session visible and renders role=alert.
});
```

- [ ] **Step 2: Run the frontend tests to verify they fail**

Run: `npm --prefix frontend test -- --run src/api.test.ts src/App.test.tsx`

Expected: FAIL because API types are strings and the controls/error state do not exist.

- [ ] **Step 3: Implement accessible sidebar controls and compact styling**

Render `display_name` as the sidebar label with a compact immutable-ID subtitle/tooltip. Add labelled rename/edit and delete buttons for the active session. Rename optimistically only after a successful PATCH; require `window.confirm` before DELETE; after successful deletion generate one new UUID, clear report/history/evidence state, and refresh records. Keep selection/report intact on failure and render the server error in a `role="alert"` element. Preserve the existing evidence-panel behavior that hides empty sections.

- [ ] **Step 4: Run frontend tests and production build**

Run: `npm --prefix frontend test -- --run src/api.test.ts src/App.test.tsx`

Expected: PASS.

Run: `npm --prefix frontend run build`

Expected: production build exits 0.

- [ ] **Step 5: Commit the browser UI change**

```powershell
git add frontend/src/api.ts frontend/src/App.tsx frontend/src/styles.css frontend/src/api.test.ts frontend/src/App.test.tsx
git commit -m "feat: manage named research sessions in web ui"
```

### Task 6: Run end-to-end verification and document the user-visible behavior

**Files:**
- Modify: `README.md` (only if command/API documentation already has a session-management section)
- Test: relevant existing integration tests under `tests/`

**Interfaces:**
- Consumes the graph, evidence, persistence, API, and frontend contracts from Tasks 1–5.
- Produces a verified local command path showing that `今年的诺贝尔奖目前出炉了哪些` neither completes from 2022 evidence nor displays a contradictory completed state after a budget stop.

- [ ] **Step 1: Add or extend the end-to-end regression test**

```python
def test_current_list_with_only_stale_read_source_is_insufficient_not_completed() -> None:
    snapshot = run_fixture_search(query="今年的诺贝尔奖目前出炉了哪些", sources=[source_2022])
    assert snapshot.status.value != "completed"
    assert "stale_for_current_query" in snapshot.evidence_gaps
```

- [ ] **Step 2: Run the new regression against the completed dependencies**

Run: `uv run pytest tests -q -k "current_list_with_only_stale_read_source"`

Expected: PASS after Tasks 1–2; if it fails, fix the responsible task before proceeding.

- [ ] **Step 3: Update concise usage documentation if an existing session command/API section exists**

Document the default graph-node budget, that latest/current questions require current evidence, and the browser's rename/delete behavior. Do not promise online verification where credentials or network access are unavailable.

- [ ] **Step 4: Run the complete automated verification suite**

Run: `uv sync --frozen`

Expected: exits 0.

Run: `uv lock --check`

Expected: exits 0.

Run: `uv run pytest`

Expected: all Python tests pass.

Run: `uv run ruff check .`

Expected: exits 0.

Run: `uv run ruff format --check .`

Expected: exits 0.

Run: `npm --prefix frontend test -- --run`

Expected: all frontend tests pass.

Run: `npm --prefix frontend run build`

Expected: exits 0.

- [ ] **Step 5: Perform a bounded local smoke test and record its evidence**

Run: `uv run researchflow chat "今年的诺贝尔奖目前出炉了哪些" --session-id freshness-smoke --enable-web --agent-mode llm`

Expected: if online search is available, output identifies current evidence or explicitly reports an evidence/fetch gap; it never uses an old-year source as a current list or reports `completed` after `max_steps`. If network/credentials prevent the test, record the concrete restriction and do not claim online success.

- [ ] **Step 6: Commit verification documentation and final changes**

```powershell
git add README.md tests
git commit -m "test: cover fresh research and session lifecycle"
git status --short
git log --oneline origin/main..HEAD
```

