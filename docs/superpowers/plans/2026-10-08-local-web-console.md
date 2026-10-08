# Local Web Console Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a loopback-only web console that exposes persistent research sessions, evidence-aware reports, live run events, explicit interrupt/resume, and a gradual migration of the research executor to LangGraph.

**Architecture:** Introduce a Python application-service layer shared by CLI and FastAPI, with SQLite-backed run metadata and a normalized event publisher. Build a React/Vite workbench over HTTP and SSE. Preserve the current `LangGraphSessionRunner` first, then migrate `GraphAgentRunner` node groups into a checkpointed LangGraph research graph behind the same service/event contracts.

**Tech Stack:** Python 3.11, FastAPI, Uvicorn, Pydantic v2, LangGraph, SQLite, pytest, React, TypeScript, Vite, Vitest, SSE, Ruff, uv.

**Spec:** `docs/superpowers/specs/2026-10-08-local-web-console-design.md`

## Global Constraints

- Bind the server to `127.0.0.1` by default; never expose API keys in HTTP, SSE, Trace, Markdown, browser storage, or logs.
- Reuse `<output-dir>/sessions/checkpoints.sqlite3`; `session_id` remains the LangGraph `thread_id` and sessions remain isolated.
- Search candidates, successfully read full-text sources, and rejected/failed sources remain distinct in API and UI models.
- Grounded mode never turns model knowledge, a browser clock, or a server clock into cited research evidence.
- HTTP clients, tool executors, LLM clients, and SQLite connections never enter checkpointed graph state.
- `messages` remains append-reduced; only the existing history compactor can explicitly replace historical messages.
- All new writes are UTF-8; Windows console output behavior and existing CLI commands must not regress.
- Keep existing `run`, `chat`, offline mode, Web evidence gate, and JSONL Trace behavior compatible throughout migration.

## Review Focus

- Browser refresh during a running request must reconnect without starting a duplicate run; covered by Task 5 SSE reconnect tests.
- Repeated `Idempotency-Key` or resume POST must never repeat a web fetch, tool write, or report save; covered by Tasks 2 and 6.
- A source shown as “read” must have passed successful full-text and relevance checks; covered by Tasks 1 and 4.
- A process restart with a pending interrupt must resume using the original `thread_id`; covered by Task 6.
- A front-end rendering path must never inject raw Markdown/HTML, credentials, full webpage bodies, or raw LLM payloads; covered by Tasks 3 and 5.

---

## File Structure

| Path | Responsibility |
| --- | --- |
| `src/researchflow/application/models.py` | Pydantic DTOs for runs, evidence display, session snapshots, and event envelopes. |
| `src/researchflow/application/run_store.py` | SQLite run metadata, idempotency records, and event sequence persistence. |
| `src/researchflow/application/events.py` | In-process ordered event broker and JSONL-to-event adaptation. |
| `src/researchflow/application/service.py` | Shared `ResearchService` lifecycle for CLI and HTTP callers. |
| `src/researchflow/api/app.py` | FastAPI application factory and loopback-safe configuration. |
| `src/researchflow/api/routes.py` | REST/SSE route handlers, response translation, and error handling. |
| `src/researchflow/api/server.py` | `researchflow serve` CLI command and Uvicorn startup. |
| `src/researchflow/agent/session.py` | Explicit interrupt state and service-oriented resume integration. |
| `src/researchflow/agent/research_graph.py` | New LangGraph research-state graph, introduced incrementally behind an adapter. |
| `src/researchflow/cli.py` | CLI adaptation to `ResearchService`; preserve existing commands. |
| `frontend/` | Vite React TypeScript workbench, API client, SSE client, and component tests. |
| `tests/application/` | Service/store/events integration and isolation tests. |
| `tests/api/` | FastAPI contract, SSE, loopback, and error tests. |
| `tests/integration/` | End-to-end checkpoint, interrupt/resume, old/new executor parity tests. |

### Task 1: Add application DTOs and evidence-preserving run metadata

**Files:**
- Create: `src/researchflow/application/__init__.py`
- Create: `src/researchflow/application/models.py`
- Create: `src/researchflow/application/run_store.py`
- Test: `tests/application/test_models.py`
- Test: `tests/application/test_run_store.py`

**Interfaces:**
- Consumes: `AgentState`, `SessionGraphState`, `EvidenceStatus`, JSONL trace records.
- Produces: `RunStatus`, `RunSnapshot`, `EvidenceSnapshot`, `EventEnvelope`, and `SqliteRunStore(database: Path)`.

- [ ] **Step 1: Write failing DTO tests**

```python
def test_evidence_snapshot_keeps_candidates_read_and_rejected_separate() -> None:
    snapshot = EvidenceSnapshot.from_agent_state(state)
    assert snapshot.candidates[0].read is False
    assert snapshot.read_sources[0].source_id == "source-1"
    assert snapshot.rejected_sources[0].reason == "web_low_quality_content"


def test_insufficient_evidence_is_not_answered_status() -> None:
    assert RunSnapshot(..., evidence_status="insufficient").status != "answered"
```

- [ ] **Step 2: Run the DTO tests and verify they fail**

Run: `uv run pytest tests/application/test_models.py -v`  
Expected: FAIL because `researchflow.application` does not exist.

- [ ] **Step 3: Implement JSON-safe DTOs in `application/models.py`**

Define `RunStatus` as `created`, `running`, `waiting_for_input`, `paused`, `completed`, `failed`, and `insufficient_evidence`. Define source display records with `source_id`, `title`, `url`, `kind`, `read`, and optional `reason`; do not include body text by default. `RunSnapshot` holds run/session identifiers, report availability/path, evidence status/gaps, end reason, generation status, interrupt prompt, and `last_event_id`.

- [ ] **Step 4: Write failing SQLite store tests**

```python
def test_run_store_returns_same_run_for_same_session_and_idempotency_key(tmp_path):
    store = SqliteRunStore(tmp_path / "checkpoints.sqlite3")
    first = store.create_run("session-a", "key-1")
    second = store.create_run("session-a", "key-1")
    assert second.run_id == first.run_id


def test_run_store_isolates_session_ids(tmp_path):
    ...
    assert store.list_session_runs("a") != store.list_session_runs("b")
```

- [ ] **Step 5: Implement `SqliteRunStore` and schema migration**

Create `researchflow_runs`, `researchflow_idempotency`, and `researchflow_events` tables in the existing SQLite database. Use transactions for create-or-return idempotency behavior, store UTC ISO timestamps, and make run/event payloads JSON. Do not modify LangGraph-owned checkpoint tables.

- [ ] **Step 6: Run focused tests and commit**

Run: `uv run pytest tests/application/test_models.py tests/application/test_run_store.py -v`  
Expected: PASS.

```bash
git add src/researchflow/application tests/application
git commit -m "feat: add persistent research run metadata"
```

### Task 2: Introduce `ResearchService` and move CLI lifecycle calls behind it

**Files:**
- Create: `src/researchflow/application/events.py`
- Create: `src/researchflow/application/service.py`
- Modify: `src/researchflow/cli.py`
- Test: `tests/application/test_service.py`
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: `SqliteRunStore`, `LangGraphSessionRunner`, the existing `_run_workflow` dependency through an injected `ResearchWorkflow` callable.
- Produces: `ResearchService.start_turn(request: StartTurn) -> RunSnapshot`, `get_run(run_id: str) -> RunSnapshot`, `get_session(session_id: str) -> SessionSnapshot`, `resume_turn(request: ResumeTurn) -> RunSnapshot`.

- [ ] **Step 1: Write failing lifecycle tests**

```python
def test_start_turn_persists_running_then_completed_snapshot(service):
    snapshot = service.start_turn(StartTurn(session_id="one", message="tool calling"))
    assert snapshot.status is RunStatus.COMPLETED


def test_duplicate_start_does_not_execute_workflow_twice(service):
    service.start_turn(StartTurn(session_id="one", message="q", idempotency_key="k"))
    service.start_turn(StartTurn(session_id="one", message="q", idempotency_key="k"))
    assert workflow.calls == 1
```

- [ ] **Step 2: Run lifecycle tests and verify they fail**

Run: `uv run pytest tests/application/test_service.py -v`  
Expected: FAIL because `ResearchService` is undefined.

- [ ] **Step 3: Implement event broker and service methods**

`EventBroker.publish(event: EventEnvelope) -> EventEnvelope` assigns monotonically increasing per-run IDs and persists before notifying subscribers. `ResearchService` creates runs through the store, translates existing `AgentState`/session results into snapshots, and emits `run_started`, graph/tool/evidence/generation, and terminal events. Inject workflow/session factories for tests; production factories keep current `_run_workflow` behavior.

- [ ] **Step 4: Adapt `run` and `chat` to construct/use the service**

Keep all public CLI flags and output formatting. For completed runs, render the service snapshot/report. Preserve `chat` session IDs and existing `sessions list/delete`; do not introduce HTTP dependencies into the command paths.

- [ ] **Step 5: Add CLI compatibility tests**

```python
def test_chat_uses_service_and_keeps_same_session_id(...): ...
def test_run_offline_mode_keeps_existing_report_output(...): ...
```

- [ ] **Step 6: Run focused tests and commit**

Run: `uv run pytest tests/application/test_service.py tests/test_cli.py -v`  
Expected: PASS.

```bash
git add src/researchflow/application src/researchflow/cli.py tests/application tests/test_cli.py
git commit -m "refactor: route research lifecycle through service"
```

### Task 3: Add loopback-only FastAPI contract and a polling web workbench

**Files:**
- Modify: `pyproject.toml`
- Create: `src/researchflow/api/__init__.py`
- Create: `src/researchflow/api/app.py`
- Create: `src/researchflow/api/routes.py`
- Create: `src/researchflow/api/server.py`
- Modify: `src/researchflow/cli.py`
- Create: `tests/api/test_routes.py`
- Create: `frontend/package.json`
- Create: `frontend/vite.config.ts`
- Create: `frontend/src/main.tsx`
- Create: `frontend/src/App.tsx`
- Create: `frontend/src/api.ts`
- Create: `frontend/src/App.test.tsx`

**Interfaces:**
- Consumes: `ResearchService`, `RunSnapshot`, `SessionSnapshot` from Task 2.
- Produces: `create_app(service: ResearchService) -> FastAPI` and `researchflow serve --host 127.0.0.1 --port 8000`.

- [ ] **Step 1: Write failing API contract tests**

```python
def test_turn_post_returns_202_and_run_id(client):
    response = client.post("/api/sessions/demo/turns", json={"message": "tool calling"})
    assert response.status_code == 202
    assert response.json()["session_id"] == "demo"


def test_resume_rejects_non_waiting_run(client):
    assert (
        client.post("/api/sessions/demo/resume", json={"answer": "x"}).status_code
        == 409
    )
```

- [ ] **Step 2: Run API tests and verify they fail**

Run: `uv run pytest tests/api/test_routes.py -v`  
Expected: FAIL because FastAPI application modules do not exist.

- [ ] **Step 3: Add optional UI backend dependencies and FastAPI app factory**

Add a new optional dependency group named `ui` containing bounded FastAPI/Uvicorn versions and the test HTTP client dependency. `create_app` must expose only the API routes in tests; production `serve` defaults host to `127.0.0.1` and rejects any non-loopback host without a future explicit unsafe flag.

- [ ] **Step 4: Implement REST routes**

Implement the routes in the approved spec: health, session list/get/delete, start turn, resume, get run, get report, and get trace. Convert validation failures to `422`, missing records to `404`, invalid lifecycle transitions to `409`, and sanitized runtime failures to `500`. Responses use DTOs only.

- [ ] **Step 5: Scaffold React/Vite polling workbench and component tests**

Use TypeScript. Implement session list, message input, report panel, and three distinct evidence lists. Poll the selected run while it is `created` or `running`. Render report Markdown as escaped plain text in this phase; do not use unsafe HTML injection.

- [ ] **Step 6: Run API and frontend tests and commit**

Run: `uv sync --frozen --extra ui && uv run pytest tests/api/test_routes.py -v`  
Expected: PASS.

Run: `npm --prefix frontend test -- --run`  
Expected: PASS.

```bash
git add pyproject.toml uv.lock src/researchflow/api src/researchflow/cli.py tests/api frontend
git commit -m "feat: add local web research console"
```

### Task 4: Publish normalized research events and preserve source distinctions

**Files:**
- Modify: `src/researchflow/agent/graph.py`
- Modify: `src/researchflow/execution/recorder.py`
- Modify: `src/researchflow/application/events.py`
- Modify: `src/researchflow/application/service.py`
- Test: `tests/application/test_events.py`
- Test: `tests/unit/agent/test_graph.py`

**Interfaces:**
- Consumes: graph-node updates, tool execution traces, `SummaryGenerationStatus`, evidence/rejection data.
- Produces: ordered `EventEnvelope` types `run_started`, `graph_node_started`, `graph_node_finished`, `search_completed`, `candidate_selected`, `source_read`, `source_rejected`, `evidence_assessed`, `generation_status`, `run_completed`, and `run_failed`.

- [ ] **Step 1: Write failing event-order and data-redaction tests**

```python
def test_read_source_event_contains_metadata_not_full_body(run_events):
    event = next(item for item in run_events if item.type == "source_read")
    assert event.data["content_length"] > 0
    assert "content" not in event.data


def test_rejected_candidate_never_appears_in_read_source_events(run_events): ...
```

- [ ] **Step 2: Run event tests and verify they fail**

Run: `uv run pytest tests/application/test_events.py -v`  
Expected: FAIL because graph events are not normalized.

- [ ] **Step 3: Add observer hooks at graph/tool boundaries**

Add an optional event sink protocol to the graph runner and executor construction path. Publish events after the corresponding state/tool result is durable enough to render; do not publish raw argument summaries that contain credentials or full source content. Keep the current JSONL output unchanged and make its records the audit source.

- [ ] **Step 4: Add event-to-snapshot projection**

Update the run store from terminal and evidence events so a browser reload can reconstruct candidates, accepted sources, rejected sources, generation state, and end reason without replaying the workflow.

- [ ] **Step 5: Run focused regressions and commit**

Run: `uv run pytest tests/application/test_events.py tests/unit/agent/test_graph.py -v`  
Expected: PASS.

```bash
git add src/researchflow/agent/graph.py src/researchflow/execution src/researchflow/application tests/application tests/unit/agent/test_graph.py
git commit -m "feat: publish evidence-aware research events"
```

### Task 5: Add SSE, reconnect behavior, and live evidence UI

**Files:**
- Modify: `src/researchflow/api/routes.py`
- Modify: `src/researchflow/application/events.py`
- Modify: `frontend/src/api.ts`
- Create: `frontend/src/useRunEvents.ts`
- Modify: `frontend/src/App.tsx`
- Create: `tests/api/test_sse.py`
- Create: `frontend/src/useRunEvents.test.tsx`

**Interfaces:**
- Consumes: persisted `EventEnvelope` records and `EventBroker.subscribe(run_id, after_id)`.
- Produces: `GET /api/runs/{run_id}/events` with `text/event-stream` and `connectRunEvents(runId, afterEventId)`.

- [ ] **Step 1: Write failing SSE tests**

```python
def test_sse_replays_events_after_last_event_id(client, service):
    service.publish_for_test("run-1", "search_completed", {"candidate_count": 2})
    response = client.get("/api/runs/run-1/events", headers={"Last-Event-ID": "0"})
    assert "id: 1" in response.text

def test_sse_payload_has_no_secret_or_full_content(...): ...
```

- [ ] **Step 2: Run SSE tests and verify they fail**

Run: `uv run pytest tests/api/test_sse.py -v`  
Expected: FAIL because the event-stream route is absent.

- [ ] **Step 3: Implement SSE with bounded replay and keepalive**

Use persisted event IDs for replay and an in-process subscription for live delivery. Send SSE `id`, `event`, and JSON `data`; emit a harmless keepalive comment on idle. A reconnect that requests an unavailable old event range must receive a current snapshot event before live events.

- [ ] **Step 4: Implement frontend SSE reducer and evidence timeline**

On each event, reduce into the same `RunSnapshot` model used by polling. Render node progress, candidate count, selected/rejected reasons, evidence status, and generation mode. If SSE fails, fall back to polling `GET /api/runs/{id}`; never resend the original POST.

- [ ] **Step 5: Run API/frontend tests and commit**

Run: `uv run pytest tests/api/test_sse.py tests/application/test_events.py -v`  
Expected: PASS.

Run: `npm --prefix frontend test -- --run`  
Expected: PASS.

```bash
git add src/researchflow/api src/researchflow/application frontend tests/api
git commit -m "feat: stream research progress to web console"
```

### Task 6: Make interrupt/resume an explicit durable service operation

**Files:**
- Modify: `src/researchflow/agent/session.py`
- Modify: `src/researchflow/application/service.py`
- Modify: `src/researchflow/application/run_store.py`
- Modify: `src/researchflow/api/routes.py`
- Modify: `src/researchflow/cli.py`
- Modify: `frontend/src/App.tsx`
- Test: `tests/unit/agent/test_session.py`
- Test: `tests/application/test_service.py`
- Test: `tests/api/test_routes.py`
- Test: `frontend/src/App.test.tsx`

**Interfaces:**
- Consumes: `LangGraphSessionRunner.resume(session_id, answer)`, `RunStatus.WAITING_FOR_INPUT`, and `ResumeTurn(session_id, answer, idempotency_key)`.
- Produces: durable interrupt snapshots, `POST /api/sessions/{id}/resume`, and UI recovery controls.

- [ ] **Step 1: Write failing restart and duplicate-resume tests**

```python
def test_new_runner_resumes_pending_interrupt_with_original_thread_id(tmp_path):
    pending = first_service.start_turn(StartTurn(session_id="clarify", message=" "))
    resumed = rebuilt_service.resume_turn(
        ResumeTurn(session_id="clarify", answer="tool calling")
    )
    assert resumed.status is RunStatus.COMPLETED
    assert workflow.calls == ["tool calling"]


def test_duplicate_resume_key_does_not_repeat_research(service): ...
```

- [ ] **Step 2: Run interrupt tests and verify they fail**

Run: `uv run pytest tests/unit/agent/test_session.py tests/application/test_service.py -v`  
Expected: FAIL because interrupt state is not represented as a durable run lifecycle.

- [ ] **Step 3: Persist `waiting_for_input` before exposing the interrupt**

Modify the service/session adapter so an interrupt records prompt, session/thread ID, and run status before publishing `interrupt_requested`. Resume must invoke `Command(resume=answer)` with exactly that stored session ID. The service, not the Web route, owns this transition.

- [ ] **Step 4: Add resume API/UI and preserve CLI convenience**

The API returns `409` unless the session has exactly one waiting run. The Web UI disables normal send while waiting and posts to `/resume`. Preserve existing CLI automatic prompt-and-resume behavior as a convenience adapter over `ResearchService.resume_turn`.

- [ ] **Step 5: Add no-side-effect regression coverage**

Use fake tools/counters to prove that no tool call, file write, or report save occurs before interruption and no completed operation repeats on resume.

- [ ] **Step 6: Run focused tests and commit**

Run: `uv run pytest tests/unit/agent/test_session.py tests/application/test_service.py tests/api/test_routes.py -v`  
Expected: PASS.

Run: `npm --prefix frontend test -- --run`  
Expected: PASS.

```bash
git add src/researchflow/agent/session.py src/researchflow/application src/researchflow/api src/researchflow/cli.py frontend tests
git commit -m "feat: add durable web interrupt recovery"
```

### Task 7: Migrate the read/plan half of `GraphAgentRunner` to a checkpointed LangGraph research graph

**Files:**
- Create: `src/researchflow/agent/research_graph.py`
- Modify: `src/researchflow/agent/graph.py`
- Modify: `src/researchflow/application/service.py`
- Test: `tests/integration/test_research_graph_parity.py`
- Test: `tests/integration/test_research_graph_checkpoint.py`

**Interfaces:**
- Consumes: existing `AgentGraphState`, planner, executor, evidence selection logic, and event sink from Task 4.
- Produces: `LangGraphResearchRunner.run(query: str, context: ToolContext, *, answer_target: str, answer_language: str, thread_id: str) -> AgentState`.

- [ ] **Step 1: Write failing parity tests for initial nodes**

```python
def test_langgraph_runner_matches_legacy_candidates_and_evidence_for_fixture(...):
    assert new.candidates == legacy.candidates
    assert new.evidence_status == legacy.evidence_status

def test_new_research_graph_restores_after_checkpoint_before_read(...): ...
```

- [ ] **Step 2: Run parity tests and verify they fail**

Run: `uv run pytest tests/integration/test_research_graph_parity.py -v`  
Expected: FAIL because `LangGraphResearchRunner` does not exist.

- [ ] **Step 3: Implement StateGraph nodes through `assess_evidence` only**

Define a JSON-safe TypedDict state mirroring the legacy graph. Compile with the existing SQLite checkpointer, use `thread_id` supplied by the service, and migrate `initialize`, `plan`, `retrieve`, and `assess_evidence`. Route with pure functions; inject runtime planner/executor through the runner instance rather than state.

- [ ] **Step 4: Add compatibility selector and run regression tests**

Keep `GraphAgentRunner` as default while a service configuration switch selects the new runner for test fixtures. Prove legacy/default CLI output remains unchanged.

- [ ] **Step 5: Commit**

```bash
git add src/researchflow/agent src/researchflow/application tests/integration
git commit -m "feat: checkpoint research planning graph"
```

### Task 8: Migrate remaining research nodes, verify sources, then retire the legacy graph

**Files:**
- Modify: `src/researchflow/agent/research_graph.py`
- Modify: `src/researchflow/agent/graph.py`
- Modify: `src/researchflow/application/service.py`
- Modify: `src/researchflow/cli.py`
- Modify: `docs/architecture.md`
- Modify: `README.md`
- Test: `tests/integration/test_research_graph_parity.py`
- Test: `tests/integration/test_research_graph_checkpoint.py`
- Test: `tests/integration/test_agent_workflow.py`

**Interfaces:**
- Consumes: Task 7 partial research graph and all Task 4/6 event and recovery contracts.
- Produces: a complete LangGraph research execution path implementing `read_sources`, `replan`, `synthesize`, `verify`, `save`, and `finish`.

- [ ] **Step 1: Write failing full-graph parity and recovery tests**

```python
def test_checkpoint_resume_does_not_repeat_completed_fetch_or_save(...): ...
def test_zero_valid_sources_ends_insufficient_evidence_not_completed_answer(...): ...
def test_llm_fallback_event_and_markdown_match_legacy_contract(...): ...
```

- [ ] **Step 2: Run tests and verify they fail**

Run: `uv run pytest tests/integration/test_research_graph_parity.py tests/integration/test_research_graph_checkpoint.py -v`  
Expected: FAIL because remaining nodes are not implemented.

- [ ] **Step 3: Implement remaining LangGraph nodes one node group at a time**

Migrate `read_sources` and replan first, then `synthesize`/`verify`, then `save`/`finish`. Preserve candidate/read limits, rejected-source reasons, `SUFFICIENT`/`PARTIAL`/`INSUFFICIENT`, LLM fallback status, source-id verification, and write-once behavior. Checkpoint after each node group; emit standard events only after state updates.

- [ ] **Step 4: Flip service/CLI default to `LangGraphResearchRunner` and remove legacy-only path**

Remove the old loop only after parity tests cover all nodes. Retain public `GraphAgentRunner` compatibility only if third-party imports or tests require it; otherwise replace it with a thin deprecated alias and update documentation to state that research execution is now LangGraph-backed.

- [ ] **Step 5: Run full verification and commit**

Run: `uv sync --frozen --extra llm --extra ui`  
Expected: environment resolves without lock changes.

Run: `uv lock --check && uv run pytest && uv run ruff check . && uv run ruff format --check .`  
Expected: all pass.

Run: `npm --prefix frontend test -- --run && npm --prefix frontend run build`  
Expected: all pass.

```bash
git add src docs README.md tests pyproject.toml uv.lock frontend
git commit -m "feat: run research workflow with LangGraph"
```

### Task 9: Perform controlled local acceptance and document operation

**Files:**
- Modify: `README.md`
- Modify: `docs/architecture.md`
- Create: `docs/local-web-console.md`
- Test: `tests/integration/test_web_console_acceptance.py`

**Interfaces:**
- Consumes: complete API, UI, events, durable interrupts, and LangGraph research runner.
- Produces: documented local startup and an automated acceptance fixture where external dependencies are faked.

- [ ] **Step 1: Write failing acceptance scenarios**

```python
def test_console_acceptance_covers_session_restart_interrupt_resume_and_evidence_views(...): ...
def test_console_never_marks_empty_evidence_as_grounded_answer(...): ...
```

- [ ] **Step 2: Run acceptance tests and verify they fail**

Run: `uv run pytest tests/integration/test_web_console_acceptance.py -v`  
Expected: FAIL until the complete API/UI lifecycle is wired.

- [ ] **Step 3: Document local operation and limitations**

Document `uv sync --frozen --extra ui --extra llm`, `uv run researchflow serve`, UI development startup, loopback scope, output directories, Trace inspection, required Tavily/LLM configuration, how to resume a paused run, and grounded-evidence limitations. Do not document unimplemented cloud or multi-user behavior.

- [ ] **Step 4: Run acceptance and manual smoke checks**

Run: `uv run pytest tests/integration/test_web_console_acceptance.py -v`  
Expected: PASS.

Run: `uv run researchflow serve --help`  
Expected: loopback host/port options are documented.

Manual smoke: start `serve`, open the local workbench, create a Web-enabled session, observe separate source categories, refresh during a run, trigger a blank-input interrupt, resume it, then inspect the persisted report and Trace.

- [ ] **Step 5: Commit**

```bash
git add README.md docs tests/integration
git commit -m "docs: document local web console operation"
```

## Plan Self-Review

- Spec coverage: Tasks 1–3 implement the local service/API/UI boundary; Tasks 4–5 add normalized SSE events; Task 6 implements durable explicit human-in-the-loop; Tasks 7–8 migrate research execution to LangGraph; Task 9 validates and documents local operation. Every requirement in the approved design has an owning task.
- Type consistency: `RunSnapshot`, `StartTurn`, `ResumeTurn`, `SqliteRunStore`, `ResearchService`, and `EventEnvelope` are introduced before later tasks consume them. `session_id` remains the only thread ID source across service, API, and graph.
- Review focus coverage: refresh/reconnect is Task 5; idempotency is Tasks 1, 2, and 6; evidence display separation is Tasks 1 and 4; restart/resume is Task 6; content/secret rendering is Tasks 3 and 5.
- Proportion: code blocks only pin tests and APIs; implementation bodies are intentionally left to task executors.
