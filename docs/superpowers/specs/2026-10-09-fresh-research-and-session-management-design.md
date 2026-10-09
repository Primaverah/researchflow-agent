# Fresh Research Completion and Session Management Design

## Goal

Make time-sensitive research truthful and recoverable: a run must never claim completion after the graph was stopped by its safety budget, and a current-year question must not be satisfied by an older result. Give the single local user readable, editable session names and safe deletion without changing LangGraph checkpoint identities.

## Scope and constraints

- The persisted `session_id` remains the only LangGraph `thread_id`; a display name never changes checkpoint identity, conversation history, or resume behavior.
- Raw search snippets and page bodies remain out of application snapshots and the browser UI.
- A source is grounded evidence only after successful read, relevance review, and the policy applicable to the question.
- The solution must retain bounded execution, SSRF safeguards, and the existing SQLite shared-database model.
- Deletion is exact-session only and must remove application metadata as well as checkpoint data.

## Research completion

### Root cause

The current node budget is 10. A no-retry graph uses all ten nodes through `finish`; a failed initial retrieval adds `assess -> replan -> retrieve -> assess`, so the graph hits its maximum at `synthesize`. The summary may already exist but verification and save do not run. The application currently derives its terminal state only from evidence status, which can expose this bounded stop as `completed`.

### Design

1. Treat `max_steps` as a graph-node safety bound and calculate a configured default that permits the fixed terminal suffix (`synthesize`, `verify`, `save`, `finish`) after the configured replan budget. The default is `10 + (3 * max_replans)`, which is 13 for one replan. The session runner reads this resolved setting instead of hard-coding 10.
2. Carry `end_reason`, `node_steps`, and `max_steps` as safe turn metadata from the research graph into `SessionResearchResult` and `RunSnapshot`.
3. When a real `max_steps` stop still occurs (for example, a user selects a lower CLI bound), return a non-completed terminal status and a plain explanation. A completed badge is valid only when graph `end_reason` is `completed`.
4. Preserve the existing insufficient-evidence result separately: it is not a successful factual answer, even if the graph reaches its normal finish node.

## Freshness-aware retrieval

### Intent model

`今年`, `本年度`, `当前`, `最新`, `currently`, and `latest` are freshness markers. The contextualizer resolves the target year from the execution clock; in this environment, the shown question becomes a 2026 query. `哪些` is treated as list intent alongside `名单` and `获奖者`.

### Candidate policy

1. Produce a compact current-year query containing the resolved year and the requested subject/facets. The original standalone query remains traceable.
2. Rank candidates stably before reading them: target-year title/summary evidence first, then source-quality/relevance signals, then original provider order. Stable ordering keeps ties deterministic.
3. For current-list questions, reject a successfully-read source when its title and content do not contain the target year or another explicit current-year marker. Record `stale_for_current_query` as its rejection reason.
4. Current-list policy requires two distinct valid current-year sources unless a configured official domain provides a complete target-year list. A candidate title, search snippet, or old-year page never independently satisfies this policy.
5. If no compliant evidence is obtained, report the precise evidence gap. Do not generate a factual list from model knowledge or stale sources.

## Session display names and deletion

### Data model

Add application-owned session metadata in the shared SQLite file:

```
researchflow_session_metadata(
  session_id TEXT PRIMARY KEY,
  display_name TEXT NOT NULL,
  updated_at TEXT NOT NULL
)
```

The first persisted turn creates metadata with a fallback display name derived from the first question. A manual rename updates only `display_name` and timestamp.

### API

- `GET /api/sessions` returns records `{session_id, display_name}` sorted by most-recent activity.
- `PATCH /api/sessions/{session_id}` accepts `{display_name}` after trimmed non-empty validation and returns the updated record.
- `DELETE /api/sessions/{session_id}` deletes application runs, events, start/resume idempotency entries, session metadata, catalog record, and exact `thread_id` checkpoint/writes rows. It returns `204`; an unknown session returns `404`.
- `GET /api/sessions/{session_id}` continues to return the immutable ID and historical runs; unknown/deleted sessions return `404`.

### UI

- Sidebar uses `display_name`, with a compact immutable ID in a tooltip/subtitle.
- The active session offers an inline rename action and a destructive delete action.
- Delete requires an explicit browser confirmation, then selects a newly generated local session and clears the displayed run/history.
- Failed rename/delete calls retain the existing selection and show an accessible error message.

## Testing and verification

- Unit-test a retried graph with one replan and default budget: it reaches `finish` without `max_steps`.
- Unit-test a deliberately insufficient budget: it records a non-completed terminal state and readable budget diagnostics.
- Unit-test freshness ranking and stale-source rejection for a 2026 current-list query containing 2022 and 2026 candidates.
- Test the exact `今年……哪些` classification and two-source/current-year evidence threshold.
- API, store, and frontend tests cover rename persistence, validation, and full deletion across application rows and LangGraph checkpoint tables.
- Run full Python tests, Ruff checks/format validation, frontend tests, and production build.
