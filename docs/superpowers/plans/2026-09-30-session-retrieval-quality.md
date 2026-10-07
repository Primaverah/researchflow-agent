# Session, Retrieval, and Grounded Answer Quality Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make persisted research chats correctly resolve follow-ups, use only relevant evidence, and answer the newest question without unsupported facts.

**Architecture:** \`LangGraphSessionRunner\` owns append-only session state and emits a typed current-turn research request. \`GraphAgentRunner\` owns query planning, source scoring, evidence assessment, verification, and diagnostic tracing. Summarizers consume only accepted evidence plus the newest answer target.

**Tech Stack:** Python 3.11, LangGraph, SQLite checkpoint saver, Pydantic, pytest, Ruff.

**Spec:** \`docs/superpowers/specs/2026-09-30-session-retrieval-quality-design.md\`

## Global Constraints

- Work only in the current repository and \`fix/session-retrieval-quality\`; do not create a worktree, push, or merge.
- Use one persistent SQLite database for all sessions; map \`session_id\` exactly to LangGraph \`thread_id\` and never use \`:memory:\`.
- Default research mode is grounded: no factual response without accepted evidence.
- Stable candidate cap is 10; stable selected-read cap is 5; traces contain no credentials.
- Production changes follow RED → GREEN: every new behavior has a test observed failing before its implementation.

## Review Focus

- Empty or pronoun-only first turns must clarify instead of guessing a subject; cover in Task 1.
- A compacted history must retain the resolved entity without replaying old question text; cover in Task 1.
- Multiple candidates with equal score must keep provider order while deduplicating URLs; cover in Task 2.
- A page with subject words but no required facet must be rejected after fetch; cover in Task 2.
- A partial evidence result must never claim an uncovered facet or cite a rejected source; cover in Task 3.

---

### Task 1: Append-only session context and checkpoint recovery

**Files:**
- Modify: \`src/researchflow/agent/session.py\`
- Modify: \`tests/unit/agent/test_session.py\`

**Interfaces:**
- Produces: session fields \`resolved_subject\`, \`intent\`, \`required_facets\`, \`standalone_query\`, \`turn_count\`, and \`interrupt_status\`.
- Produces: \`chat(message, *, session_id)\` and \`resume(session_id, answer)\` that submit a message delta under the same thread configuration.
- Consumes: later tasks receive a current-turn research request whose \`current_input\` is the answer target and whose \`standalone_query\` is the retrieval target.

- [ ] **Step 1: Write failing session tests**

Add literal, real-runner tests for: \`C和C++有什么区别\` then \`它们都用什么编译器\` yields a standalone query containing \`C\`, \`C++\`, and \`编译器\` but not \`区别们\`; the three Jackie Chan turns retain \`成龙\` and yield work/age facets; the game update follow-up retains \`重返未来1999\`; independent session ids do not share messages or subject; a second runner restores the first runner's state; and an interrupt/resume calls research once and records no response before resume.

- [ ] **Step 2: Run the new session tests and verify RED**

Run: \`uv run pytest tests/unit/agent/test_session.py -v\`

Expected: FAIL because the current state has no resolved subject, append reducer, or complete-token pronoun resolver.

- [ ] **Step 3: Implement typed session context and delta checkpointing**

In \`session.py\`, use an append reducer for messages; submit only the new user message from \`chat\`; persist user and assistant messages as deltas; add deterministic subject extraction, intent/facet derivation, and complete-token reference replacement. Ensure blank input interrupts before research and resume uses the unchanged thread id. Preserve compact summaries without using them as query text.

- [ ] **Step 4: Run the session tests and verify GREEN**

Run: \`uv run pytest tests/unit/agent/test_session.py -v\`

Expected: PASS.

- [ ] **Step 5: Commit**

\`\`\`bash
git add src/researchflow/agent/session.py tests/unit/agent/test_session.py
git commit -m "fix: persist structured session context"
\`\`\`

### Task 2: Relevant source selection and evidence status

**Files:**
- Modify: \`src/researchflow/agent/graph.py\`
- Modify: \`src/researchflow/agent/web.py\` if query planning must be shared by web mode
- Modify: \`tests/unit/agent/test_graph.py\`
- Modify: \`tests/unit/agent/test_web_selector.py\` if web selector API changes

**Interfaces:**
- Consumes: a query, intent, and required facets from Task 1 or direct \`GraphAgentRunner.run\` inputs.
- Produces: typed \`EvidenceStatus\` (\`SUFFICIENT\`, \`PARTIAL\`, \`INSUFFICIENT\`), scored \`GraphCandidate\` values, accepted source ids, gaps, and rejection reasons in \`AgentGraphState\`.
- Consumes: Task 3 receives only accepted documents/web sources and their source ids.

- [ ] **Step 1: Write failing graph tests**

Add fixtures proving simple requests make one query, complex facets make a small complementary query set, deduplication preserves ranking/provider order and caps at 10, reads cap at 5, and a fetched page mentioning the subject but lacking GCC/Clang/MSVC is rejected for a compiler request. Add cases for birth-date and representative-work facet evidence, plus no accepted source producing \`INSUFFICIENT\` after bounded replan.

- [ ] **Step 2: Run graph tests and verify RED**

Run: \`uv run pytest tests/unit/agent/test_graph.py tests/unit/agent/test_web_selector.py -v\`

Expected: FAIL because retrieval currently uses one query, lexical ordering, and permissive relevance.

- [ ] **Step 3: Implement planning, scoring, and evidence assessment**

Add typed intent/facet-aware query planning. Score title, snippet, content, subject coverage, facet coverage, and domain quality; stable-deduplicate without lexical resort; retain at most 10 and read at most 5. Record accept/reject reasons. Require actual facet terms before accepting a source, and bound replans before setting evidence status and gaps.

- [ ] **Step 4: Run graph tests and verify GREEN**

Run: \`uv run pytest tests/unit/agent/test_graph.py tests/unit/agent/test_web_selector.py -v\`

Expected: PASS.

- [ ] **Step 5: Commit**

\`\`\`bash
git add src/researchflow/agent/graph.py src/researchflow/agent/web.py tests/unit/agent/test_graph.py tests/unit/agent/test_web_selector.py
git commit -m "fix: gate retrieval on relevant evidence"
\`\`\`

### Task 3: Grounded answering, verification, and diagnostic traces

**Files:**
- Modify: \`src/researchflow/agent/summarizer.py\`
- Modify: \`src/researchflow/agent/llm_components.py\`
- Modify: \`src/researchflow/agent/graph.py\`
- Modify: \`src/researchflow/execution/recorder.py\`
- Modify: \`tests/unit/agent/test_summarizer.py\`
- Modify: \`tests/unit/agent/test_graph.py\`

**Interfaces:**
- Consumes: Task 2's evidence status, accepted sources, ids, gaps, current input, intent, and facets.
- Produces: a direct answer with source-id-bound factual claims, or an explicit evidence-insufficient / partial answer and structured diagnostic trace payload.

- [ ] **Step 1: Write failing grounded-answer tests**

Add tests that an empty evidence set returns an explicit evidence-insufficient answer without LLM output; a partial result names the unsupported age/work/update gap; a compiler answer returns compiler-supported extracts rather than C/C++ history; unknown or rejected source ids cause verification failure; and trace payloads include all specified data-flow fields but redact secret-shaped values.

- [ ] **Step 2: Run answer and graph tests and verify RED**

Run: \`uv run pytest tests/unit/agent/test_summarizer.py tests/unit/agent/test_graph.py -v\`

Expected: FAIL because summarization accepts empty evidence, verifier is empty, and graph traces omit required diagnostics.

- [ ] **Step 3: Implement evidence-gated synthesis and verification**

Make deterministic and LLM summarizers receive current input, accepted sources, source ids, facets, status, and gaps. In LLM prompts require a direct answer and valid source ids; reject unsupported claims/citations. Prevent synthesis on \`INSUFFICIENT\`, limit \`PARTIAL\` to supported facets, and add sanitized per-turn diagnostic trace records.

- [ ] **Step 4: Run answer and graph tests and verify GREEN**

Run: \`uv run pytest tests/unit/agent/test_summarizer.py tests/unit/agent/test_graph.py -v\`

Expected: PASS.

- [ ] **Step 5: Commit**

\`\`\`bash
git add src/researchflow/agent/summarizer.py src/researchflow/agent/llm_components.py src/researchflow/agent/graph.py src/researchflow/execution/recorder.py tests/unit/agent/test_summarizer.py tests/unit/agent/test_graph.py
git commit -m "fix: require grounded current-turn answers"
\`\`\`

### Task 4: Encoding regression coverage, CLI compatibility, and full offline verification

**Files:**
- Modify: \`src/researchflow/tools/web/http.py\` only if the failing test exposes a decoder gap
- Modify: \`tests/unit/web/test_http.py\`
- Modify: \`tests/test_cli.py\`
- Modify: \`README.md\` only if existing CLI documentation becomes inaccurate

**Interfaces:**
- Consumes: Tasks 1–3 public APIs without changing \`researchflow run\`, \`chat\`, \`sessions\`, or offline-mode commands.
- Produces: correct GBK/GB18030 text decoding and executable regression evidence for all supported CLI paths.

- [ ] **Step 1: Write failing encoding and CLI tests**

Add a GB18030 body/title fixture with no charset declaration and a conflicting/invalid header fixture that must still decode Chinese text. Add CLI tests for chat session reuse and sessions list/delete without exposing trace secrets.

- [ ] **Step 2: Run encoding and CLI tests and verify RED**

Run: \`uv run pytest tests/unit/web/test_http.py tests/test_cli.py -v\`

Expected: FAIL if fallback ordering or CLI compatibility is incomplete; otherwise record that the existing implementation already satisfies the decoder test and do not change production decoder code.

- [ ] **Step 3: Implement only the revealed decoder or compatibility correction**

Preserve safe HTML restrictions. Decode declared valid encodings first, then metadata, UTF-8, GB18030, and GBK, rejecting only truly unreadable content. Keep CLI flags and persistent shared SQLite semantics unchanged.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run: \`uv run pytest tests/unit/web/test_http.py tests/test_cli.py -v\`

Expected: PASS.

- [ ] **Step 5: Run required offline verification and commit**

Run:

\`\`\`bash
uv sync --frozen
uv lock --check
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run researchflow --help
uv run researchflow chat --help
uv run researchflow sessions --help
\`\`\`

Commit any Task 4 changes with \`test: cover encoding and chat regressions\`.

### Task 5: Controlled online acceptance verification

**Files:**
- No production files expected.

**Interfaces:**
- Consumes: all completed behavior and \`.env.local\` configuration only when present and usable.
- Produces: recorded command, elapsed time, trace-derived query/evidence/citation checks, or a precise external-configuration limitation.

- [ ] **Step 1: Inspect \`.env.local\` without printing secret values**

Verify only required variable presence and load it through the project's supported configuration path.

- [ ] **Step 2: Execute bounded online conversations**

Run C/C++ → compiler, Jackie Chan → works → age, game → latest update, and two isolated session ids. Capture elapsed time and sanitized traces.

- [ ] **Step 3: Validate acceptance evidence**

Confirm no LLM/search fallback; correct standalone query; at most 10 candidates and five reads; direct newest-turn answers; source ids from successful reads; and no grounded factual answer when source reads fail.

- [ ] **Step 4: Commit only if test-only acceptance artifacts are deliberately retained**

Do not commit \`.env.local\`, credentials, or generated runtime output.

