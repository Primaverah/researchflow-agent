# Structured Query Understanding Implementation Plan

> For agentic workers: REQUIRED SUB-SKILL: use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Make follow-up and freshness-sensitive research queries retain their subject and required evidence facets before bounded, source-grounded retrieval.

**Architecture:** Add a JSON-safe QueryUnderstanding contract shared by the session runner and both graph executors. A deterministic builder is always available; the LLM rewriter may refine only ambiguous session context through schema-validated JSON and otherwise atomically falls back. Graph retrieval consumes the bounded query set and applies deterministic subject/facet/freshness gates before reads and evidence assessment.

**Tech Stack:** Python 3.12, Pydantic v2, LangGraph SQLite checkpointers, existing OpenAI-compatible structured provider, Typer, pytest, ruff.

**Spec:** docs/superpowers/specs/2026-10-10-structured-query-understanding-design.md

## Global Constraints

- Grounded answers may use only successfully read, accepted source bodies; candidates and snippets are never evidence.
- Do not disable SSRF, redirect, source-quality, citation, or freshness safeguards.
- The answer target is the current input; bounded history is solely for disambiguation.
- Query/candidate/read/replan limits are 3/10/5/1.
- Invalid, timeout, or provider-failed LLM output must atomically fall back and never log secrets, prompts, raw output, or bodies.
- Persist by existing thread_id / session_id; restart/retry must not repeat a completed rewriter call.
- Do not push or merge. Commit each completed task locally.

## Review Focus

- A follow-up with two prior subjects retains the latest resolved subject, rather than merging all old questions — Task 3 test.
- An explicit non-current year requires that exact year, rather than accepting arbitrary dated pages — Task 4 test.
- A title mentioning subjects but whose body lacks the requested facet is rejected — Task 4 test.
- Syntactically valid LLM JSON which drops a required subject is wholly rejected — Task 2 test.
- A restored SQLite checkpoint does not duplicate a completed rewriter call — Task 3 test.

---

## File Structure

- Create src/researchflow/agent/query_understanding.py: immutable contract, deterministic fallback builder, facet/query helpers.
- Modify session.py: persist/project the understanding and pass it to research.
- Modify llm_components.py, llm/config.py, and researchflow.toml: structured rewriter and budget.
- Modify graph.py, evidence_policy.py, and research_graph.py: bounded multi-query retrieval, gates, and replan.
- Modify cli.py and minimally service/models only for additive safe event projection.
- Test in tests/unit/agent/test_query_understanding.py, test_llm_components.py, test_session.py, test_graph.py, integration LLM/parity tests, and relevant CLI/application/SSE tests.

### Task 1: Deterministic Query Understanding Contract

**Files:**
- Create: src/researchflow/agent/query_understanding.py
- Modify: src/researchflow/agent/session.py
- Test: tests/unit/agent/test_query_understanding.py, tests/unit/agent/test_session.py

**Interfaces:**
- Produce QueryUnderstanding(DomainModel) with current_input, resolved_subject, standalone_query, intent, required_facets, search_queries, freshness: FreshnessIntent, decision_source, and safe fallback_reason.
- Produce build_deterministic_understanding(current_input: str, *, prior_subject: str, prior_query: str, prior_answer_target: str, answer_language: str) -> QueryUnderstanding.

- [ ] **Step 1: Write failing contract tests**

~~~python
def test_c_cpp_history_keeps_subjects_and_facet() -> None:
    result = build_deterministic_understanding(
        "它们的发展历史",
        prior_subject="C和C++",
        prior_query="C和C++有什么区别",
        prior_answer_target="C和C++有什么区别",
        answer_language="zh",
    )
    assert result.standalone_query == "C和C++的发展历史"
    assert result.required_facets == ["C", "C++", "history"]
    assert result.search_queries == ["C和C++的发展历史"]


def test_current_nobel_list_has_year_and_facets() -> None:
    result = build_deterministic_understanding(
        "今年的诺贝尔奖获得者有哪些",
        prior_subject="",
        prior_query="",
        prior_answer_target="",
        answer_language="zh",
    )
    assert str(result.freshness.target_year) in result.search_queries[0]
    assert {"nobel_prize", "winners", "target_year"} <= set(result.required_facets)
~~~

- [ ] **Step 2: Verify tests fail**

Run: uv run pytest tests/unit/agent/test_query_understanding.py -v  
Expected: FAIL because the module and builder do not exist.

- [ ] **Step 3: Implement the contract and deterministic builder**

Move/refactor existing non-LLM subject, pronoun, language, intent, and facet decisions out of LangGraphSessionRunner._contextualize. Preserve current input verbatim; replace pronouns only with resolved subject, never a string slice or full old-question concatenation. Apply build_retrieval_query, stable-dedupe 1–3 nonblank queries, and keep language-only behavior.

- [ ] **Step 4: Persist the result in session state**

Add a JSON-safe query_understanding field and derive existing compatibility fields from it. Do not alter the append reducer, checkpoint ID, or interrupt-before-research sequence.

- [ ] **Step 5: Verify**

Run: uv run pytest tests/unit/agent/test_query_understanding.py tests/unit/agent/test_session.py -v  
Expected: PASS, including compiler, person, game, language-only, isolation, and resume cases.

- [ ] **Step 6: Commit**

~~~powershell
git add src/researchflow/agent/query_understanding.py src/researchflow/agent/session.py tests/unit/agent/test_query_understanding.py tests/unit/agent/test_session.py
git commit -m "feat: add deterministic query understanding"
~~~

### Task 2: Schema-Constrained LLM Query Rewriter

**Files:**
- Modify: src/researchflow/agent/llm_components.py, src/researchflow/llm/config.py, researchflow.toml
- Test: tests/unit/agent/test_llm_components.py

**Interfaces:**
- Produce LLMQueryRewriter.rewrite(fallback: QueryUnderstanding, *, recent_user_inputs: list[str]) -> QueryUnderstanding.
- Use LLMQueryUnderstandingOutput containing only subject, standalone query, intent, facets, and query set.
- Add LLMConfig.rewriter_max_tokens: int = 400.

- [ ] **Step 1: Write failing rewriter tests**

~~~python
def test_rewriter_accepts_subject_preserving_output() -> None:
    result = rewriter.rewrite(fallback, recent_user_inputs=["C和C++有什么区别"])
    assert result.decision_source == "llm"
    assert "C" in result.standalone_query and "C++" in result.standalone_query


def test_rewriter_rejects_json_that_drops_subject() -> None:
    result = rewriter.rewrite(fallback, recent_user_inputs=["C和C++有什么区别"])
    assert result.decision_source == "deterministic_fallback"
    assert rewriter.last_decision.fallback_reason == "structured_output_invalid"
~~~

Also test invalid JSON, provider failure, and the configured 400-token budget without inspecting raw prompts or keys.

- [ ] **Step 2: Verify tests fail**

Run: uv run pytest tests/unit/agent/test_llm_components.py -k rewriter -v  
Expected: FAIL because LLMQueryRewriter does not exist.

- [ ] **Step 3: Implement the rewriter**

Use existing _LLMComponent._complete and LLMDecision. Prompt with fallback fields, bounded user messages, current date and schema; prohibit answer text, URLs, sources, and tools. Validate all fallback subject tokens and time/list constraints survive, facets are not dropped, and queries are 1–3 stable unique values. On any LLM, schema, or semantic error call _fallback("query_rewriter", exc) and return an object marked deterministic_fallback.

- [ ] **Step 4: Implement configuration**

Add rewriter_max_tokens = 400 under [llm]; resolve it through existing precedence and enforce minimum 1. Reuse existing provider timeout, retries, and API-key path.

- [ ] **Step 5: Verify**

Run: uv run pytest tests/unit/agent/test_llm_components.py tests/test_cli.py -k "rewriter or llm" -v  
Expected: PASS.

- [ ] **Step 6: Commit**

~~~powershell
git add src/researchflow/agent/llm_components.py src/researchflow/llm/config.py researchflow.toml tests/unit/agent/test_llm_components.py tests/test_cli.py
git commit -m "feat: add safe llm query rewriter"
~~~

### Task 3: Session and Workflow Integration

**Files:**
- Modify: src/researchflow/agent/session.py, src/researchflow/cli.py, src/researchflow/agent/research_graph.py
- Test: tests/unit/agent/test_session.py, tests/integration/test_llm_agent_workflow.py, tests/integration/test_research_graph_parity.py

**Interfaces:**
- Add SessionResearchRequest.understanding: QueryUnderstanding, retaining compatibility projections only where callers require them.
- Add optional rewrite_understanding: Callable[[QueryUnderstanding, list[str]], QueryUnderstanding] to LangGraphSessionRunner.
- Add optional understanding: QueryUnderstanding | None to direct and LangGraph graph-runner run methods.

- [ ] **Step 1: Write failing integration tests**

~~~python
def test_ambiguous_follow_up_rewrites_once_and_researches_standalone_query(tmp_path):
    runner = LangGraphSessionRunner(
        tmp_path / "checkpoints.sqlite3",
        research=lambda query: query,
        research_with_context=requests.append,
        rewrite_understanding=spy_rewriter,
    )
    runner.chat("C和C++有什么区别", session_id="languages")
    runner.chat("它们的发展历史", session_id="languages")
    assert spy_rewriter.calls == 1
    assert requests[-1].understanding.standalone_query == "C和C++的发展历史"


def test_recreated_runner_does_not_repeat_completed_rewriter(tmp_path):
    # Open the same SQLite database after the rewritten turn has checkpointed.
    assert spy_rewriter.calls == 1
~~~

Add an LLM integration fixture asserting a separate rewriter decision is recorded and the graph consumes its standalone query.

- [ ] **Step 2: Verify tests fail**

Run: uv run pytest tests/unit/agent/test_session.py tests/integration/test_llm_agent_workflow.py -k "rewriter or follow_up" -v  
Expected: FAIL because request, runner, and graph do not accept understanding.

- [ ] **Step 3: Wire conditional rewriting**

Implement needs_llm_rewrite(understanding): true for references, continuation/short contextual queries, unresolved subjects, or freshness ambiguity; false for clear self-contained first turns. Build it only for valid agent-mode llm configuration, persist the final result before research, and use it in both direct and session workflows. Reuse the same session thread ID; no tool/model call may occur before clarification interrupt().

- [ ] **Step 4: Verify checkpoint/parity behavior**

Run: uv run pytest tests/unit/agent/test_session.py tests/integration/test_llm_agent_workflow.py tests/integration/test_research_graph_parity.py -v  
Expected: PASS.

- [ ] **Step 5: Commit**

~~~powershell
git add src/researchflow/agent/session.py src/researchflow/cli.py src/researchflow/agent/research_graph.py tests/unit/agent/test_session.py tests/integration/test_llm_agent_workflow.py tests/integration/test_research_graph_parity.py
git commit -m "feat: apply query understanding before research"
~~~

### Task 4: Facet-Aware Retrieval and One Bounded Replan

**Files:**
- Modify: src/researchflow/agent/graph.py, src/researchflow/agent/evidence_policy.py, src/researchflow/agent/research_graph.py
- Test: tests/unit/agent/test_graph.py, tests/unit/agent/test_evidence_policy.py

**Interfaces:**
- Extend GraphCandidate with query_indexes: list[int], relevance_score: int, and optional rejection_reason.
- Store understanding-derived query set, facets, and subject in AgentGraphState.
- Produce _candidate_rejection_reason(candidate, understanding) -> str | None and _candidate_score(candidate, understanding) -> int.

- [ ] **Step 1: Write failing graph/evidence tests**

~~~python
def test_c_cpp_history_rejects_marx_and_gis_before_read(tmp_path):
    state = runner.run("C和C++的发展历史", context, understanding=c_cpp_history)
    assert "marx" not in fetched_urls and "gis" not in fetched_urls
    assert "missing_subject_coverage" in rejected_reasons


def test_current_nobel_list_rejects_microsoft_and_accepts_official_list(tmp_path):
    state = runner.run("今年的诺贝尔奖获得者有哪些", context, understanding=nobel_2026)
    assert state.evidence_status is EvidenceStatus.SUFFICIENT
    assert state.official_complete_source_id == "https://www.nobelprize.org/prizes/"


def test_two_sources_without_winners_facet_are_insufficient(tmp_path):
    assert state.evidence_status is EvidenceStatus.INSUFFICIENT
    assert "missing_required_facet:winners" in state.evidence_gaps
~~~

Also pin stable tie ordering, query/candidate/read limits of 3/10/5, one targeted replan, explicit year handling, and no summary with insufficient evidence.

- [ ] **Step 2: Verify tests fail**

Run: uv run pytest tests/unit/agent/test_graph.py tests/unit/agent/test_evidence_policy.py -k "history or nobel or facet or replan" -v  
Expected: FAIL because ranking is only year-sensitive and no facet candidate gate exists.

- [ ] **Step 3: Implement bounded multi-query search and candidate gates**

Search each understanding query via existing _parallel_calls, stable merge by type/locator, retain provenance, then score title/snippet using subject coverage, required facets, source-quality signals, and exact target year. Apply hard subject/facet/year rejection before reads; stable provider order breaks score ties. Record pre-read rejections separately from failed reads.

- [ ] **Step 4: Make body admission and evidence facet-aware**

Apply the same understanding-aware gate after a successful read. Pass only accepted bodies to evaluate_evidence_policy; extend it to use exact target year/facet coverage while retaining the configured official complete-list exception. Sufficient requires all facets; partial reports only supported facets; insufficient cannot run factual synthesis.

- [ ] **Step 5: Implement one targeted replan**

When no unread accepted candidates remain, derive at most three subject-preserving queries for unmet facets and replace the query set once. Account for search/read/assess/replan nodes in step budget so an allowed retry does not become an accidental maximum-step failure.

- [ ] **Step 6: Verify**

Run: uv run pytest tests/unit/agent/test_graph.py tests/unit/agent/test_evidence_policy.py tests/integration/test_research_graph_parity.py -v  
Expected: PASS.

- [ ] **Step 7: Commit**

~~~powershell
git add src/researchflow/agent/graph.py src/researchflow/agent/evidence_policy.py src/researchflow/agent/research_graph.py tests/unit/agent/test_graph.py tests/unit/agent/test_evidence_policy.py tests/integration/test_research_graph_parity.py
git commit -m "fix: gate retrieval by subject and evidence facets"
~~~

### Task 5: Safe Diagnostics, CLI Check, and Acceptance

**Files:**
- Modify: src/researchflow/application/service.py, src/researchflow/application/models.py only if an additive event projection needs typed fields
- Modify: src/researchflow/cli.py, docs/architecture.md
- Test: tests/application/test_service.py, tests/api/test_sse.py, tests/test_cli.py

**Interfaces:**
- Existing event types remain compatible; safe optional data fields report understanding, query count/list, score/reason, facet gaps, and replan status.
- researchflow llm-check validates the rewriter with planner, selector, and summarizer.

- [ ] **Step 1: Write failing trace/event tests**

~~~python
def test_trace_contains_safe_understanding_not_body(tmp_path):
    record = next(item for item in records if item["event_type"] == "graph_node")
    assert record["standalone_query"] == "C和C++的发展历史"
    assert "web page body" not in json.dumps(record, ensure_ascii=False)


def test_llm_check_includes_query_rewriter(runner):
    assert "LLMQueryRewriter" in runner.stdout
~~~

Also assert a pre-read rejection is visible as rejected but never read, and max-steps projects as failed.

- [ ] **Step 2: Verify tests fail**

Run: uv run pytest tests/application/test_service.py tests/api/test_sse.py tests/test_cli.py -k "rewriter or candidate or trace" -v  
Expected: FAIL because projection and LLM check do not expose the new component.

- [ ] **Step 3: Implement additive diagnostics and docs**

Record only spec-approved fields in JSONL and existing service/SSE projection; do not persist bodies, raw model output, prompts, keys, or headers. Add rewriter validation to llm-check without web calls. Update architecture documentation with the current-input → understanding → query set → evidence flow and fallback rule.

- [ ] **Step 4: Complete offline verification**

~~~powershell
uv sync --frozen --extra llm --extra ui
uv lock --check
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run researchflow --help
uv run researchflow chat --help
uv run researchflow sessions --help
~~~

Expected: every command exits 0.

- [ ] **Step 5: Controlled online acceptance when credentials and network are available**

Use the actual CLI in LLM mode for C/C++ then “它们的发展历史”, and for “今年的诺贝尔奖获得者有哪些”. Record elapsed time, rewriter decision/fallback, query set, candidate/read/rejection counts, evidence status, source URLs, end reason, and answer. If provider/network fails, preserve JSONL/Markdown and report the concrete limitation; do not claim online success.

- [ ] **Step 6: Commit**

~~~powershell
git add src/researchflow/application/service.py src/researchflow/application/models.py src/researchflow/cli.py docs/architecture.md tests/application/test_service.py tests/api/test_sse.py tests/test_cli.py
git commit -m "feat: expose structured retrieval diagnostics"
~~~
