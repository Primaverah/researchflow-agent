# Web Evidence Quality Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prevent boilerplate-only webpages from becoming evidence, enforce intent-specific evidence quality, and make LLM-summary fallbacks visible in grounded reports.

**Architecture:** Preserve the existing safe HTTP and fixed graph boundaries. Replace flat HTML text collection with scored text blocks, reject boilerplate-only fetches before they become `WebSource`, and make graph relevance return an explicit rejection reason. Keep `summarize()` source-compatible while exposing a typed last-generation status consumed by `GraphAgentRunner` to render a safe report status and trace data.

**Tech Stack:** Python 3.11, standard-library `html.parser`, Pydantic v2, LangGraph-adjacent fixed graph, pytest, Ruff.

**Spec:** `docs/superpowers/specs/2026-10-07-web-evidence-quality-design.md`

## Global Constraints

- Do not weaken HTTP(S), SSRF, mixed-DNS, domain allowlist, or existing redirect rejection safeguards.
- Search snippets remain candidates, never fetched-body evidence.
- Do not add a parsing dependency; use the standard library already used by `SafeHttpClient`.
- Never emit API keys, authorization headers, raw prompts, model responses, or complete source content in diagnostics.
- LLM mode must attempt a grounded LLM summary before deterministic fallback.
- Preserve UTF-8 Markdown/JSONL and Windows-safe console output.

## Review Focus

- A page with no `<main>`/`<article>` but a `div.article-content` must retain prose, not reject it; Task 1 tests it.
- Navigation expressed only as links inside generic `<div>` containers must not dominate fallback text; Task 1 tests it.
- A page naming a person but describing an unrelated opinion/discussion must fail identity evidence; Task 2 tests it.
- A valid person page with Chinese biographical wording and no exact “是谁” phrase must pass; Task 2 tests it.
- An LLM provider error with an HTTP diagnostic must be visible without leaking credentials; Task 3 tests it.

---

### Task 1: Extract and validate web main content

**Files:**
- Modify: `src/researchflow/tools/web/http.py`
- Modify: `tests/unit/web/test_http.py`

**Interfaces:**
- Consumes: `SafeHttpClient.fetch(url: str) -> HtmlPage`
- Produces: `_TextExtractor.content: str` containing selected primary prose, or `WebFetchError(..., "web_low_quality_content")` when no usable prose block exists.

- [ ] **Step 1: Write failing HTML extraction tests**

Add fixtures with generic `div` navigation links plus `div.article-content` prose, and a navigation-only page. Assert the first returns only title/body prose and the second raises `WebFetchError` with `web_low_quality_content`.

- [ ] **Step 2: Run the focused tests to verify they fail**

Run: `uv run pytest tests/unit/web/test_http.py -q`

Expected: FAIL because generic content containers and boilerplate-only pages are currently flattened into fallback text.

- [ ] **Step 3: Implement block-based extraction in `http.py`**

Replace the fallback flat-text bucket with nested text blocks. Prefer semantic main/article/role-main blocks, then class/id markers `article`, `content`, `post`, `entry`, `story`, `detail`. For generic fallback blocks, exclude anchor-heavy/navigation/chrome text and select the highest-quality prose block using non-link text length and sentence punctuation. Reject content below the defined prose quality threshold using `web_low_quality_content`.

- [ ] **Step 4: Run focused extraction tests**

Run: `uv run pytest tests/unit/web/test_http.py -q`

Expected: PASS.

- [ ] **Step 5: Commit Task 1**

```powershell
git add src/researchflow/tools/web/http.py tests/unit/web/test_http.py
git commit -m "fix: extract primary web content"
```

### Task 2: Add explicit evidence-quality rejection reasons

**Files:**
- Modify: `src/researchflow/agent/graph.py`
- Modify: `tests/unit/agent/test_graph.py`

**Interfaces:**
- Consumes: `GraphAgentRunner._evidence_rejection_reason(query: str, title: str, content: str) -> str | None`
- Produces: accepted source only if the method returns `None`; rejected source strings formatted as `<locator> — <reason>` and retained in graph trace/report diagnostics.

- [ ] **Step 1: Write failing intent-quality tests**

Add tests where `"村上春树是谁"` with only navigation/discussion text is rejected as `missing_person_identity_evidence`, while content containing `"村上春树是日本小说家"` is accepted. Retain existing compiler/works/age tests and assert failures name a reason.

- [ ] **Step 2: Run focused graph tests to verify they fail**

Run: `uv run pytest tests/unit/agent/test_graph.py -q`

Expected: FAIL because `_is_relevant()` currently returns true for all generic/person queries and rejected successful reads omit a reason.

- [ ] **Step 3: Implement explicit evidence rejection in `graph.py`**

Implement `_evidence_rejection_reason()` and retain `_is_relevant()` as its boolean compatibility wrapper. Identify person-introduction queries with generic Chinese/English patterns, extract the subject, require subject occurrence plus biography evidence terms (`作家`, `演员`, `科学家`, `记者`, `出生`, `国籍`, `代表作`, and English equivalents), and require meaningful content length. Attach the returned reason to local and web rejected entries.

- [ ] **Step 4: Run focused graph tests**

Run: `uv run pytest tests/unit/agent/test_graph.py -q`

Expected: PASS.

- [ ] **Step 5: Commit Task 2**

```powershell
git add src/researchflow/agent/graph.py tests/unit/agent/test_graph.py
git commit -m "fix: require quality evidence for profile answers"
```

### Task 3: Surface safe LLM-summary outcome and fallback reason

**Files:**
- Modify: `src/researchflow/agent/summarizer.py`
- Modify: `src/researchflow/agent/llm_components.py`
- Modify: `src/researchflow/agent/graph.py`
- Modify: `tests/unit/agent/test_llm_components.py`
- Modify: `tests/unit/agent/test_graph.py`

**Interfaces:**
- Produces: `SummaryGenerationStatus(mode: Literal["llm_grounded", "extractive_fallback", "extractive"], fallback_reason: str | None, error_type: str | None)` exposed by `last_generation_status` after every `summarize()`.
- Consumes: `LLMSummarizer.last_decision` only as sanitized source metadata; Graph copies only mode/reason/error type into `AgentGraphState` and trace.

- [ ] **Step 1: Write failing summarizer and graph-output tests**

Add a fake successful LLM test asserting `last_generation_status.mode == "llm_grounded"`. Add a provider-error test asserting the LLM is invoked once, fallback output is used, and report contains `## 回答生成状态` plus `extractive_fallback` and `provider_error`, without the raw exception message. Add graph trace assertions for the same safe fields.

- [ ] **Step 2: Run the focused tests to verify they fail**

Run: `uv run pytest tests/unit/agent/test_llm_components.py tests/unit/agent/test_graph.py -q`

Expected: FAIL because summary generation state is not an interface and Markdown has no visible fallback section.

- [ ] **Step 3: Implement `SummaryGenerationStatus` and report rendering**

Define the Pydantic/DomainModel status in `summarizer.py`; `ExtractiveSummarizer.summarize()` sets mode `extractive`. `LLMSummarizer.summarize()` sets `llm_grounded` only after source/citation/language validation; its exception handler maps the already-sanitized decision fields to `extractive_fallback` before invoking the deterministic summarizer. In `GraphAgentRunner._synthesize()`, render a short “回答生成状态” section after the answer, persist safe status fields in graph state, and add them to `_record_graph()` without raw exception text.

- [ ] **Step 4: Run focused tests**

Run: `uv run pytest tests/unit/agent/test_llm_components.py tests/unit/agent/test_graph.py -q`

Expected: PASS.

- [ ] **Step 5: Commit Task 3**

```powershell
git add src/researchflow/agent/summarizer.py src/researchflow/agent/llm_components.py src/researchflow/agent/graph.py tests/unit/agent/test_llm_components.py tests/unit/agent/test_graph.py
git commit -m "feat: report LLM summary fallback status"
```

### Task 4: Verify full workflow and the real CLI path

**Files:**
- Modify: `docs/architecture.md`
- Test: existing full suite and real output under ignored `output/`

**Interfaces:**
- Consumes: Tasks 1–3 implementation.
- Produces: documentation matching source behavior and a recorded real-run result or explicit external failure limitation.

- [ ] **Step 1: Update architecture documentation**

Document primary-content selection, evidence-quality rejections, LLM-first generation, the exact Markdown fallback status, and the retained redirect limitation.

- [ ] **Step 2: Run all offline checks**

Run: `uv sync --frozen; uv lock --check; uv run pytest; uv run ruff check .; uv run ruff format --check .; uv run researchflow --help; uv run researchflow chat --help; uv run researchflow sessions --help`

Expected: all commands succeed.

- [ ] **Step 3: Run the real CLI acceptance command**

Run: `uv run researchflow chat "村上春树是谁" --session-id cscs-demo2 --enable-web --agent-mode llm --output-dir output/web-evidence-quality`

Inspect its Markdown and JSONL. Assert no navigation string appears in the answer, every displayed source was successfully read and accepted, and the report says either `llm_grounded` or the safe fallback mode/reason. If external search/LLM/network prevents this, report the JSONL/Markdown paths and exact safe failure instead of claiming acceptance.

- [ ] **Step 4: Commit Task 4**

```powershell
git add docs/architecture.md
git commit -m "docs: document web evidence quality behavior"
```

## Plan Self-Review

- Spec coverage: Tasks 1–3 map directly to primary-content extraction, evidence thresholds, and LLM-visible fallback; Task 4 covers trace/report acceptance and documentation.
- Type consistency: `SummaryGenerationStatus` is produced by both summarizers and read by `GraphAgentRunner`; raw exceptions never cross that boundary.
- Review focus coverage: generic content containers and anchor-heavy navigation are Task 1; biographical acceptance/rejection is Task 2; sanitized LLM provider failure is Task 3.
- Scope: redirects, DNS, key repair, and semantic-ranker replacement remain explicitly out of scope.
