# Web Sources and Trusted Citations Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add opt-in Tavily search, safe HTML retrieval, and trusted numbered web citations to the bounded Agent Loop.

**Architecture:** A focused `tools.web` package owns provider, HTTP validation, source models, and tool adapters behind injected protocols. A web-aware rule planner/selector and source-aware summarizer extend the existing runner only when `--enable-web` is selected; offline composition is unchanged.

**Tech Stack:** Python 3.11 standard library (`urllib`, `ipaddress`, `html.parser`), Pydantic, Typer, pytest, Ruff.

**Spec:** `docs/superpowers/specs/2026-09-29-web-sources-citations-design.md`

## Global Constraints

- Keep default execution fully offline; network is opt-in through `--enable-web`.
- Read Tavily credentials only from `RESEARCHFLOW_SEARCH_API_KEY`.
- Add no runtime dependency and do not access the real network in tests.
- Accept only HTTP/HTTPS HTML; reject localhost, non-public IPs, redirects to unsafe targets, oversized responses, and non-text responses.
- Cite only sources returned by successful `web_search` and successful `fetch_url` calls.
- Never write API keys, headers, prompts, full HTTP bodies, or credentials to traces.
- Do not add browser automation, PDF parsing, vector databases, MCP, or long-term memory.

## Review Focus

- Redirects to loopback/private IP addresses must be rejected after redirect resolution; Task 2 test mocks a public initial URL redirecting to `127.0.0.1`.
- A URL supplied directly to `fetch_url` without a matching successful search candidate must not be fetched; Task 4 selector test constructs state with an unsearched URL.
- An HTML response exceeding the byte limit must fail without producing body output; Task 2 test streams an oversized mock response.
- A web candidate that fails retrieval must never appear in the final numbered sources; Task 4 integration test mixes one successful and one failed candidate.
- Explicit web mode with absent API configuration must return a clear input error while ordinary offline runs still work; Task 5 CLI tests cover both paths.

---

## File Structure

- `src/researchflow/tools/web/models.py`: validated search, source, and tool I/O models.
- `src/researchflow/tools/web/provider.py`: `WebSearchProvider`, Tavily adapter, and safe configuration error.
- `src/researchflow/tools/web/http.py`: injected HTTP client and URL/response safety boundary.
- `src/researchflow/tools/web/tools.py`: `web_search` and `fetch_url` tool adapters.
- `src/researchflow/tools/web/factory.py`: opt-in web tool construction.
- `src/researchflow/agent/web.py`: web-aware planner/selector and source collection helpers.
- `src/researchflow/agent/llm_components.py`: validation for web actions when LLM mode is combined with web mode.
- `src/researchflow/agent/summarizer.py`: stable numbering across successful local and web sources.
- `src/researchflow/cli.py`: `--enable-web` composition and explicit configuration error.
- `tests/unit/web/*`, `tests/unit/agent/*`, `tests/integration/*`, `tests/test_cli.py`: fake provider/mock HTTP and workflow coverage.

### Task 1: Search provider and canonical web models

**Files:**
- Create: `src/researchflow/tools/web/models.py`
- Create: `src/researchflow/tools/web/provider.py`
- Create: `src/researchflow/tools/web/__init__.py`
- Test: `tests/unit/web/test_provider.py`

**Interfaces:**
- Produces `WebSearchProvider.search(query: str, limit: int) -> list[WebSearchResult]`.
- Produces `TavilySearchProvider.from_environment()` and validated `WebSource`.
- Consumed by web tool adapters in Task 3.

- [ ] Write failing Fake Provider and mocked-`urlopen` tests for Tavily request payload, missing key, normalized remote failure, and UTF-8 result mapping.
- [ ] Run `uv run pytest tests/unit/web/test_provider.py` and verify the expected missing-module failure.
- [ ] Implement the Pydantic models, provider protocol, environment-only Tavily REST adapter, and non-secret error handling.
- [ ] Re-run the targeted test and verify it passes.
- [ ] Commit `feat: add web search provider models`.

### Task 2: Safe HTML fetching boundary

**Files:**
- Create: `src/researchflow/tools/web/http.py`
- Test: `tests/unit/web/test_http.py`

**Interfaces:**
- Consumes `WebSearchResult` URLs.
- Produces `SafeHttpClient.fetch(url: str) -> HtmlPage` with title/body/content metadata.
- Consumed by `FetchUrlTool` in Task 3.

- [ ] Write failing mock-client tests for accepted public HTML, unsupported schemes, localhost/private IPs, unsafe redirects, non-HTML content, and oversized bodies.
- [ ] Run `uv run pytest tests/unit/web/test_http.py` and verify the expected missing-module failure.
- [ ] Implement public-address validation using `urllib.parse`, DNS resolution, `ipaddress`, bounded reads, response content-type checks, redirect revalidation, and HTML text/title extraction.
- [ ] Re-run the targeted test and verify it passes.
- [ ] Commit `feat: add safe HTML fetch boundary`.

### Task 3: Web tools and registry composition

**Files:**
- Create: `src/researchflow/tools/web/tools.py`
- Create: `src/researchflow/tools/web/factory.py`
- Modify: `src/researchflow/tools/__init__.py`
- Test: `tests/unit/web/test_tools.py`

**Interfaces:**
- Produces `WebSearchTool(provider: WebSearchProvider)` named `web_search`.
- Produces `FetchUrlTool(client: SafeHttpClient)` named `fetch_url`.
- Produces `create_web_tools(provider, client) -> tuple[BaseTool, ...]`.

- [ ] Write failing tool tests using Fake Provider/HTTP for result serialization and tool-failure normalization.
- [ ] Run `uv run pytest tests/unit/web/test_tools.py` and verify the expected missing-module failure.
- [ ] Implement Pydantic schemas and BaseTool adapters; include only URL/query metadata in output and preserve `WebSource` fields in successful fetch results.
- [ ] Re-run the targeted test and verify it passes.
- [ ] Commit `feat: add bounded web tools`.

### Task 4: Web-aware Agent flow and trusted citations

**Files:**
- Create: `src/researchflow/agent/web.py`
- Modify: `src/researchflow/agent/runner.py`
- Modify: `src/researchflow/agent/summarizer.py`
- Modify: `src/researchflow/agent/models.py`
- Modify: `src/researchflow/agent/llm_components.py`
- Test: `tests/unit/agent/test_web_selector.py`
- Test: `tests/integration/test_web_agent_workflow.py`

**Interfaces:**
- Produces `WebRulePlanner.create_plan(query: str) -> ResearchPlan` and `WebStateSelector.select(state: AgentState) -> AgentAction`.
- Extends runner summarization to consume successful `ReadDocumentOutput` and successful `WebSource` outputs only.
- Produces deterministic `[N] title — URL` source lines in first successful-read order.

- [ ] Write failing selector/integration tests for local-plus-web ordering, no-result continuation, failed fetch exclusion, step bound, and stable web citation numbering.
- [ ] Run `uv run pytest tests/unit/agent/test_web_selector.py tests/integration/test_web_agent_workflow.py` and verify the expected failure.
- [ ] Implement bounded web search/fetch actions; constrain every fetch URL to a successful `web_search` candidate; extend LLM action validation for the same permitted actions; add runner dispatch and source-aware summarization without changing offline selector behavior.
- [ ] Re-run the targeted tests and verify they pass.
- [ ] Commit `feat: add trusted web citations to agent flow`.

### Task 5: CLI opt-in, documentation, and acceptance

**Files:**
- Modify: `src/researchflow/cli.py`
- Modify: `.env.example`
- Modify: `README.md`
- Modify: `tests/test_cli.py`
- Modify: `tests/unit/execution/test_recorder.py`

**Interfaces:**
- Adds `researchflow run --enable-web` while retaining `--agent-mode rule|llm`.
- Calls `TavilySearchProvider.from_environment()` only when web mode is enabled.

- [ ] Write failing CLI tests for missing web configuration, Fake Provider/Mock HTTP web execution, offline default behavior, and trace payloads excluding secrets/bodies.
- [ ] Run `uv run pytest tests/test_cli.py tests/unit/execution/test_recorder.py` and verify the expected failure.
- [ ] Implement CLI composition, explicit configuration error, UTF-8 documentation/example environment variable, and trace documentation.
- [ ] Run `uv lock --check`, `uv run pytest`, `uv run ruff check .`, `uv run ruff format --check .`, and offline `researchflow` help/run smoke tests; verify all pass.
- [ ] Commit `feat: add opt-in web sources and citations`.
