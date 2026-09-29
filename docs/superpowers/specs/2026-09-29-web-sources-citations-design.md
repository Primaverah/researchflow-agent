# Web Sources and Trusted Citations Design

## Goal

Add an optional, bounded web-research path to the existing Agent Loop. Default
execution remains fully offline. When explicitly enabled, the agent may search
with Tavily, fetch safe ordinary HTML pages, and cite only pages that were both
returned by search and read successfully.

## Scope and constraints

- Network access is opt-in through the CLI; the default path stays unchanged.
- Tavily credentials are read only from `RESEARCHFLOW_SEARCH_API_KEY`.
- No browser automation, JavaScript rendering, PDF extraction, vector database,
  MCP, or long-term memory is added.
- The implementation uses the standard library and injected protocols; it adds
  no runtime dependency.
- Tests never contact Tavily or the public internet.

## Components

### Search provider

`WebSearchProvider` exposes a small synchronous `search(query, limit)` method.
`TavilySearchProvider` is a lazy REST adapter that reads its API key from the
environment only when web mode is requested. Provider errors are normalized to
safe tool failures and never include keys or request headers.

### Sources and web tools

`WebSearchResult` represents a search candidate. `WebSource` is the canonical
successfully-read source model and stores title, canonical URL, search summary,
UTC access time, and extracted body text.

`web_search` invokes the provider and returns candidates. `fetch_url` accepts
only a candidate URL from the preceding successful web-search result. It fetches
ordinary HTML, extracts readable text, and returns a `WebSource` only after all
validations pass.

### Safe HTTP boundary

The injected HTTP client handles requests with a bounded timeout and response
size. Before every request, including each redirect destination, URL validation
requires HTTP or HTTPS and rejects localhost plus loopback, private, link-local,
unspecified, multicast, and reserved resolved IP addresses. Non-text content,
PDFs, oversized bodies, unsupported schemes, and redirect targets that fail the
same validation are rejected. JavaScript-rendered content is out of scope.

## Agent and report flow

Offline mode keeps the current local search, read, summarize, save workflow.
Web mode adds web search and candidate fetches to the bounded selector. Tool
failure, timeout, empty results, or failed page reads allow the selector to
continue with other candidates and then summarize what was successfully read.

The summarizer receives only successful local documents and `WebSource`
instances. It emits a stable source list in first-read order with numbered
citations. Web entries use `[N] title — URL`; a source never appears unless it
was searched and fetched successfully. With no readable sources, the report
contains the existing safe empty-source result.

## CLI and observability

`researchflow run --enable-web` constructs the optional provider and registers
the web tools. Missing configuration is reported as an input/configuration error
when web mode is explicitly requested; it never changes offline operation.
`--agent-mode rule|llm` remains independent of the web switch.

Tool and decision traces retain their existing redaction. Web traces may contain
safe URL/query metadata and failure categories, but never API keys, headers, or
full response bodies. README and `.env.example` document the opt-in variable,
limits, and citation behavior.

## Tests and acceptance

Fake search providers and injected mock HTTP responses cover Tavily request
configuration, empty/error responses, redirects, unsafe host addresses,
non-HTML and oversized content, successful extraction, source filtering,
stable citations, and no-network CLI fallback. Existing offline Agent and CLI
tests remain unchanged and pass alongside full pytest and Ruff validation.
