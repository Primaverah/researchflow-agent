# ResearchFlow Agent

A tool-using AI agent for technical research and literature analysis.

## Planned Features

- [ ] Task planning
- [ ] Paper search
- [ ] Document reading
- [ ] Tool selection
- [ ] Citation-based answering
- [ ] Execution tracing
- [ ] Agent evaluation
- [ ] MCP integration
- [ ] Agent Skills support

## Architecture

The project will consist of:

- Agent orchestration
- Tool calling
- MCP server integration
- Reusable Agent Skills
- Evaluation and observability

## Status

🚧 This project is currently under development.

## Development

ResearchFlow Agent targets Python 3.11 and uses
[uv](https://docs.astral.sh/uv/) for dependency management.

```bash
uv sync
uv run researchflow --help
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

### Offline tools

The offline tools use separate document and output roots and can be registered in
the shared tool registry:

```python
from pathlib import Path

from researchflow.domain import ToolCall
from researchflow.tools import ToolContext, ToolRegistry
from researchflow.tools.offline import create_offline_tools

notes = Path("notes")
notes.mkdir(exist_ok=True)
context = ToolContext(
    working_directory=Path("examples/documents"),
    output_directory=notes,
    run_id="example-run",
)
registry = ToolRegistry()
for tool in create_offline_tools(context):
    registry.register(tool)

result = registry.get("search_documents").execute(
    ToolCall(
        call_id="search-1",
        tool_name="search_documents",
        arguments={"query": "工具调用", "limit": 5},
    ),
    context,
)
print(result.output)
```

### Traced tool execution

`ToolExecutor` is the single synchronous entry point for registered tool calls. It
returns the existing `ToolResult` and records success, validation failures, missing
tools, and execution failures through a trace recorder.

```python
from researchflow.execution import JsonlTraceRecorder, ToolExecutor

executor = ToolExecutor(registry, JsonlTraceRecorder())
result = executor.execute(
    ToolCall(
        call_id="search-2",
        tool_name="search_documents",
        arguments={"query": "Agent 安全", "limit": 3},
    ),
    context,
)
```

The JSONL recorder appends one independently parseable JSON object per line to
`<output_directory>/traces/<run_id>.jsonl`. For example:

```json
{"trace_id":"...","run_id":"example-run","call_id":"search-2","tool_name":"search_documents","arguments":{"query":"Agent 安全","limit":3},"status":"succeeded","started_at":"2026-09-20T10:00:00Z","duration_ms":1.25,"error_type":null,"error_message":null}
```

### Rule-driven agent

The current agent is synchronous, offline, and rule-driven. It creates a fixed
research plan, selects the local search/read/save tools, extracts relevant source
text, saves a Markdown report, and records every real tool call. It does not use an
LLM or any network service.

```bash
uv run researchflow run "tool calling" \
  --documents-dir examples/documents \
  --output-dir output \
  --max-steps 10
```

Each run writes its report and execution traces below the selected output root:

```text
output/notes/<run_id>.md
output/traces/<run_id>.jsonl
```

## License

MIT
