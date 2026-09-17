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

## License

MIT
