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

## License

MIT
