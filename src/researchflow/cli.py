"""Command-line interface for ResearchFlow Agent."""

from pathlib import Path
from typing import Annotated
from uuid import uuid4

import typer

from researchflow import __version__
from researchflow.agent import (
    AgentRunner,
    ExtractiveSummarizer,
    RulePlanner,
    StateSelector,
)
from researchflow.domain import AgentStatus
from researchflow.execution import JsonlTraceRecorder, ToolExecutor
from researchflow.tools import ToolContext, ToolRegistry
from researchflow.tools.offline import create_offline_tools

app = typer.Typer(
    name="researchflow",
    help="Research technical topics with local documents and offline tools.",
    no_args_is_help=True,
)


def version_callback(value: bool) -> None:
    """Print the package version and exit."""
    if value:
        typer.echo(f"researchflow {__version__}")
        raise typer.Exit


@app.callback()
def main(
    version: Annotated[
        bool | None,
        typer.Option(
            "--version",
            callback=version_callback,
            is_eager=True,
            help="Show the version and exit.",
        ),
    ] = None,
) -> None:
    """Research technical topics with local documents and offline tools."""


@app.command("run")
def run_agent(
    query: Annotated[str, typer.Argument(help="Research question to investigate.")],
    documents_dir: Annotated[
        Path,
        typer.Option("--documents-dir", help="Directory containing local documents."),
    ] = Path("examples/documents"),
    output_dir: Annotated[
        Path,
        typer.Option("--output-dir", help="Directory for notes and traces."),
    ] = Path("output"),
    max_steps: Annotated[
        int,
        typer.Option("--max-steps", min=1, help="Maximum number of agent actions."),
    ] = 10,
) -> None:
    """Run the offline rule-driven research workflow."""
    if not documents_dir.is_dir():
        raise typer.BadParameter(
            "documents directory does not exist or is not a directory",
            param_hint="--documents-dir",
        )
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise typer.BadParameter(
            "output directory could not be created",
            param_hint="--output-dir",
        ) from exc

    context = ToolContext(
        working_directory=documents_dir,
        output_directory=output_dir,
        run_id=uuid4().hex,
    )
    registry = ToolRegistry()
    for tool in create_offline_tools(context):
        registry.register(tool)
    runner = AgentRunner(
        RulePlanner(),
        StateSelector(),
        ExtractiveSummarizer(),
        ToolExecutor(registry, JsonlTraceRecorder()),
        max_steps=max_steps,
    )
    state = runner.run(query, context)
    typer.echo(state.final_answer or "研究流程未生成报告。")
    if state.status is AgentStatus.FAILED:
        raise typer.Exit(code=1)
