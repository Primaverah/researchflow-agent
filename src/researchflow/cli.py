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
from researchflow.domain import AgentState, AgentStatus, ToolResult
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


def _input_error(message: str) -> None:
    typer.echo(f"错误: {message}", err=True)
    raise typer.Exit(code=2)


def _resolve_query(query: str | None) -> str:
    if query is None:
        try:
            query = typer.prompt("请输入研究问题")
        except (EOFError, typer.Abort):
            _input_error("研究问题不能为空")
    cleaned = query.strip()
    if not cleaned:
        _input_error("研究问题不能为空")
    return cleaned


def _create_context(documents_dir: Path, output_dir: Path) -> ToolContext:
    if not documents_dir.exists():
        _input_error("文档目录不存在")
    if not documents_dir.is_dir():
        _input_error("文档路径不是目录")
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        _input_error("输出目录无法创建")
    if not output_dir.is_dir():
        _input_error("输出路径不是目录")
    return ToolContext(
        working_directory=documents_dir,
        output_directory=output_dir,
        run_id=uuid4().hex,
    )


def _run_workflow(
    query: str,
    documents_dir: Path,
    output_dir: Path,
    max_steps: int,
) -> tuple[AgentState, ToolContext]:
    context = _create_context(documents_dir, output_dir)
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
    return runner.run(query, context), context


def _latest_result(state: AgentState, tool_name: str) -> ToolResult | None:
    return next(
        (
            result
            for result in reversed(state.tool_results)
            if result.tool_name == tool_name
        ),
        None,
    )


def _render_state(state: AgentState, context: ToolContext, *, verbose: bool) -> None:
    typer.echo("研究计划")
    if state.plan is None:
        typer.echo("未生成研究计划")
    else:
        for index, step in enumerate(state.plan.steps, start=1):
            typer.echo(f"{index}. [{step.status.value.upper()}] {step.description}")

    typer.echo("\n工具状态")
    if not state.tool_results:
        typer.echo("未执行工具")
    for result in state.tool_results:
        if result.success:
            typer.echo(f"[OK] {result.tool_name}")
        else:
            message = result.error_message or "未知错误"
            typer.echo(f"[FAILED] {result.tool_name}: {message}")

    typer.echo("\n最终摘要")
    typer.echo(state.final_answer or "研究流程未生成摘要。")

    save_result = _latest_result(state, "save_note")
    typer.echo("\n输出文件")
    if save_result is not None and save_result.success:
        relative_path = str(save_result.output["path"])
        typer.echo(f"报告: {(context.output_directory / relative_path).resolve()}")
    elif save_result is not None:
        typer.echo(f"报告未保存: {save_result.error_message or '未知错误'}")
    else:
        typer.echo("报告未保存")
    trace_path = context.output_directory / "traces" / f"{context.run_id}.jsonl"
    typer.echo(f"Trace: {trace_path.resolve() if trace_path.is_file() else '未生成'}")

    if verbose:
        typer.echo("\n详细信息")
        typer.echo(f"run_id: {state.run_id}")
        if not state.traces:
            typer.echo("未产生工具 Trace")
        for index, trace in enumerate(state.traces, start=1):
            status = "OK" if trace.status.value == "succeeded" else "FAILED"
            detail = f"{index}. [{status}] {trace.tool_name} {trace.duration_ms:.2f} ms"
            if trace.error_type is not None:
                detail += f" — {trace.error_type}: {trace.error_message}"
            typer.echo(detail)


@app.command("run")
def run_agent(
    query: Annotated[
        str | None,
        typer.Argument(help="Research question to investigate."),
    ] = None,
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
        typer.Option("--max-steps", help="Maximum number of agent actions."),
    ] = 10,
    verbose: Annotated[
        bool,
        typer.Option("--verbose/--no-verbose", help="Show execution details."),
    ] = False,
) -> None:
    """Run one offline rule-driven research workflow."""
    if max_steps < 1:
        _input_error("--max-steps 必须大于或等于 1")
    resolved_query = _resolve_query(query)
    try:
        state, context = _run_workflow(
            resolved_query,
            documents_dir,
            output_dir,
            max_steps,
        )
        _render_state(state, context, verbose=verbose)
    except typer.Exit:
        raise
    except Exception as exc:
        message = "研究流程运行失败"
        if verbose:
            message = f"{message} ({type(exc).__name__}: {exc})"
        typer.echo(f"错误: {message}", err=True)
        raise typer.Exit(code=1) from None

    save_result = _latest_result(state, "save_note")
    if state.status is AgentStatus.FAILED or (
        save_result is not None and not save_result.success
    ):
        raise typer.Exit(code=1)
