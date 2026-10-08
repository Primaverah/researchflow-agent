"""Command-line interface for ResearchFlow Agent."""

import json
import sys
from pathlib import Path
from typing import Annotated
from uuid import uuid4

import typer
from pydantic import BaseModel

from researchflow import __version__
from researchflow.agent import (
    AgentOrchestrator,
    AgentRunner,
    ExtractiveSummarizer,
    GraphAgentRunner,
    LangGraphSessionRunner,
    LLMPlanner,
    LLMSelector,
    LLMSummarizer,
    RulePlanner,
    SessionCatalog,
    SessionResearchRequest,
    StateSelector,
    WebRulePlanner,
    WebStateSelector,
)
from researchflow.application import ResearchService, StartTurn
from researchflow.config import load_local_secrets, load_project_config
from researchflow.domain import AgentState, AgentStatus, ToolResult
from researchflow.evaluation import evaluate_retriever
from researchflow.execution import JsonlTraceRecorder, ToolExecutor
from researchflow.llm import (
    LLMConfig,
    LLMConfigurationError,
    LLMDependencyError,
    LLMError,
    OpenAICompatibleProvider,
    require_openai_dependency,
)
from researchflow.tools import ToolContext, ToolRegistry
from researchflow.tools.offline import ReadDocumentOutput, create_offline_tools
from researchflow.tools.web import (
    TavilySearchProvider,
    WebSearchConfigurationError,
    create_web_tools,
)

app = typer.Typer(
    name="researchflow",
    help="Research technical topics with local documents and offline tools.",
    no_args_is_help=True,
)
sessions_app = typer.Typer(help="Manage persisted chat sessions.")
app.add_typer(sessions_app, name="sessions")


class LLMCheckOutput(BaseModel):
    status: str


def terminal_safe_text(text: str, *, encoding: str | None = None) -> str:
    """Return terminal-safe text without changing UTF-8 artifacts on disk."""
    target = encoding or sys.stdout.encoding or "utf-8"
    return text.encode(target, errors="replace").decode(target)


def _create_llm_provider(
    config: LLMConfig | None = None,
) -> OpenAICompatibleProvider:
    require_openai_dependency()
    return OpenAICompatibleProvider(config or LLMConfig.resolve())


def version_callback(value: bool) -> None:
    """Print the package version and exit."""
    if value:
        typer.echo(f"researchflow {__version__}")
        raise typer.Exit


@app.callback()
def main(
    ctx: typer.Context,
    version: Annotated[
        bool | None,
        typer.Option(
            "--version",
            callback=version_callback,
            is_eager=True,
            help="Show the version and exit.",
        ),
    ] = None,
    llm_base_url: Annotated[
        str | None, typer.Option("--llm-base-url", help="LLM API base URL.")
    ] = None,
    llm_model: Annotated[
        str | None, typer.Option("--llm-model", help="LLM model name.")
    ] = None,
    llm_response_format: Annotated[
        str | None,
        typer.Option("--llm-response-format", help="LLM response format."),
    ] = None,
    llm_thinking: Annotated[
        bool | None,
        typer.Option("--llm-thinking/--no-llm-thinking", help="Enable LLM thinking."),
    ] = None,
    llm_timeout: Annotated[
        float | None, typer.Option("--llm-timeout", help="LLM timeout in seconds.")
    ] = None,
    llm_retries: Annotated[
        int | None, typer.Option("--llm-retries", help="LLM retry count.")
    ] = None,
    llm_planner_max_tokens: Annotated[
        int | None,
        typer.Option("--llm-planner-max-tokens", help="Planner output-token budget."),
    ] = None,
    llm_selector_max_tokens: Annotated[
        int | None,
        typer.Option("--llm-selector-max-tokens", help="Selector output-token budget."),
    ] = None,
    llm_summarizer_max_tokens: Annotated[
        int | None,
        typer.Option(
            "--llm-summarizer-max-tokens", help="Summarizer output-token budget."
        ),
    ] = None,
) -> None:
    """Research technical topics with local documents and offline tools."""
    load_local_secrets()
    ctx.ensure_object(dict)
    ctx.obj["llm_overrides"] = {
        "base_url": llm_base_url,
        "model": llm_model,
        "response_format": llm_response_format,
        "thinking": llm_thinking,
        "timeout": llm_timeout,
        "retries": llm_retries,
        "planner_max_tokens": llm_planner_max_tokens,
        "selector_max_tokens": llm_selector_max_tokens,
        "summarizer_max_tokens": llm_summarizer_max_tokens,
    }


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
    _validate_context_paths(documents_dir, output_dir)
    return ToolContext(
        working_directory=documents_dir,
        output_directory=output_dir,
        run_id=uuid4().hex,
    )


def _validate_context_paths(documents_dir: Path, output_dir: Path) -> None:
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


def _allow_llm_fallback() -> bool:
    agent_config = load_project_config().get("agent", {})
    return (
        isinstance(agent_config, dict)
        and agent_config.get("allow_fallback", True) is True
    )


def _graph_options() -> dict[str, int]:
    """Load bounded graph settings, leaving validation to the runner."""
    graph = load_project_config().get("graph", {})
    if not isinstance(graph, dict):
        return {}
    names = ("candidate_limit", "read_limit", "max_concurrency", "max_replans")
    return {
        name: value for name in names if isinstance((value := graph.get(name)), int)
    }


def _session_database(output_dir: Path) -> Path:
    return output_dir / "sessions" / "checkpoints.sqlite3"


def _session_runner(
    documents_dir: Path,
    output_dir: Path,
    *,
    agent_mode: str,
    enable_web: bool,
    allowed_domains: tuple[str, ...],
    llm_overrides: dict[str, object] | None,
) -> LangGraphSessionRunner:
    def research(request: SessionResearchRequest) -> str:
        state, _ = _run_workflow(
            request.standalone_query,
            documents_dir,
            output_dir,
            10,
            agent_mode,
            enable_web,
            allowed_domains,
            llm_overrides,
            "graph",
            answer_target=request.answer_target,
            answer_language=request.answer_language,
        )
        return state.final_answer or "未找到相关文档。"

    return LangGraphSessionRunner(
        _session_database(output_dir),
        research=lambda query: research(
            SessionResearchRequest(
                current_input=query,
                standalone_query=query,
                answer_target=query,
            )
        ),
        research_with_context=research,
    )


def _run_workflow(
    query: str,
    documents_dir: Path,
    output_dir: Path,
    max_steps: int,
    agent_mode: str = "rule",
    enable_web: bool = False,
    allowed_domains: tuple[str, ...] = (),
    llm_overrides: dict[str, object] | None = None,
    orchestrator: str = "loop",
    *,
    answer_target: str | None = None,
    answer_language: str = "",
) -> tuple[AgentState, ToolContext]:
    context = _create_context(documents_dir, output_dir)
    registry = ToolRegistry()
    for tool in create_offline_tools(context):
        registry.register(tool)
    planner = RulePlanner()
    selector = StateSelector()
    summarizer = ExtractiveSummarizer(allowed_domains=allowed_domains)
    if enable_web:
        provider = TavilySearchProvider.from_environment()
        for tool in create_web_tools(provider, allowed_domains=allowed_domains):
            registry.register(tool)
        planner = WebRulePlanner()
        selector = WebStateSelector(allowed_domains=allowed_domains)
    if agent_mode == "llm":
        try:
            llm_config = LLMConfig.resolve(cli=llm_overrides)
            provider = _create_llm_provider(llm_config)
        except LLMConfigurationError:
            if not _allow_llm_fallback():
                raise
            typer.echo("LLM 配置不完整，已回退到规则模式")
        except LLMDependencyError:
            raise
        else:
            model_name = getattr(provider, "model_name", "configured-llm")
            planner = LLMPlanner(
                provider,
                planner,
                model_name=model_name,
                max_output_tokens=llm_config.planner_max_tokens,
                response_format=llm_config.response_format,
                thinking=llm_config.thinking,
            )
            selector = LLMSelector(
                provider,
                selector,
                model_name=model_name,
                max_output_tokens=llm_config.selector_max_tokens,
                response_format=llm_config.response_format,
                thinking=llm_config.thinking,
            )
            summarizer = LLMSummarizer(
                provider,
                summarizer,
                model_name=model_name,
                max_output_tokens=llm_config.summarizer_max_tokens,
                response_format=llm_config.response_format,
                thinking=llm_config.thinking,
                allowed_domains=allowed_domains,
            )
    executor = ToolExecutor(registry, JsonlTraceRecorder())
    runner: AgentOrchestrator
    if orchestrator == "graph":
        runner = GraphAgentRunner(
            planner,
            selector,
            summarizer,
            executor,
            max_steps=max_steps,
            **_graph_options(),
        )
    else:
        runner = AgentRunner(
            planner, selector, summarizer, executor, max_steps=max_steps
        )
    if orchestrator == "graph":
        assert isinstance(runner, GraphAgentRunner)
        return (
            runner.run(
                query,
                context,
                answer_target=answer_target,
                answer_language=answer_language,
            ),
            context,
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
    typer.echo(terminal_safe_text(state.final_answer or "研究流程未生成摘要。"))

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
        for index, trace in enumerate(state.decision_traces, start=1):
            fallback = " fallback" if trace.fallback else ""
            typer.echo(
                f"LLM {index}. {trace.component} {trace.model} "
                f"{trace.input_tokens}/{trace.output_tokens} tokens{fallback}"
            )


@app.command("evaluate")
def evaluate(
    retriever: Annotated[
        str,
        typer.Option("--retriever", help="Retriever baseline: keyword, bm25, or all."),
    ] = "all",
    output: Annotated[
        Path | None,
        typer.Option("--output", help="Optional UTF-8 JSON result path."),
    ] = None,
    embedding_model: Annotated[
        str,
        typer.Option("--embedding-model", help="Local multilingual embedding model."),
    ] = "paraphrase-multilingual-MiniLM-L12-v2",
) -> None:
    """Evaluate keyword and BM25 retrieval on the bilingual baseline."""
    if retriever not in {"keyword", "bm25", "embedding", "hybrid", "all"}:
        _input_error("--retriever 必须为 keyword、bm25、embedding、hybrid 或 all")
    try:
        result = evaluate_retriever(retriever, model_name=embedding_model)
    except RuntimeError as exc:
        _input_error(str(exc))
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if output is not None:
        try:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(rendered + "\n", encoding="utf-8")
        except OSError:
            _input_error("评测结果无法保存")
    typer.echo(rendered)


@app.command("llm-check")
def llm_check(ctx: typer.Context) -> None:
    """Validate planner, selector, and summarizer structured LLM output."""
    try:
        config = LLMConfig.resolve(cli=ctx.obj["llm_overrides"])
        provider = _create_llm_provider(config)
        model_name = getattr(provider, "model_name", "configured-llm")
        planner = LLMPlanner(
            provider,
            RulePlanner(),
            model_name=model_name,
            max_output_tokens=config.planner_max_tokens,
            response_format=config.response_format,
            thinking=config.thinking,
        )
        plan = planner.create_plan("tool calling")
        state = AgentState(
            run_id="llm-check",
            query="tool calling",
            status=AgentStatus.RUNNING,
            plan=plan,
        )
        selector = LLMSelector(
            provider,
            StateSelector(),
            model_name=model_name,
            max_output_tokens=config.selector_max_tokens,
            response_format=config.response_format,
            thinking=config.thinking,
        )
        selector.select(state)
        summarizer = LLMSummarizer(
            provider,
            ExtractiveSummarizer(),
            model_name=model_name,
            max_output_tokens=config.summarizer_max_tokens,
            response_format=config.response_format,
            thinking=config.thinking,
        )
        summarizer.summarize(
            "tool calling",
            [
                ReadDocumentOutput(
                    path="llm-check-source.md",
                    title="LLM check source",
                    content="This is a verified source supplied to llm-check.",
                    char_count=48,
                )
            ],
        )
        components = (planner, selector, summarizer)
    except LLMError as exc:
        _input_error(str(exc))
    accepted = True
    for component in components:
        decision = component.last_decision
        if decision is None:
            accepted = False
            typer.echo(f"{component.__class__.__name__}: no decision")
            continue
        typer.echo(
            f"{decision.component}: success={decision.success} "
            f"fallback={decision.fallback} finish_reason={decision.finish_reason} "
            f"usage={decision.input_tokens}/{decision.output_tokens} "
            f"error_type={decision.error_type} diagnostic={decision.diagnostic}"
        )
        accepted = accepted and (
            decision.success
            and not decision.fallback
            and decision.finish_reason == "stop"
            and decision.input_tokens > 0
            and decision.output_tokens > 0
        )
    if not accepted:
        _input_error("LLM validation did not meet the acceptance criteria")


@app.command("run")
def run_agent(
    ctx: typer.Context,
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
    agent_mode: Annotated[
        str,
        typer.Option("--agent-mode", help="Agent mode: rule (default) or llm."),
    ] = "rule",
    orchestrator: Annotated[
        str,
        typer.Option("--orchestrator", help="Orchestrator: loop (default) or graph."),
    ] = "loop",
    enable_web: Annotated[
        bool,
        typer.Option("--enable-web", help="Enable optional Tavily web sources."),
    ] = False,
    allowed_domain: Annotated[
        list[str] | None,
        typer.Option(
            "--allowed-domain",
            help="Allowed web source domain; repeat to allow multiple domains.",
        ),
    ] = None,
    verbose: Annotated[
        bool,
        typer.Option("--verbose/--no-verbose", help="Show execution details."),
    ] = False,
) -> None:
    """Run one bounded local research workflow."""
    if max_steps < 1:
        _input_error("--max-steps 必须大于或等于 1")
    if agent_mode not in {"rule", "llm"}:
        _input_error("--agent-mode 必须为 rule 或 llm")
    if orchestrator not in {"loop", "graph"}:
        _input_error("--orchestrator 必须为 loop 或 graph")
    resolved_query = _resolve_query(query)
    try:
        _validate_context_paths(documents_dir, output_dir)
        execution: dict[str, tuple[AgentState, ToolContext]] = {}

        def workflow(message: str, _: str) -> AgentState:
            state, context = _run_workflow(
                message,
                documents_dir,
                output_dir,
                max_steps,
                agent_mode,
                enable_web,
                tuple(allowed_domain or ()),
                ctx.obj["llm_overrides"],
                orchestrator,
            )
            execution["result"] = (state, context)
            return state

        service = ResearchService(_session_database(output_dir), workflow=workflow)
        service.start_turn(
            StartTurn(
                session_id=f"run-{uuid4().hex}",
                message=resolved_query,
                idempotency_key=uuid4().hex,
            )
        )
        state, context = execution["result"]
    except WebSearchConfigurationError as exc:
        _input_error(str(exc))
    except LLMDependencyError as exc:
        _input_error(str(exc))
    except typer.Exit:
        raise
    except Exception as exc:
        message = "研究流程运行失败"
        if verbose:
            message = f"{message} ({type(exc).__name__}: {exc})"
        typer.echo(f"错误: {message}", err=True)
        raise typer.Exit(code=1) from None

    _render_state(state, context, verbose=verbose)

    save_result = _latest_result(state, "save_note")
    if state.status is AgentStatus.FAILED or (
        save_result is not None and not save_result.success
    ):
        raise typer.Exit(code=1)


@app.command("serve")
def serve(
    ctx: typer.Context,
    documents_dir: Annotated[Path, typer.Option("--documents-dir")] = Path(
        "examples/documents"
    ),
    output_dir: Annotated[Path, typer.Option("--output-dir")] = Path("output"),
    host: Annotated[str, typer.Option("--host")] = "127.0.0.1",
    port: Annotated[int, typer.Option("--port", min=1, max=65535)] = 8000,
    enable_web: Annotated[
        bool, typer.Option("--enable-web", help="Enable configured Web search.")
    ] = False,
    agent_mode: Annotated[
        str, typer.Option("--agent-mode", help="Research agent mode: rule or llm.")
    ] = "rule",
    allowed_domain: Annotated[
        list[str] | None,
        typer.Option(
            "--allowed-domain",
            help="Allowed web source domain; repeat to allow multiple.",
        ),
    ] = None,
) -> None:
    """Serve the local Web API on a loopback address."""
    if host not in {"127.0.0.1", "::1", "localhost"}:
        _input_error("--host 只能使用本地回环地址")
    if agent_mode not in {"rule", "llm"}:
        _input_error("--agent-mode 必须为 rule 或 llm")
    _validate_context_paths(documents_dir, output_dir)
    from researchflow.api.server import create_local_app

    def workflow(message: str, _: str) -> AgentState:
        state, _context = _run_workflow(
            message,
            documents_dir,
            output_dir,
            10,
            agent_mode,
            enable_web,
            tuple(allowed_domain or ()),
            ctx.obj["llm_overrides"],
            "graph",
        )
        return state

    import uvicorn

    uvicorn.run(
        create_local_app(_session_database(output_dir), workflow),
        host=host,
        port=port,
    )


@app.command("chat")
def chat(
    ctx: typer.Context,
    message: Annotated[str | None, typer.Argument(help="Message to research.")] = None,
    session_id: Annotated[str | None, typer.Option("--session-id")] = None,
    documents_dir: Annotated[Path, typer.Option("--documents-dir")] = Path(
        "examples/documents"
    ),
    output_dir: Annotated[Path, typer.Option("--output-dir")] = Path("output"),
    agent_mode: Annotated[
        str, typer.Option("--agent-mode", help="Agent mode: rule or llm.")
    ] = "rule",
    enable_web: Annotated[
        bool, typer.Option("--enable-web", help="Enable Tavily web search.")
    ] = False,
    allowed_domain: Annotated[
        list[str] | None,
        typer.Option("--allowed-domain", help="Allowed web domain; repeatable."),
    ] = None,
) -> None:
    """Run one persisted research chat turn."""
    if agent_mode not in {"rule", "llm"}:
        _input_error("--agent-mode 必须为 rule 或 llm")
    session_runner = _session_runner(
        documents_dir,
        output_dir,
        agent_mode=agent_mode,
        enable_web=enable_web,
        allowed_domains=tuple(allowed_domain or ()),
        llm_overrides=ctx.obj["llm_overrides"],
    )
    resolved_session = session_id or uuid4().hex
    try:
        content = message if message is not None else typer.prompt("消息")
        result = session_runner.chat(content, session_id=resolved_session)
        if result.interrupted:
            result = session_runner.resume(
                resolved_session, typer.prompt(result.prompt or "澄清")
            )
        typer.echo(f"session_id: {resolved_session}")
        typer.echo(terminal_safe_text(result.response))
    except WebSearchConfigurationError as exc:
        _input_error(str(exc))
    except LLMDependencyError as exc:
        _input_error(str(exc))
    finally:
        session_runner.close()


@sessions_app.command("list")
def list_sessions(
    output_dir: Annotated[Path, typer.Option("--output-dir")] = Path("output"),
) -> None:
    """List known session identifiers."""
    for record in SessionCatalog(_session_database(output_dir)).list():
        typer.echo(f"{record.session_id}\t{record.updated_at.isoformat()}")


@sessions_app.command("delete")
def delete_session(
    session_id: Annotated[str, typer.Argument(help="Exact session identifier.")],
    output_dir: Annotated[Path, typer.Option("--output-dir")] = Path("output"),
) -> None:
    """Delete one exact session and its checkpoints."""
    if not SessionCatalog(_session_database(output_dir)).delete(session_id):
        _input_error("会话不存在")
