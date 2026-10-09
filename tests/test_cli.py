"""Tests for the command-line interface."""

import json
import sqlite3
from pathlib import Path

import pytest
from typer.testing import CliRunner

from researchflow import __version__, cli
from researchflow.cli import app
from researchflow.domain import AgentState, AgentStatus
from researchflow.llm import (
    BaseLLMProvider,
    LLMConfigurationError,
    LLMDependencyError,
    LLMResponse,
    TokenUsage,
)
from researchflow.tools import ToolFailure
from researchflow.tools.offline import FileSystemNoteStore

runner = CliRunner()


class FakeCheckProvider(BaseLLMProvider):
    def __init__(self) -> None:
        self.requests = []
        self._responses = iter(
            [
                {
                    "steps": [
                        {
                            "step_id": "search",
                            "description": "Search",
                            "tool_name": "search_documents",
                        },
                        {
                            "step_id": "read",
                            "description": "Read",
                            "tool_name": "read_document",
                        },
                        {"step_id": "summarize", "description": "Summarize"},
                        {
                            "step_id": "save",
                            "description": "Save",
                            "tool_name": "save_note",
                        },
                    ]
                },
                {
                    "action_type": "search",
                    "tool_name": "search_documents",
                    "arguments": {"query": "tool calling", "limit": 5},
                },
                {
                    "summary": "The supplied source is available.",
                    "source_paths": ["llm-check-source.md"],
                },
            ]
        )

    def complete(self, request):
        self.requests.append(request)
        return LLMResponse(
            content=json.dumps(next(self._responses)),
            usage=TokenUsage(input_tokens=1, output_tokens=1),
            finish_reason="stop",
        )


class FakeAgentProvider(BaseLLMProvider):
    def __init__(self) -> None:
        self._responses = iter(
            [
                {
                    "steps": [
                        {
                            "step_id": "search",
                            "description": "Search local documents",
                            "tool_name": "search_documents",
                        },
                        {
                            "step_id": "read",
                            "description": "Read local documents",
                            "tool_name": "read_document",
                        },
                        {"step_id": "summarize", "description": "Summarize"},
                        {
                            "step_id": "save",
                            "description": "Save report",
                            "tool_name": "save_note",
                        },
                    ]
                },
                {
                    "action_type": "search",
                    "tool_name": "search_documents",
                    "arguments": {"query": "工具调用", "limit": 5},
                },
                {
                    "action_type": "read",
                    "tool_name": "read_document",
                    "arguments": {"path": "agent.md"},
                },
                {
                    "action_type": "summarize",
                    "tool_name": None,
                    "arguments": {},
                },
                {"summary": "工具调用需要参数校验。", "source_paths": ["agent.md"]},
                {"action_type": "save", "tool_name": "save_note", "arguments": {}},
                {"action_type": "finish", "tool_name": None, "arguments": {}},
            ]
        )

    def complete(self, request):
        return LLMResponse(
            content=json.dumps(next(self._responses)),
            usage=TokenUsage(input_tokens=2, output_tokens=1),
        )


@pytest.fixture
def cli_paths(tmp_path: Path) -> tuple[Path, Path]:
    documents = tmp_path / "documents"
    output = tmp_path / "output"
    documents.mkdir()
    (documents / "agent.md").write_text(
        "# Agent 工具调用\n\n工具调用需要参数校验和安全执行。\n\n"
        "不相关的完整文档正文不应出现在默认输出。",
        encoding="utf-8",
    )
    return documents, output


def invoke_run(
    documents: Path,
    output: Path,
    *extra: str,
    query: str = "工具调用",
):
    return runner.invoke(
        app,
        [
            "run",
            query,
            "--documents-dir",
            str(documents),
            "--output-dir",
            str(output),
            *extra,
        ],
    )


def test_help_is_available() -> None:
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "Research technical topics" in result.stdout


def test_run_help_lists_modes_and_options() -> None:
    result = runner.invoke(app, ["run", "--help"])

    assert result.exit_code == 0
    assert "[query]" in result.stdout
    assert "--documents-dir" in result.stdout
    assert "--output-dir" in result.stdout
    assert "--max-steps" in result.stdout
    assert "--agent-mode" in result.stdout
    assert "--allowed-domain" in result.stdout
    assert "--verbose" in result.stdout


def test_serve_help_documents_loopback_options() -> None:
    result = runner.invoke(app, ["serve", "--help"])

    assert result.exit_code == 0
    assert "--host" in result.stdout
    assert "--port" in result.stdout
    assert "--enable-web" in result.stdout
    assert "--agent-mode" in result.stdout


def test_version_is_available() -> None:
    result = runner.invoke(app, ["--version"])

    assert result.exit_code == 0
    assert result.stdout.strip() == f"researchflow {__version__}"


def test_run_records_completed_application_run(cli_paths) -> None:
    documents, output = cli_paths

    result = invoke_run(documents, output)

    assert result.exit_code == 0
    with sqlite3.connect(output / "sessions" / "checkpoints.sqlite3") as database:
        rows = database.execute("SELECT status FROM researchflow_runs").fetchall()
    assert rows == [("completed",)]


def test_llm_check_uses_structured_fake_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RESEARCHFLOW_LLM_API_KEY", "test-key")
    provider = FakeCheckProvider()
    monkeypatch.setattr(cli, "_create_llm_provider", lambda _config: provider)

    result = runner.invoke(app, ["llm-check"])

    assert result.exit_code == 0
    assert "planner: success=True" in result.stdout
    assert "selector: success=True" in result.stdout
    assert "summarizer: success=True" in result.stdout
    assert "success=True" in result.stdout
    assert "fallback=False" in result.stdout
    assert "finish_reason=stop" in result.stdout
    assert "llm-check-source.md" in provider.requests[2].user_prompt


def test_cli_loads_local_llm_configuration_before_llm_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    for name in (
        "RESEARCHFLOW_LLM_API_KEY",
        "RESEARCHFLOW_LLM_BASE_URL",
        "RESEARCHFLOW_LLM_MODEL",
    ):
        monkeypatch.delenv(name, raising=False)
    (tmp_path / ".env.local").write_text(
        "\n".join(
            (
                "RESEARCHFLOW_LLM_API_KEY=test-key",
                "RESEARCHFLOW_LLM_BASE_URL=https://llm.example/v1",
                "RESEARCHFLOW_LLM_MODEL=test-model",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    provider = FakeCheckProvider()
    monkeypatch.setattr(cli, "_create_llm_provider", lambda _config: provider)

    result = runner.invoke(app, ["llm-check"])

    assert result.exit_code == 0
    assert "planner: success=True" in result.stdout


def test_llm_mode_reports_missing_optional_client_instead_of_fallback(
    cli_paths, monkeypatch: pytest.MonkeyPatch
) -> None:
    documents, output = cli_paths
    monkeypatch.setenv("RESEARCHFLOW_LLM_API_KEY", "test-key")
    monkeypatch.setattr(
        cli,
        "_create_llm_provider",
        lambda _config: (_ for _ in ()).throw(
            LLMDependencyError("Install it with `uv sync --extra llm`")
        ),
    )

    result = invoke_run(documents, output, "--agent-mode", "llm")

    assert result.exit_code == 2
    assert "uv sync --extra llm" in result.stderr
    assert "已回退到规则模式" not in result.stdout


def test_llm_agent_mode_runs_with_fake_provider_and_records_decisions(
    cli_paths, monkeypatch: pytest.MonkeyPatch
) -> None:
    documents, output = cli_paths
    monkeypatch.setenv("RESEARCHFLOW_LLM_API_KEY", "test-key")
    monkeypatch.setattr(
        cli, "_create_llm_provider", lambda _config: FakeAgentProvider()
    )

    result = invoke_run(documents, output, "--agent-mode", "llm")

    assert result.exit_code == 0
    trace = next((output / "traces").glob("*.jsonl"))
    records = [
        json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines()
    ]
    assert any(record.get("event_type") == "llm_decision" for record in records)


def test_llm_agent_mode_falls_back_to_rule_mode_without_configuration(
    cli_paths, monkeypatch: pytest.MonkeyPatch
) -> None:
    documents, output = cli_paths
    monkeypatch.delenv("RESEARCHFLOW_LLM_API_KEY", raising=False)

    def missing_provider(_config):
        raise LLMConfigurationError("LLM API key is not configured")

    monkeypatch.setattr(cli, "_create_llm_provider", missing_provider)

    result = invoke_run(documents, output, "--agent-mode", "llm")

    assert result.exit_code == 0
    assert "LLM 配置不完整，已回退到规则模式" in result.stdout
    trace = next((output / "traces").glob("*.jsonl"))
    assert "llm_decision" not in trace.read_text(encoding="utf-8")


def test_chat_llm_mode_falls_back_without_configuration(
    cli_paths, monkeypatch: pytest.MonkeyPatch
) -> None:
    documents, output = cli_paths
    monkeypatch.delenv("RESEARCHFLOW_LLM_API_KEY", raising=False)
    monkeypatch.setattr(
        cli,
        "_create_llm_provider",
        lambda _config: (_ for _ in ()).throw(
            LLMConfigurationError("LLM API key is not configured")
        ),
    )

    result = runner.invoke(
        app,
        [
            "chat",
            "工具调用",
            "--session-id",
            "fallback-chat",
            "--agent-mode",
            "llm",
            "--documents-dir",
            str(documents),
            "--output-dir",
            str(output),
        ],
    )

    assert result.exit_code == 0
    assert "LLM 配置不完整，已回退到规则模式" in result.stdout


def test_web_mode_requires_explicit_search_configuration(
    cli_paths, monkeypatch: pytest.MonkeyPatch
) -> None:
    documents, output = cli_paths
    monkeypatch.delenv("RESEARCHFLOW_SEARCH_API_KEY", raising=False)
    monkeypatch.setattr(
        "researchflow.tools.web.provider.load_local_secrets", lambda: None
    )
    monkeypatch.setattr(cli, "load_local_secrets", lambda: None)

    result = invoke_run(documents, output, "--enable-web")

    assert result.exit_code == 2
    assert "SEARCH_API_KEY" in result.stderr


def test_evaluate_outputs_all_retrievers_and_writes_utf8_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "metrics" / "evaluation.json"
    monkeypatch.setattr(
        cli,
        "evaluate_retriever",
        lambda *_args, **_kwargs: {
            "embedding": {"zh": {"queries": 2}, "overall": {"queries": 2}}
        },
    )

    result = runner.invoke(
        app,
        ["evaluate", "--retriever", "all", "--output", str(output)],
    )

    assert result.exit_code == 0
    saved = json.loads(output.read_text(encoding="utf-8"))
    assert set(saved) == {"embedding"}
    assert saved["embedding"]["zh"]["queries"] == 2
    assert '"overall"' in result.stdout


def test_evaluate_rejects_unknown_retriever() -> None:
    result = runner.invoke(app, ["evaluate", "--retriever", "unknown"])

    assert result.exit_code == 2
    assert "--retriever" in result.output


def test_direct_mode_displays_run_sections_and_real_paths(cli_paths) -> None:
    documents, output = cli_paths

    result = invoke_run(documents, output, query="  工具调用  ")

    assert result.exit_code == 0
    assert "研究计划" in result.stdout
    assert "1. [COMPLETED]" in result.stdout
    assert "工具状态" in result.stdout
    assert "[OK] search_documents" in result.stdout
    assert "[OK] read_document" in result.stdout
    assert "[OK] save_note" in result.stdout
    assert "最终摘要" in result.stdout
    assert "# 研究报告" in result.stdout
    note = next((output / "notes").glob("*.md"))
    trace = next((output / "traces").glob("*.jsonl"))
    assert f"报告: {note.resolve()}" in result.stdout
    assert f"Trace: {trace.resolve()}" in result.stdout
    tool_names = [
        json.loads(line)["tool_name"]
        for line in trace.read_text(encoding="utf-8").splitlines()
    ]
    assert tool_names == [
        "search_documents",
        "read_document",
        "save_note",
    ]
    assert "不相关的完整文档正文" not in result.stdout
    assert "run_id:" not in result.stdout


def test_interactive_mode_runs_once(cli_paths) -> None:
    documents, output = cli_paths

    result = runner.invoke(
        app,
        [
            "run",
            "--documents-dir",
            str(documents),
            "--output-dir",
            str(output),
        ],
        input="工具调用\n",
    )

    assert result.exit_code == 0
    assert "请输入研究问题:" in result.stdout
    assert result.stdout.count("研究计划") == 1
    assert list((output / "notes").glob("*.md"))


def test_interactive_mode_rejects_blank_input(cli_paths) -> None:
    documents, output = cli_paths

    result = runner.invoke(
        app,
        ["run", "--documents-dir", str(documents), "--output-dir", str(output)],
        input="   \n",
    )

    assert result.exit_code == 2
    assert "错误: 研究问题不能为空" in result.stderr
    assert "Traceback" not in result.output


def test_direct_mode_rejects_blank_query(cli_paths) -> None:
    documents, output = cli_paths

    result = invoke_run(documents, output, query="   ")

    assert result.exit_code == 2
    assert "错误: 研究问题不能为空" in result.stderr
    assert "Traceback" not in result.output


def test_missing_documents_directory_is_input_error(tmp_path: Path) -> None:
    result = invoke_run(tmp_path / "missing", tmp_path / "output")

    assert result.exit_code == 2
    assert "错误: 文档目录不存在" in result.stderr
    assert "Traceback" not in result.output


def test_document_path_must_be_directory(tmp_path: Path) -> None:
    document_file = tmp_path / "document.md"
    document_file.write_text("内容", encoding="utf-8")

    result = invoke_run(document_file, tmp_path / "output")

    assert result.exit_code == 2
    assert "错误: 文档路径不是目录" in result.stderr


def test_output_directory_creation_failure_is_input_error(
    cli_paths, monkeypatch: pytest.MonkeyPatch
) -> None:
    documents, output = cli_paths
    original_mkdir = Path.mkdir

    def fail_output(self: Path, *args, **kwargs) -> None:
        if self == output:
            raise OSError("read only")
        original_mkdir(self, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", fail_output)
    result = invoke_run(documents, output)

    assert result.exit_code == 2
    assert "错误: 输出目录无法创建" in result.stderr
    assert "Traceback" not in result.output


def test_max_steps_must_be_positive(cli_paths) -> None:
    documents, output = cli_paths

    result = invoke_run(documents, output, "--max-steps", "0")

    assert result.exit_code == 2
    assert "错误: --max-steps 必须大于或等于 1" in result.stderr


def test_graph_orchestrator_runs_offline(cli_paths) -> None:
    documents, output = cli_paths

    result = invoke_run(documents, output, "--orchestrator", "graph")

    assert result.exit_code == 0
    assert "最终摘要" in result.stdout
    assert any((output / "traces").glob("*.jsonl"))


def test_default_run_uses_checkpointed_langgraph_research(cli_paths) -> None:
    documents, output = cli_paths

    result = invoke_run(documents, output)

    assert result.exit_code == 0
    with sqlite3.connect(output / "sessions" / "checkpoints.sqlite3") as database:
        checkpoint_count = database.execute(
            "SELECT COUNT(*) FROM checkpoints"
        ).fetchone()[0]
        payload = database.execute("SELECT payload FROM researchflow_runs").fetchone()[
            0
        ]
    assert checkpoint_count > 0
    snapshot = json.loads(payload)
    assert snapshot["evidence"]["candidates"][0]["source_id"] == "agent.md"
    assert snapshot["evidence"]["read_sources"][0]["source_id"] == "agent.md"


def test_graph_run_projects_evidence_into_persisted_application_snapshot(
    cli_paths,
) -> None:
    documents, output = cli_paths

    result = invoke_run(documents, output, "--orchestrator", "graph")

    assert result.exit_code == 0
    with sqlite3.connect(output / "sessions" / "checkpoints.sqlite3") as database:
        payload = database.execute("SELECT payload FROM researchflow_runs").fetchone()[
            0
        ]
        event_types = [
            json.loads(row[0])["type"]
            for row in database.execute(
                "SELECT payload FROM researchflow_events ORDER BY event_id"
            )
        ]
    snapshot = json.loads(payload)
    assert snapshot["evidence"]["candidates"][0]["source_id"] == "agent.md"
    assert snapshot["evidence"]["read_sources"][0]["source_id"] == "agent.md"
    assert "source_read" in event_types
    assert event_types.count("run_completed") == 1


def test_session_runner_projects_research_events_into_session_result(
    cli_paths, monkeypatch: pytest.MonkeyPatch
) -> None:
    documents, output = cli_paths

    def workflow(*_args, **kwargs):
        event_sink = kwargs["event_sink"]
        event_sink(
            "candidate_selected",
            {
                "source_id": "https://example.test/news",
                "title": "官方新闻",
                "kind": "web",
            },
        )
        event_sink(
            "source_read",
            {
                "source_id": "https://example.test/news",
                "title": "官方新闻",
                "kind": "web",
            },
        )
        event_sink(
            "evidence_assessed",
            {
                "status": "sufficient",
                "gaps": [],
                "policy": "overview",
                "accepted_source_count": 1,
                "required_source_count": 1,
                "official_complete_source_id": None,
            },
        )
        event_sink("generation_status", {"mode": "llm_grounded"})
        return (
            AgentState(
                run_id="session-run",
                query="问题",
                status=AgentStatus.COMPLETED,
                final_answer="基于官方新闻的回答",
            ),
            object(),
        )

    monkeypatch.setattr(cli, "_run_workflow", workflow)
    session_runner = cli._session_runner(
        documents,
        output,
        agent_mode="rule",
        enable_web=True,
        allowed_domains=(),
        llm_overrides=None,
    )
    result = session_runner.chat("问题", session_id="session-evidence")
    session_runner.close()

    assert result.research_result is not None
    assert result.research_result.read_sources[0].title == "官方新闻"
    assert result.research_result.evidence_status == "sufficient"
    assert result.research_result.generation_mode == "llm_grounded"


def test_invalid_max_steps_does_not_start_interactive_prompt() -> None:
    result = runner.invoke(app, ["run", "--max-steps", "0"])

    assert result.exit_code == 2
    assert "错误: --max-steps 必须大于或等于 1" in result.stderr
    assert "请输入研究问题" not in result.stdout


def test_verbose_displays_run_id_order_status_and_duration(cli_paths) -> None:
    documents, output = cli_paths

    result = invoke_run(documents, output, "--verbose")

    assert result.exit_code == 0
    assert "详细信息" in result.stdout
    assert "run_id:" in result.stdout
    assert "1. [OK] search_documents" in result.stdout
    assert "2. [OK] read_document" in result.stdout
    assert "3. [OK] save_note" in result.stdout
    assert " ms" in result.stdout
    assert '"content"' not in result.stdout


def test_max_step_agent_failure_renders_report_then_exits_one(cli_paths) -> None:
    documents, output = cli_paths

    result = invoke_run(documents, output, "--max-steps", "1")

    assert result.exit_code == 1
    assert "最终摘要" in result.stdout
    assert "最大步骤限制" in result.stdout
    assert "报告未保存" in result.stdout
    assert "Trace:" in result.stdout
    assert "Traceback" not in result.output


def test_save_failure_omits_nonexistent_report_path(
    cli_paths, monkeypatch: pytest.MonkeyPatch
) -> None:
    documents, output = cli_paths

    def fail_save(self, path: str, content: str, overwrite: bool = False) -> str:
        raise ToolFailure("磁盘已满", error_type="write_failed")

    monkeypatch.setattr(FileSystemNoteStore, "save", fail_save)
    result = invoke_run(documents, output)

    assert result.exit_code == 1
    assert "报告未保存: 磁盘已满" in result.stdout
    assert "\n报告: " not in result.stdout
    trace = next((output / "traces").glob("*.jsonl"))
    assert f"Trace: {trace.resolve()}" in result.stdout
    assert not (output / "notes").exists()


@pytest.mark.parametrize("verbose", [False, True])
def test_unexpected_runtime_error_has_no_traceback(
    cli_paths, monkeypatch: pytest.MonkeyPatch, verbose: bool
) -> None:
    documents, output = cli_paths

    def fail_run(*args, **kwargs):
        raise RuntimeError("simulated failure")

    monkeypatch.setattr(cli, "_run_workflow", fail_run)
    extra = ("--verbose",) if verbose else ()
    result = invoke_run(documents, output, *extra)

    assert result.exit_code == 1
    assert "错误: 研究流程运行失败" in result.stderr
    assert ("RuntimeError: simulated failure" in result.stderr) is verbose
    assert "Traceback" not in result.output
