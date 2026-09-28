"""Tests for the command-line interface."""

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from researchflow import __version__, cli
from researchflow.cli import app
from researchflow.tools import ToolFailure
from researchflow.tools.offline import FileSystemNoteStore

runner = CliRunner()


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
    assert "--verbose" in result.stdout


def test_version_is_available() -> None:
    result = runner.invoke(app, ["--version"])

    assert result.exit_code == 0
    assert result.stdout.strip() == f"researchflow {__version__}"


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
