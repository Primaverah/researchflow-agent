"""Tests for the command-line interface."""

from pathlib import Path

from typer.testing import CliRunner

from researchflow import __version__
from researchflow.cli import app

runner = CliRunner()


def test_help_is_available() -> None:
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "Research technical topics" in result.stdout


def test_version_is_available() -> None:
    result = runner.invoke(app, ["--version"])

    assert result.exit_code == 0
    assert result.stdout.strip() == f"researchflow {__version__}"


def test_run_executes_offline_agent(tmp_path: Path) -> None:
    documents = tmp_path / "documents"
    output = tmp_path / "output"
    documents.mkdir()
    (documents / "agent.md").write_text(
        "# Agent\n\nTool calling uses validated tools.", encoding="utf-8"
    )

    result = runner.invoke(
        app,
        [
            "run",
            "tool calling",
            "--documents-dir",
            str(documents),
            "--output-dir",
            str(output),
        ],
    )

    assert result.exit_code == 0
    assert "# 研究报告" in result.stdout
    assert list((output / "notes").glob("*.md"))
    assert list((output / "traces").glob("*.jsonl"))


def test_run_rejects_missing_documents_directory(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        ["run", "question", "--documents-dir", str(tmp_path / "missing")],
    )

    assert result.exit_code == 2
    assert "documents directory" in result.stderr
