"""Tests for the command-line interface."""

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
