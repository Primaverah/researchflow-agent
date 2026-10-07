"""Tests for terminal-safe CLI output."""

from researchflow.cli import terminal_safe_text


def test_terminal_safe_text_replaces_unencodable_characters() -> None:
    rendered = terminal_safe_text("来源 • 成龙", encoding="gbk")

    assert rendered == "来源 ? 成龙"
