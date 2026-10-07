"""Minimal non-secret project configuration loading."""

import os
import tomllib
from pathlib import Path
from typing import Any


class ConfigurationError(ValueError):
    """Raised when a requested project configuration cannot be read."""


def load_project_config(path: Path | None = None) -> dict[str, Any]:
    config_path = path or Path.cwd() / "researchflow.toml"
    if not config_path.is_file():
        return {}
    try:
        return tomllib.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ConfigurationError("researchflow.toml is invalid") from exc


def load_local_secrets(directory: Path | None = None) -> None:
    """Load supported local runtime settings without overriding process values."""
    path = (directory or Path.cwd()) / ".env.local"
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition("=")
        if separator and key in {
            "RESEARCHFLOW_LLM_API_KEY",
            "RESEARCHFLOW_LLM_BASE_URL",
            "RESEARCHFLOW_LLM_MODEL",
            "RESEARCHFLOW_LLM_TIMEOUT",
            "RESEARCHFLOW_SEARCH_API_KEY",
        }:
            os.environ.setdefault(key, _unquote_secret(value.strip()))


def _unquote_secret(value: str) -> str:
    """Accept conventional matching .env quote delimiters without altering keys."""
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def value(config: dict[str, Any], section: str, name: str, default: Any) -> Any:
    return os.getenv(
        f"RESEARCHFLOW_{section.upper()}_{name.upper()}",
        config.get(section, {}).get(name, default),
    )
