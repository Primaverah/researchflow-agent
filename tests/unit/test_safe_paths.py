"""Tests for safe path resolution."""

from pathlib import Path

import pytest

from researchflow.tools import UnsafePathError, resolve_safe_path


def test_resolves_path_inside_root(tmp_path) -> None:
    document = tmp_path / "docs" / "example.md"
    document.parent.mkdir()
    document.write_text("example", encoding="utf-8")

    assert resolve_safe_path(tmp_path, "docs/example.md", must_exist=True) == document


@pytest.mark.parametrize(
    "requested_path",
    ["../secret.txt", "docs/../../secret.txt", "docs/../example.md"],
)
def test_rejects_parent_traversal(tmp_path, requested_path: str) -> None:
    with pytest.raises(UnsafePathError, match="parent traversal"):
        resolve_safe_path(tmp_path, requested_path)


def test_rejects_absolute_paths(tmp_path) -> None:
    with pytest.raises(UnsafePathError, match="absolute"):
        resolve_safe_path(tmp_path, tmp_path / "example.md")


def test_rejects_symlink_escape(tmp_path) -> None:
    root = tmp_path / "root"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    link = root / "linked"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symbolic links are unavailable: {exc}")

    with pytest.raises(UnsafePathError, match="outside"):
        resolve_safe_path(root, "linked/new-note.md")


def test_allows_nonexistent_file_under_safe_parent(tmp_path) -> None:
    expected = tmp_path / "new-note.md"

    assert resolve_safe_path(tmp_path, "new-note.md") == expected


def test_must_exist_rejects_missing_path(tmp_path) -> None:
    with pytest.raises(UnsafePathError, match="does not exist"):
        resolve_safe_path(tmp_path, "missing.md", must_exist=True)


def test_rejects_invalid_root(tmp_path) -> None:
    root_file = tmp_path / "file.txt"
    root_file.write_text("not a directory", encoding="utf-8")

    with pytest.raises(UnsafePathError, match="root"):
        resolve_safe_path(root_file, Path("child"))
