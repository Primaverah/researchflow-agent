"""Safe path handling for local tools."""

from pathlib import Path, PureWindowsPath

from researchflow.tools.errors import UnsafePathError


def resolve_safe_path(
    root: Path,
    requested_path: str | Path,
    *,
    must_exist: bool = False,
) -> Path:
    """Resolve a relative path and ensure it remains inside ``root``."""
    try:
        resolved_root = root.expanduser().resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise UnsafePathError("allowed root does not exist") from exc
    if not resolved_root.is_dir():
        raise UnsafePathError("allowed root must be a directory")

    raw_path = str(requested_path)
    if not raw_path.strip():
        raise UnsafePathError("requested path cannot be empty")

    path = Path(requested_path)
    if path.is_absolute() or PureWindowsPath(raw_path).is_absolute():
        raise UnsafePathError("absolute paths are not allowed")
    portable_parts = raw_path.replace("\\", "/").split("/")
    if ".." in portable_parts:
        raise UnsafePathError("parent traversal is not allowed")

    try:
        candidate = (resolved_root / path).resolve(strict=must_exist)
    except (OSError, RuntimeError) as exc:
        if must_exist:
            raise UnsafePathError("requested path does not exist") from exc
        raise UnsafePathError("requested path cannot be resolved") from exc

    try:
        candidate.relative_to(resolved_root)
    except ValueError as exc:
        raise UnsafePathError(
            "requested path resolves outside the allowed root"
        ) from exc
    return candidate
