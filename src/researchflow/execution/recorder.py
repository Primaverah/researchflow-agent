"""Synchronous execution trace recorders."""

import json
from pathlib import Path, PureWindowsPath
from typing import Protocol

from researchflow.domain import ExecutionTrace
from researchflow.execution.errors import TraceRecordingError
from researchflow.tools import ToolContext, UnsafePathError, resolve_safe_path


class TraceRecorder(Protocol):
    """Destination for completed execution traces."""

    def record(self, trace: ExecutionTrace, context: ToolContext) -> None:
        """Persist one completed trace."""
        ...


class JsonlTraceRecorder:
    """Append one JSON object per line to a run-specific trace file."""

    def record(self, trace: ExecutionTrace, context: ToolContext) -> None:
        """Append a trace below ``context.output_directory/traces``."""
        self._validate_run_id(context.run_id)
        if trace.run_id != context.run_id:
            raise TraceRecordingError("trace run_id does not match the tool context")

        try:
            traces_directory = context.output_directory / "traces"
            traces_directory.mkdir(parents=True, exist_ok=True)
            safe_directory = resolve_safe_path(
                context.output_directory, "traces", must_exist=True
            )
            trace_file = resolve_safe_path(
                safe_directory,
                f"{context.run_id}.jsonl",
            )
            payload = json.dumps(
                trace.model_dump(mode="json"),
                ensure_ascii=False,
                separators=(",", ":"),
            )
            with trace_file.open("a", encoding="utf-8", newline="\n") as stream:
                stream.write(payload)
                stream.write("\n")
        except TraceRecordingError:
            raise
        except (OSError, TypeError, ValueError, UnsafePathError) as exc:
            raise TraceRecordingError("failed to record execution trace") from exc

    @staticmethod
    def _validate_run_id(run_id: str) -> None:
        """Reject run identifiers that could influence the output path."""
        path = Path(run_id)
        windows_path = PureWindowsPath(run_id)
        if (
            run_id in {".", ".."}
            or path.is_absolute()
            or windows_path.is_absolute()
            or path.name != run_id
            or windows_path.name != run_id
            or "/" in run_id
            or "\\" in run_id
        ):
            raise TraceRecordingError("run_id must be a single safe path component")
