"""Tests for JSON Lines trace recording."""

import json
from pathlib import Path

import pytest

from researchflow.domain import ExecutionStatus, ExecutionTrace
from researchflow.execution import JsonlTraceRecorder, TraceRecordingError
from researchflow.tools import ToolContext


def make_context(tmp_path: Path, run_id: str = "run-1") -> ToolContext:
    documents = tmp_path / "documents"
    output = tmp_path / "output"
    documents.mkdir(exist_ok=True)
    output.mkdir(exist_ok=True)
    return ToolContext(
        working_directory=documents,
        output_directory=output,
        run_id=run_id,
    )


def make_trace(run_id: str = "run-1", trace_id: str = "trace-1") -> ExecutionTrace:
    return ExecutionTrace(
        trace_id=trace_id,
        run_id=run_id,
        call_id="call-1",
        tool_name="example",
        arguments={"query": "中文检索"},
        status=ExecutionStatus.SUCCEEDED,
        duration_ms=1.5,
    )


def test_record_creates_jsonl_and_preserves_chinese(tmp_path: Path) -> None:
    context = make_context(tmp_path)

    JsonlTraceRecorder().record(make_trace(), context)

    trace_file = context.output_directory / "traces" / "run-1.jsonl"
    assert trace_file.exists()
    assert trace_file.read_bytes().endswith(b"\n")
    text = trace_file.read_text(encoding="utf-8")
    assert "中文检索" in text
    assert json.loads(text)["arguments"] == {"query": "中文检索"}


def test_records_append_without_overwriting(tmp_path: Path) -> None:
    context = make_context(tmp_path)
    recorder = JsonlTraceRecorder()

    recorder.record(make_trace(trace_id="trace-1"), context)
    recorder.record(make_trace(trace_id="trace-2"), context)

    lines = (
        (context.output_directory / "traces" / "run-1.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    )
    assert [json.loads(line)["trace_id"] for line in lines] == ["trace-1", "trace-2"]


def test_different_runs_use_different_files(tmp_path: Path) -> None:
    recorder = JsonlTraceRecorder()
    first = make_context(tmp_path, "run-1")
    second = ToolContext(
        working_directory=first.working_directory,
        output_directory=first.output_directory,
        run_id="run-2",
    )

    recorder.record(make_trace("run-1", "trace-1"), first)
    recorder.record(make_trace("run-2", "trace-2"), second)

    traces = first.output_directory / "traces"
    assert json.loads((traces / "run-1.jsonl").read_text())["trace_id"] == "trace-1"
    assert json.loads((traces / "run-2.jsonl").read_text())["trace_id"] == "trace-2"


@pytest.mark.parametrize(
    "run_id",
    ["../escape", "/absolute", "nested/run", r"nested\run"],
)
def test_rejects_unsafe_run_ids(tmp_path: Path, run_id: str) -> None:
    context = make_context(tmp_path, run_id)

    with pytest.raises(TraceRecordingError, match="run_id"):
        JsonlTraceRecorder().record(make_trace(run_id), context)


def test_write_failure_raises_clear_error(tmp_path: Path) -> None:
    context = make_context(tmp_path)
    (context.output_directory / "traces").write_text("not a directory")

    with pytest.raises(TraceRecordingError, match="trace"):
        JsonlTraceRecorder().record(make_trace(), context)
