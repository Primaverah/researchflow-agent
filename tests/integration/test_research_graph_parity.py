from pathlib import Path

from researchflow.agent import (
    ExtractiveSummarizer,
    LangGraphResearchRunner,
    RulePlanner,
    StateSelector,
)
from researchflow.execution import JsonlTraceRecorder, ToolExecutor
from researchflow.tools import ToolContext, ToolRegistry
from researchflow.tools.offline import create_offline_tools


def test_langgraph_research_runner_checkpoints_planning_and_retrieval(
    tmp_path: Path,
) -> None:
    documents = tmp_path / "documents"
    output = tmp_path / "output"
    documents.mkdir()
    output.mkdir()
    (documents / "evidence.md").write_text("# Evidence\n\n可靠内容。", encoding="utf-8")
    context = ToolContext(
        working_directory=documents, output_directory=output, run_id="langgraph-run"
    )
    registry = ToolRegistry()
    for tool in create_offline_tools(context):
        registry.register(tool)
    runner = LangGraphResearchRunner(
        tmp_path / "checkpoints.sqlite3",
        RulePlanner(),
        StateSelector(),
        ExtractiveSummarizer(),
        ToolExecutor(registry, JsonlTraceRecorder()),
    )

    state = runner.run("可靠内容", context, thread_id="session-a")

    assert state.candidates[0].locator == "evidence.md"
    assert state.evidence_status is not None
