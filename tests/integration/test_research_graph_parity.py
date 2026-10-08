from pathlib import Path

from researchflow.agent import (
    ExtractiveSummarizer,
    GraphAgentRunner,
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


def test_langgraph_research_graph_state_is_available_to_a_new_runner(
    tmp_path: Path,
) -> None:
    documents = tmp_path / "documents"
    output = tmp_path / "output"
    documents.mkdir()
    output.mkdir()
    (documents / "evidence.md").write_text("# Evidence\n\n可靠内容。", encoding="utf-8")
    context = ToolContext(
        working_directory=documents, output_directory=output, run_id="checkpoint-run"
    )
    registry = ToolRegistry()
    for tool in create_offline_tools(context):
        registry.register(tool)
    database = tmp_path / "checkpoints.sqlite3"
    first = LangGraphResearchRunner(
        database,
        RulePlanner(),
        StateSelector(),
        ExtractiveSummarizer(),
        ToolExecutor(registry, JsonlTraceRecorder()),
    )
    first.run("可靠内容", context, thread_id="recoverable")
    second = LangGraphResearchRunner(
        database,
        RulePlanner(),
        StateSelector(),
        ExtractiveSummarizer(),
        ToolExecutor(registry, JsonlTraceRecorder()),
    )

    restored = second.get_state("recoverable")

    assert restored.candidates[0].locator == "evidence.md"


def test_langgraph_planning_keeps_legacy_candidate_order(tmp_path: Path) -> None:
    documents = tmp_path / "documents"
    output = tmp_path / "output"
    documents.mkdir()
    output.mkdir()
    (documents / "first.md").write_text("# First\n\n内容", encoding="utf-8")
    (documents / "second.md").write_text("# Second\n\n内容", encoding="utf-8")

    def build_context(run_id: str) -> ToolContext:
        return ToolContext(
            working_directory=documents, output_directory=output, run_id=run_id
        )

    def build_executor(context: ToolContext) -> ToolExecutor:
        registry = ToolRegistry()
        for tool in create_offline_tools(context):
            registry.register(tool)
        return ToolExecutor(registry, JsonlTraceRecorder())

    legacy_context = build_context("legacy")
    legacy = GraphAgentRunner(
        RulePlanner(),
        StateSelector(),
        ExtractiveSummarizer(),
        build_executor(legacy_context),
    ).run("内容", legacy_context)
    new_context = build_context("new")
    modern = LangGraphResearchRunner(
        tmp_path / "checkpoints.sqlite3",
        RulePlanner(),
        StateSelector(),
        ExtractiveSummarizer(),
        build_executor(new_context),
    ).run("内容", new_context, thread_id="parity")

    legacy_candidates = [
        call.arguments["query"]
        for call in legacy.tool_calls
        if call.tool_name == "search_documents"
    ]
    assert legacy_candidates == ["内容"]
    assert [candidate.locator for candidate in modern.candidates] == [
        "first.md",
        "second.md",
    ]
