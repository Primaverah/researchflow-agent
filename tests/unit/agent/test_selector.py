"""Tests for state-driven action selection."""

from researchflow.agent import RulePlanner, StateSelector
from researchflow.agent.models import AgentActionType
from researchflow.domain import AgentState, ToolCall, ToolResult


def make_state() -> AgentState:
    return AgentState(
        run_id="run-1",
        query="Agent 工具",
        plan=RulePlanner().create_plan("Agent 工具"),
    )


def add_result(
    state: AgentState,
    tool_name: str,
    arguments: dict,
    *,
    success: bool = True,
    output=None,
) -> None:
    call_id = f"call-{len(state.tool_calls) + 1}"
    state.tool_calls.append(
        ToolCall(call_id=call_id, tool_name=tool_name, arguments=arguments)
    )
    state.tool_results.append(
        ToolResult(
            call_id=call_id,
            tool_name=tool_name,
            success=success,
            output=output,
            error_type=None if success else "failure",
            error_message=None if success else "failed",
        )
    )


def test_initial_state_selects_search() -> None:
    action = StateSelector().select(make_state())

    assert action.action_type is AgentActionType.SEARCH
    assert action.arguments == {"query": "Agent 工具", "limit": 5}


def test_reads_candidates_in_order_without_repeating_failures() -> None:
    state = make_state()
    hits = [{"path": f"{name}.md"} for name in ("a", "b", "c")]
    add_result(state, "search_documents", {"query": state.query}, output={"hits": hits})

    first = StateSelector().select(state)
    add_result(state, "read_document", first.arguments, success=False)
    second = StateSelector().select(state)

    assert first.arguments == {"path": "a.md"}
    assert second.arguments == {"path": "b.md"}


def test_stops_reading_after_three_successes() -> None:
    state = make_state()
    hits = [{"path": f"{index}.md"} for index in range(5)]
    add_result(state, "search_documents", {}, output={"hits": hits})
    for hit in hits[:3]:
        add_result(
            state,
            "read_document",
            {"path": hit["path"]},
            output={"path": hit["path"]},
        )

    assert StateSelector().select(state).action_type is AgentActionType.SUMMARIZE


def test_only_considers_first_five_search_hits() -> None:
    state = make_state()
    hits = [{"path": f"{index}.md"} for index in range(6)]
    add_result(state, "search_documents", {}, output={"hits": hits})
    for hit in hits[:5]:
        add_result(state, "read_document", {"path": hit["path"]}, success=False)

    action = StateSelector(search_limit=5).select(state)

    assert action.action_type is AgentActionType.SUMMARIZE


def test_empty_search_summarizes_and_failed_search_finishes() -> None:
    empty = make_state()
    add_result(empty, "search_documents", {}, output={"hits": []})
    failed = make_state()
    add_result(failed, "search_documents", {}, success=False)

    assert StateSelector().select(empty).action_type is AgentActionType.SUMMARIZE
    assert StateSelector().select(failed).action_type is AgentActionType.FINISH


def test_summary_selects_save_and_any_save_result_finishes() -> None:
    state = make_state()
    add_result(state, "search_documents", {}, output={"hits": []})
    state.final_answer = "report"

    save = StateSelector().select(state)
    add_result(state, "save_note", save.arguments, success=False)

    assert save.action_type is AgentActionType.SAVE
    assert save.arguments["path"] == "notes/run-1.md"
    assert StateSelector().select(state).action_type is AgentActionType.FINISH
