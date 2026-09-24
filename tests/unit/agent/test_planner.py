"""Tests for the fixed rule planner."""

import pytest

from researchflow.agent import RulePlanner


def test_planner_creates_stable_four_step_plan() -> None:
    plan = RulePlanner().create_plan("如何设计工具调用？")

    assert plan.goal == "如何设计工具调用？"
    assert [step.step_id for step in plan.steps] == [
        "search",
        "read",
        "summarize",
        "save",
    ]
    assert [step.tool_name for step in plan.steps] == [
        "search_documents",
        "read_document",
        None,
        "save_note",
    ]


@pytest.mark.parametrize("query", ["", "   ", "\n\t"])
def test_planner_rejects_blank_query(query: str) -> None:
    with pytest.raises(ValueError, match="blank"):
        RulePlanner().create_plan(query)
