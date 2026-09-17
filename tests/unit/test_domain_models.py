"""Tests for ResearchFlow domain models."""

from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from researchflow.domain import (
    AgentState,
    AgentStatus,
    ExecutionStatus,
    ExecutionTrace,
    PlanStep,
    PlanStepStatus,
    ResearchPlan,
    ToolCall,
    ToolResult,
    ToolResultStatus,
)


def test_research_plan_requires_unique_steps() -> None:
    step = PlanStep(step_id="search", description="Search documents")

    with pytest.raises(ValidationError, match="step_id values must be unique"):
        ResearchPlan(plan_id="plan-1", goal="Research tools", steps=[step, step])


def test_models_reject_unknown_fields() -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        ToolCall(
            call_id="call-1",
            tool_name="example",
            arguments={},
            unexpected=True,
        )


@pytest.mark.parametrize(
    ("status", "error_message"),
    [(PlanStepStatus.FAILED, None), (PlanStepStatus.COMPLETED, "unexpected")],
)
def test_plan_step_status_and_error_must_agree(
    status: PlanStepStatus, error_message: str | None
) -> None:
    with pytest.raises(ValidationError):
        PlanStep(
            step_id="step-1",
            description="A step",
            status=status,
            error_message=error_message,
        )


def test_failed_tool_result_requires_error_details() -> None:
    with pytest.raises(ValidationError):
        ToolResult(
            call_id="call-1",
            tool_name="example",
            status=ToolResultStatus.FAILED,
        )


def test_execution_trace_rejects_negative_duration() -> None:
    with pytest.raises(ValidationError):
        ExecutionTrace(
            trace_id="trace-1",
            run_id="run-1",
            call_id="call-1",
            tool_name="example",
            arguments={},
            status=ExecutionStatus.SUCCEEDED,
            duration_ms=-1,
        )


def test_agent_state_validates_current_step_and_completion() -> None:
    plan = ResearchPlan(
        plan_id="plan-1",
        goal="Research tools",
        steps=[PlanStep(step_id="step-1", description="Search")],
    )

    with pytest.raises(ValidationError, match="current_step_id"):
        AgentState(
            run_id="run-1", query="Research tools", plan=plan, current_step_id="x"
        )

    with pytest.raises(ValidationError, match="final_answer"):
        AgentState(
            run_id="run-1",
            query="Research tools",
            status=AgentStatus.COMPLETED,
            plan=plan,
        )


def test_agent_state_rejects_backwards_timestamp() -> None:
    now = datetime.now(UTC)

    with pytest.raises(ValidationError, match="updated_at"):
        AgentState(
            run_id="run-1",
            query="Research tools",
            created_at=now,
            updated_at=now - timedelta(seconds=1),
        )


def test_agent_state_collections_are_not_shared() -> None:
    first = AgentState(run_id="run-1", query="First")
    second = AgentState(run_id="run-2", query="Second")

    first.tool_calls.append(
        ToolCall(call_id="call-1", tool_name="example", arguments={})
    )

    assert second.tool_calls == []
