"""Validated domain models shared by agent and tool layers."""

from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated, Any, Self

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    model_validator,
)

NonEmptyString = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
ToolName = Annotated[str, StringConstraints(min_length=1, pattern=r"^[a-z0-9_]+$")]


def utc_now() -> datetime:
    """Return an aware UTC timestamp."""
    return datetime.now(UTC)


class DomainModel(BaseModel):
    """Base validation configuration for public domain models."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class PlanStepStatus(StrEnum):
    """Lifecycle states for one plan step."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class ExecutionStatus(StrEnum):
    """Possible outcomes recorded in an execution trace."""

    SUCCEEDED = "succeeded"
    FAILED = "failed"


class AgentStatus(StrEnum):
    """Lifecycle states for an agent run."""

    INITIALIZED = "initialized"
    PLANNED = "planned"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class PlanStep(DomainModel):
    """One ordered unit of work in a research plan."""

    step_id: NonEmptyString
    description: NonEmptyString
    status: PlanStepStatus = PlanStepStatus.PENDING
    tool_name: NonEmptyString | None = None
    result_summary: NonEmptyString | None = None
    error_message: NonEmptyString | None = None

    @model_validator(mode="after")
    def validate_status_details(self) -> Self:
        """Keep status and error information consistent."""
        if self.status is PlanStepStatus.FAILED and self.error_message is None:
            raise ValueError("failed steps require error_message")
        if self.status is not PlanStepStatus.FAILED and self.error_message is not None:
            raise ValueError("only failed steps may include error_message")
        return self


class ResearchPlan(DomainModel):
    """An ordered plan for a research goal."""

    plan_id: NonEmptyString
    goal: NonEmptyString
    steps: list[PlanStep] = Field(min_length=1)
    created_at: AwareDatetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def validate_unique_steps(self) -> Self:
        """Require stable, unique identifiers within a plan."""
        step_ids = [step.step_id for step in self.steps]
        if len(step_ids) != len(set(step_ids)):
            raise ValueError("step_id values must be unique within a plan")
        return self


class ToolCall(DomainModel):
    """A request to invoke one registered tool."""

    call_id: NonEmptyString
    tool_name: ToolName
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolResult(DomainModel):
    """The normalized result of one tool invocation."""

    call_id: NonEmptyString
    tool_name: ToolName
    success: bool
    output: Any | None = None
    error_type: NonEmptyString | None = None
    error_message: NonEmptyString | None = None

    @model_validator(mode="after")
    def validate_status_details(self) -> Self:
        """Require error details only for failed results."""
        has_complete_error = (
            self.error_type is not None and self.error_message is not None
        )
        has_any_error = self.error_type is not None or self.error_message is not None
        if not self.success and not has_complete_error:
            raise ValueError("failed results require error_type and error_message")
        if self.success and has_any_error:
            raise ValueError("successful results cannot include error details")
        return self


class ExecutionTrace(DomainModel):
    """Observable facts recorded for one tool execution."""

    trace_id: NonEmptyString
    run_id: NonEmptyString
    call_id: NonEmptyString
    tool_name: NonEmptyString
    arguments: dict[str, Any] = Field(default_factory=dict)
    status: ExecutionStatus
    started_at: AwareDatetime = Field(default_factory=utc_now)
    duration_ms: float = Field(ge=0)
    error_type: NonEmptyString | None = None
    error_message: NonEmptyString | None = None

    @model_validator(mode="after")
    def validate_status_details(self) -> Self:
        """Keep trace status and error details consistent."""
        has_complete_error = (
            self.error_type is not None and self.error_message is not None
        )
        has_any_error = self.error_type is not None or self.error_message is not None
        if self.status is ExecutionStatus.FAILED and not has_complete_error:
            raise ValueError("failed traces require error_type and error_message")
        if self.status is ExecutionStatus.SUCCEEDED and has_any_error:
            raise ValueError("successful traces cannot include error details")
        return self


class AgentState(DomainModel):
    """A serializable snapshot of one research run."""

    run_id: NonEmptyString
    query: NonEmptyString
    status: AgentStatus = AgentStatus.INITIALIZED
    plan: ResearchPlan | None = None
    current_step_id: NonEmptyString | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tool_results: list[ToolResult] = Field(default_factory=list)
    traces: list[ExecutionTrace] = Field(default_factory=list)
    final_answer: NonEmptyString | None = None
    created_at: AwareDatetime = Field(default_factory=utc_now)
    updated_at: AwareDatetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def validate_state(self) -> Self:
        """Validate relationships between state fields."""
        if self.current_step_id is not None:
            known_steps = (
                set()
                if self.plan is None
                else {step.step_id for step in self.plan.steps}
            )
            if self.current_step_id not in known_steps:
                raise ValueError("current_step_id must reference a step in the plan")
        if self.status is AgentStatus.COMPLETED and self.final_answer is None:
            raise ValueError("completed agent state requires final_answer")
        if self.updated_at < self.created_at:
            raise ValueError("updated_at cannot be earlier than created_at")
        return self
