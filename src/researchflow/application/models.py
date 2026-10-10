"""JSON-safe application DTOs for observable research runs."""

from datetime import UTC, datetime
from enum import StrEnum
from typing import TYPE_CHECKING

from pydantic import Field

from researchflow.domain.models import DomainModel

if TYPE_CHECKING:
    from researchflow.agent.graph import AgentGraphState


def _utc_now() -> datetime:
    return datetime.now(UTC)


class RunStatus(StrEnum):
    CREATED = "created"
    RUNNING = "running"
    WAITING_FOR_INPUT = "waiting_for_input"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


class StartTurn(DomainModel):
    session_id: str
    message: str
    idempotency_key: str


class ResumeTurn(DomainModel):
    """A durable answer to a pending session clarification."""

    session_id: str
    answer: str
    idempotency_key: str


class SourceSnapshot(DomainModel):
    source_id: str
    title: str
    url: str = ""
    kind: str
    read: bool
    reason: str | None = None


class EvidenceSnapshot(DomainModel):
    candidates: list[SourceSnapshot] = Field(default_factory=list)
    read_sources: list[SourceSnapshot] = Field(default_factory=list)
    rejected_sources: list[SourceSnapshot] = Field(default_factory=list)

    @classmethod
    def from_graph_state(cls, state: "AgentGraphState") -> "EvidenceSnapshot":
        candidates = [
            SourceSnapshot(
                source_id=candidate.locator,
                title=candidate.title,
                url=candidate.locator if candidate.source_type == "web" else "",
                kind=candidate.source_type,
                read=False,
            )
            for candidate in state.candidates
        ]
        documents = [
            SourceSnapshot(
                source_id=document.path,
                title=document.title,
                kind="document",
                read=True,
            )
            for document in state.documents
        ]
        web_sources = [
            SourceSnapshot(
                source_id=source.url,
                title=source.title,
                url=source.url,
                kind="web",
                read=True,
            )
            for source in state.web_sources
        ]
        rejected = [cls._rejected_source(item) for item in state.rejected_sources]
        return cls(
            candidates=candidates,
            read_sources=[*documents, *web_sources],
            rejected_sources=rejected,
        )

    @staticmethod
    def _rejected_source(item: str) -> SourceSnapshot:
        source_id, separator, reason = item.rpartition(" — ")
        if not separator:
            source_id, reason = item, "rejected"
        return SourceSnapshot(
            source_id=source_id,
            title=source_id,
            url=source_id if source_id.startswith(("http://", "https://")) else "",
            kind="web" if source_id.startswith(("http://", "https://")) else "document",
            read=False,
            reason=reason,
        )


class RunSnapshot(DomainModel):
    run_id: str
    session_id: str
    status: RunStatus
    question: str = ""
    created_at: datetime = Field(default_factory=_utc_now)
    updated_at: datetime = Field(default_factory=_utc_now)
    evidence: EvidenceSnapshot = Field(default_factory=EvidenceSnapshot)
    evidence_status: str | None = None
    evidence_gaps: list[str] = Field(default_factory=list)
    evidence_policy: str | None = None
    accepted_source_count: int = 0
    required_source_count: int = 0
    official_complete_source_id: str | None = None
    answer: str | None = None
    end_reason: str | None = None
    node_steps: int | None = None
    max_steps: int | None = None
    generation_mode: str | None = None
    report_path: str | None = None
    interrupt_prompt: str | None = None
    last_event_id: int = 0


class SessionSnapshot(DomainModel):
    session_id: str
    runs: list[RunSnapshot] = Field(default_factory=list)


class SessionRecord(DomainModel):
    """Human-facing metadata that never replaces the checkpoint identity."""

    session_id: str
    display_name: str
    updated_at: datetime


class EventEnvelope(DomainModel):
    event_id: int
    run_id: str
    session_id: str
    timestamp: datetime = Field(default_factory=_utc_now)
    type: str
    data: dict[str, object] = Field(default_factory=dict)
