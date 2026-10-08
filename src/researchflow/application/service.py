"""Shared synchronous lifecycle service for research callers."""

from collections.abc import Callable
from pathlib import Path

from researchflow.application.events import EventBroker
from researchflow.application.models import (
    EventEnvelope,
    RunSnapshot,
    RunStatus,
    SessionSnapshot,
    StartTurn,
)
from researchflow.application.run_store import SqliteRunStore
from researchflow.domain import AgentState


class ResearchService:
    def __init__(
        self,
        database: Path,
        *,
        workflow: Callable[[str, str], AgentState],
    ) -> None:
        self._store = SqliteRunStore(database)
        self._events = EventBroker(self._store)
        self._workflow = workflow

    def start_turn(self, request: StartTurn) -> RunSnapshot:
        snapshot = self._store.create_run(request.session_id, request.idempotency_key)
        if snapshot.status is not RunStatus.CREATED:
            return snapshot
        started = self._events.publish(
            run_id=snapshot.run_id,
            session_id=request.session_id,
            event_type="run_started",
            data={"status": RunStatus.RUNNING.value},
        )
        running = self._store.save(
            snapshot.model_copy(
                update={"status": RunStatus.RUNNING, "last_event_id": started.event_id}
            )
        )
        state = self._workflow(request.message, request.session_id)
        status = (
            RunStatus.COMPLETED
            if state.final_answer is not None
            else RunStatus.INSUFFICIENT_EVIDENCE
        )
        completed = self._events.publish(
            run_id=running.run_id,
            session_id=running.session_id,
            event_type=(
                "run_completed"
                if status is RunStatus.COMPLETED
                else "evidence_assessed"
            ),
            data={"status": status.value},
        )
        return self._store.save(
            running.model_copy(
                update={
                    "status": status,
                    "answer": state.final_answer,
                    "last_event_id": completed.event_id,
                }
            )
        )

    def get_run(self, run_id: str) -> RunSnapshot:
        return self._store.get_run(run_id)

    def get_session(self, session_id: str) -> SessionSnapshot:
        return SessionSnapshot(
            session_id=session_id, runs=self._store.list_session_runs(session_id)
        )

    def list_sessions(self) -> list[str]:
        return self._store.list_session_ids()

    def events_for_run(
        self, run_id: str, *, after_event_id: int = 0
    ) -> list[EventEnvelope]:
        return self._events.replay(run_id, after_event_id=after_event_id)
