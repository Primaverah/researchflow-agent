"""Shared synchronous lifecycle service for research callers."""

from collections.abc import Callable
from pathlib import Path

from researchflow.application.models import RunSnapshot, RunStatus, StartTurn
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
        self._workflow = workflow

    def start_turn(self, request: StartTurn) -> RunSnapshot:
        snapshot = self._store.create_run(request.session_id, request.idempotency_key)
        if snapshot.status is not RunStatus.CREATED:
            return snapshot
        running = self._store.save(
            snapshot.model_copy(update={"status": RunStatus.RUNNING})
        )
        state = self._workflow(request.message, request.session_id)
        return self._store.save(
            running.model_copy(
                update={
                    "status": (
                        RunStatus.COMPLETED
                        if state.final_answer is not None
                        else RunStatus.INSUFFICIENT_EVIDENCE
                    )
                }
            )
        )

    def get_run(self, run_id: str) -> RunSnapshot:
        return self._store.get_run(run_id)
