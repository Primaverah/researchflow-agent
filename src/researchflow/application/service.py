"""Shared synchronous lifecycle service for research callers."""

from collections.abc import Callable, Mapping
from inspect import Parameter, signature
from pathlib import Path

from researchflow.application.events import EventBroker
from researchflow.application.models import (
    EventEnvelope,
    ResumeTurn,
    RunSnapshot,
    RunStatus,
    SessionSnapshot,
    SourceSnapshot,
    StartTurn,
)
from researchflow.application.run_store import SqliteRunStore
from researchflow.domain import AgentState


class ResearchService:
    def __init__(
        self,
        database: Path,
        *,
        workflow: Callable[..., AgentState],
        session_runner: object | None = None,
    ) -> None:
        self._store = SqliteRunStore(database)
        self._events = EventBroker(self._store)
        self._workflow = workflow
        self._session_runner = session_runner

    def start_turn(self, request: StartTurn) -> RunSnapshot:
        snapshot = self._store.create_run(
            request.session_id, request.idempotency_key, question=request.message
        )
        if snapshot.status is not RunStatus.CREATED:
            return snapshot
        if self._session_runner is not None:
            return self._start_session_turn(snapshot, request)
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
        state = self._run_workflow(request.message, request.session_id, running.run_id)
        projected = self._store.get_run(running.run_id)
        status = (
            RunStatus.INSUFFICIENT_EVIDENCE
            if projected.evidence_status == "insufficient" or state.final_answer is None
            else RunStatus.COMPLETED
        )
        existing_events = self._events.replay(projected.run_id)
        latest_event = existing_events[-1] if existing_events else None
        if latest_event is not None and latest_event.type in {
            "run_completed",
            "run_failed",
        }:
            terminal_event_id = latest_event.event_id
        else:
            terminal_event_id = self._events.publish(
                run_id=projected.run_id,
                session_id=projected.session_id,
                event_type="run_completed",
                data={"status": status.value},
            ).event_id
        return self._store.save(
            projected.model_copy(
                update={
                    "status": status,
                    "answer": state.final_answer,
                    "last_event_id": terminal_event_id,
                }
            )
        )

    def resume_turn(self, request: ResumeTurn) -> RunSnapshot:
        previous = self._store.get_resume(request.session_id, request.idempotency_key)
        if previous is not None:
            return previous
        waiting = [
            run
            for run in self._store.list_session_runs(request.session_id)
            if run.status is RunStatus.WAITING_FOR_INPUT
        ]
        if len(waiting) != 1 or self._session_runner is None:
            raise ValueError("session does not have exactly one waiting run")
        result = self._session_runner.resume(request.session_id, request.answer)
        snapshot = self._store.save(
            waiting[0].model_copy(
                update={
                    "status": RunStatus.COMPLETED,
                    "answer": result.response,
                    "interrupt_prompt": None,
                }
            )
        )
        self._store.save_resume(request.session_id, request.idempotency_key, snapshot)
        return snapshot

    def _start_session_turn(
        self, snapshot: RunSnapshot, request: StartTurn
    ) -> RunSnapshot:
        result = self._session_runner.chat(
            request.message, session_id=request.session_id
        )
        if result.interrupted:
            return self._store.save(
                snapshot.model_copy(
                    update={
                        "status": RunStatus.WAITING_FOR_INPUT,
                        "interrupt_prompt": result.prompt,
                    }
                )
            )
        return self._store.save(
            snapshot.model_copy(
                update={"status": RunStatus.COMPLETED, "answer": result.response}
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

    def subscribe_to_events(self, run_id: str, *, after_event_id: int = 0):
        return self._events.subscribe(run_id, after_event_id=after_event_id)

    def publish_event(
        self,
        run_id: str,
        event_type: str,
        data: Mapping[str, object] | None = None,
    ) -> RunSnapshot:
        """Persist a safe event and project its display-safe fields onto its run."""
        snapshot = self._store.get_run(run_id)
        event = self._events.publish(
            run_id=run_id,
            session_id=snapshot.session_id,
            event_type=event_type,
            data=data,
        )
        return self._store.save(self._project_event(snapshot, event))

    def _run_workflow(self, message: str, session_id: str, run_id: str) -> AgentState:
        if not self._workflow_accepts_events():
            return self._workflow(message, session_id)

        def publish(event_type: str, data: Mapping[str, object]) -> None:
            self.publish_event(run_id, event_type, data)

        return self._workflow(message, session_id, publish)

    def _workflow_accepts_events(self) -> bool:
        try:
            parameters = tuple(signature(self._workflow).parameters.values())
        except (TypeError, ValueError):
            return False
        positional = tuple(
            parameter
            for parameter in parameters
            if parameter.kind
            in {Parameter.POSITIONAL_ONLY, Parameter.POSITIONAL_OR_KEYWORD}
        )
        return len(positional) >= 3 or any(
            parameter.kind is Parameter.VAR_POSITIONAL for parameter in parameters
        )

    @staticmethod
    def _project_event(snapshot: RunSnapshot, event: EventEnvelope) -> RunSnapshot:
        evidence = snapshot.evidence.model_copy(deep=True)
        data = event.data
        if event.type == "candidate_selected":
            evidence.candidates = ResearchService._append_source(
                evidence.candidates,
                ResearchService._source_from_event(data, read=False),
            )
        elif event.type == "source_read":
            evidence.read_sources = ResearchService._append_source(
                evidence.read_sources,
                ResearchService._source_from_event(data, read=True),
            )
        elif event.type == "source_rejected":
            evidence.rejected_sources = ResearchService._append_source(
                evidence.rejected_sources,
                ResearchService._source_from_event(data, read=False),
            )
        update: dict[str, object] = {
            "evidence": evidence,
            "last_event_id": event.event_id,
        }
        if event.type == "evidence_assessed":
            status = data.get("status")
            gaps = data.get("gaps")
            if isinstance(status, str):
                update["evidence_status"] = status
            if isinstance(gaps, list) and all(isinstance(item, str) for item in gaps):
                update["evidence_gaps"] = gaps
            policy = data.get("policy")
            if isinstance(policy, str):
                update["evidence_policy"] = policy
            for field in ("accepted_source_count", "required_source_count"):
                value = data.get(field)
                if isinstance(value, int) and value >= 0:
                    update[field] = value
            official_source_id = data.get("official_complete_source_id")
            if isinstance(official_source_id, str) or official_source_id is None:
                update["official_complete_source_id"] = official_source_id
        if event.type == "generation_status":
            mode = data.get("mode")
            if isinstance(mode, str):
                update["generation_mode"] = mode
        return snapshot.model_copy(update=update)

    @staticmethod
    def _source_from_event(data: Mapping[str, object], *, read: bool) -> SourceSnapshot:
        source_id = data.get("source_id")
        title = data.get("title")
        kind = data.get("kind")
        reason = data.get("reason")
        identifier = source_id if isinstance(source_id, str) else "unknown-source"
        return SourceSnapshot(
            source_id=identifier,
            title=title if isinstance(title, str) else identifier,
            url=identifier if identifier.startswith(("http://", "https://")) else "",
            kind=kind if isinstance(kind, str) else "web",
            read=read,
            reason=reason if isinstance(reason, str) else None,
        )

    @staticmethod
    def _append_source(
        sources: list[SourceSnapshot], source: SourceSnapshot
    ) -> list[SourceSnapshot]:
        return (
            [*sources, source]
            if all(item.source_id != source.source_id for item in sources)
            else sources
        )
