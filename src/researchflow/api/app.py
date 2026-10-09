"""FastAPI application factory for local-only use."""

import json

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field, field_validator

from researchflow.application.models import ResumeTurn, SessionRecord, StartTurn
from researchflow.application.service import ResearchService


class TurnRequest(BaseModel):
    message: str = Field(min_length=1)


class ResumeRequest(BaseModel):
    answer: str = Field(min_length=1)


class SessionRenameRequest(BaseModel):
    display_name: str = Field(min_length=1, max_length=80)

    @field_validator("display_name")
    @classmethod
    def require_non_blank_name(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("display name must not be empty")
        return cleaned


def create_app(service: ResearchService) -> FastAPI:
    app = FastAPI(title="ResearchFlow Local API")

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"scope": "loopback"}

    @app.get("/api/sessions")
    def list_sessions() -> list[SessionRecord]:
        return service.list_sessions()

    @app.patch("/api/sessions/{session_id}")
    def rename_session(session_id: str, request: SessionRenameRequest) -> SessionRecord:
        try:
            return service.rename_session(session_id, request.display_name)
        except (KeyError, ValueError) as exc:
            status_code = 404 if isinstance(exc, KeyError) else 422
            raise HTTPException(status_code=status_code, detail=str(exc)) from exc

    @app.delete("/api/sessions/{session_id}", status_code=204)
    def delete_session(session_id: str) -> Response:
        try:
            service.delete_session(session_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="session not found") from exc
        return Response(status_code=204)

    @app.post("/api/sessions/{session_id}/turns", status_code=202)
    def start_turn(
        session_id: str,
        request: TurnRequest,
        idempotency_key: str | None = Header(default=None),
    ):
        if not idempotency_key:
            raise HTTPException(status_code=422, detail="Idempotency-Key is required")
        return service.start_turn(
            StartTurn(
                session_id=session_id,
                message=request.message,
                idempotency_key=idempotency_key,
            )
        )

    @app.post("/api/sessions/{session_id}/resume")
    def resume_turn(
        session_id: str,
        request: ResumeRequest,
        idempotency_key: str | None = Header(default=None),
    ):
        if not idempotency_key:
            raise HTTPException(status_code=422, detail="Idempotency-Key is required")
        try:
            return service.resume_turn(
                ResumeTurn(
                    session_id=session_id,
                    answer=request.answer,
                    idempotency_key=idempotency_key,
                )
            )
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.get("/api/runs/{run_id}")
    def get_run(run_id: str):
        try:
            return service.get_run(run_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="run not found") from exc

    @app.get("/api/runs/{run_id}/events")
    def run_events(
        run_id: str,
        last_event_id: str | None = Header(default=None),
    ) -> StreamingResponse:
        try:
            service.get_run(run_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="run not found") from exc
        try:
            after_event_id = int(last_event_id or "0")
        except ValueError as exc:
            raise HTTPException(
                status_code=422, detail="Last-Event-ID must be an integer"
            ) from exc
        if after_event_id < 0:
            raise HTTPException(
                status_code=422, detail="Last-Event-ID must not be negative"
            )

        def event_stream():
            for event in service.subscribe_to_events(
                run_id, after_event_id=after_event_id
            ):
                if event is None:
                    yield ": keepalive\n\n"
                    continue
                payload = json.dumps(
                    event.data, ensure_ascii=False, separators=(",", ":")
                )
                yield f"id: {event.event_id}\nevent: {event.type}\ndata: {payload}\n\n"
                if event.type in {"run_completed", "run_failed"}:
                    break

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    @app.get("/api/sessions/{session_id}")
    def get_session(session_id: str):
        snapshot = service.get_session(session_id)
        if not snapshot.runs:
            raise HTTPException(status_code=404, detail="session not found")
        return snapshot

    return app
