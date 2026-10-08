"""FastAPI application factory for local-only use."""

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from researchflow.application.models import StartTurn
from researchflow.application.service import ResearchService


class TurnRequest(BaseModel):
    message: str = Field(min_length=1)


def create_app(service: ResearchService) -> FastAPI:
    app = FastAPI(title="ResearchFlow Local API")

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"scope": "loopback"}

    @app.get("/api/sessions")
    def list_sessions() -> list[str]:
        return service.list_sessions()

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

    @app.get("/api/runs/{run_id}")
    def get_run(run_id: str):
        try:
            return service.get_run(run_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="run not found") from exc

    @app.get("/api/sessions/{session_id}")
    def get_session(session_id: str):
        return service.get_session(session_id)

    return app
