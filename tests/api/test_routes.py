from fastapi.testclient import TestClient

from researchflow.api.app import create_app
from researchflow.application.service import ResearchService
from researchflow.domain import AgentState, AgentStatus


def test_health_reports_local_capabilities(tmp_path) -> None:
    service = ResearchService(
        tmp_path / "checkpoints.sqlite3",
        workflow=lambda message, _: AgentState(
            run_id="workflow-run",
            query=message,
            status=AgentStatus.COMPLETED,
            final_answer="verified answer",
        ),
    )
    client = TestClient(create_app(service))

    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json()["scope"] == "loopback"


def test_turn_post_returns_accepted_run_snapshot(tmp_path) -> None:
    service = ResearchService(
        tmp_path / "checkpoints.sqlite3",
        workflow=lambda message, _: AgentState(
            run_id="workflow-run",
            query=message,
            status=AgentStatus.COMPLETED,
            final_answer="verified answer",
        ),
    )
    client = TestClient(create_app(service))

    response = client.post(
        "/api/sessions/demo/turns",
        json={"message": "tool calling"},
        headers={"Idempotency-Key": "request-1"},
    )

    assert response.status_code == 202
    assert response.json()["session_id"] == "demo"
    assert response.json()["status"] == "completed"
