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


def test_run_and_session_routes_return_persisted_snapshots(tmp_path) -> None:
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
    created = client.post(
        "/api/sessions/demo/turns",
        json={"message": "tool calling"},
        headers={"Idempotency-Key": "request-1"},
    ).json()

    run = client.get(f"/api/runs/{created['run_id']}")
    session = client.get("/api/sessions/demo")

    assert run.status_code == 200
    assert run.json()["run_id"] == created["run_id"]
    assert session.status_code == 200
    assert [item["run_id"] for item in session.json()["runs"]] == [created["run_id"]]


def test_unknown_run_is_not_found(tmp_path) -> None:
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

    response = client.get("/api/runs/missing")

    assert response.status_code == 404


def test_sessions_route_lists_application_sessions(tmp_path) -> None:
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
    client.post(
        "/api/sessions/demo/turns",
        json={"message": "tool calling"},
        headers={"Idempotency-Key": "request-1"},
    )

    response = client.get("/api/sessions")

    assert response.status_code == 200
    assert response.json() == ["demo"]
