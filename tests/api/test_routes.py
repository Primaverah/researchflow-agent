from fastapi.testclient import TestClient

from researchflow.agent.session import ChatResult
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
    assert response.json()[0]["session_id"] == "demo"
    assert response.json()[0]["display_name"] == "tool calling"


def test_session_api_renames_then_deletes_exact_session(tmp_path) -> None:
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
    for session_id, key in (("alpha", "one"), ("beta", "two")):
        client.post(
            f"/api/sessions/{session_id}/turns",
            json={"message": session_id},
            headers={"Idempotency-Key": key},
        )

    renamed = client.patch("/api/sessions/alpha", json={"display_name": "作者资料"})
    deleted = client.delete("/api/sessions/alpha")

    assert renamed.status_code == 200
    assert renamed.json()["display_name"] == "作者资料"
    assert deleted.status_code == 204
    assert client.get("/api/sessions/alpha").status_code == 404
    assert [item["session_id"] for item in client.get("/api/sessions").json()] == [
        "beta"
    ]


def test_session_api_rejects_blank_or_unknown_rename(tmp_path) -> None:
    service = ResearchService(
        tmp_path / "checkpoints.sqlite3",
        workflow=lambda message, _: AgentState(
            run_id="workflow-run", query=message, status=AgentStatus.COMPLETED
        ),
    )
    client = TestClient(create_app(service))

    missing = client.patch("/api/sessions/missing", json={"display_name": "名称"})
    blank = client.patch("/api/sessions/missing", json={"display_name": " "})

    assert missing.status_code == 404
    assert blank.status_code == 422
    assert client.delete("/api/sessions/missing").status_code == 404


def test_resume_rejects_session_without_a_waiting_run(tmp_path) -> None:
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
        "/api/sessions/demo/resume",
        json={"answer": "answer"},
        headers={"Idempotency-Key": "resume-1"},
    )

    assert response.status_code == 409


def test_resume_uses_waiting_session_runner(tmp_path) -> None:
    class FakeSessionRunner:
        def chat(self, _message: str, *, session_id: str) -> ChatResult:
            assert session_id == "clarify"
            return ChatResult(interrupted=True, prompt="请补充问题")

        def resume(self, session_id: str, answer: str) -> ChatResult:
            assert (session_id, answer) == ("clarify", "具体问题")
            return ChatResult(response="已恢复")

    service = ResearchService(
        tmp_path / "checkpoints.sqlite3",
        workflow=lambda message, _: AgentState(
            run_id="workflow-run",
            query=message,
            status=AgentStatus.COMPLETED,
            final_answer="unused",
        ),
        session_runner=FakeSessionRunner(),
    )
    client = TestClient(create_app(service))
    pending = client.post(
        "/api/sessions/clarify/turns",
        json={"message": " "},
        headers={"Idempotency-Key": "start"},
    )
    resumed = client.post(
        "/api/sessions/clarify/resume",
        json={"answer": "具体问题"},
        headers={"Idempotency-Key": "resume"},
    )

    assert pending.json()["status"] == "waiting_for_input"
    assert resumed.status_code == 200
    assert resumed.json()["answer"] == "已恢复"
