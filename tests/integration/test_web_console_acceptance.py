"""Offline acceptance coverage for the local API and durable console state."""

from fastapi.testclient import TestClient

from researchflow.agent.session import LangGraphSessionRunner
from researchflow.api.app import create_app
from researchflow.application.models import RunStatus
from researchflow.application.service import ResearchService
from researchflow.domain import AgentState, AgentStatus


def _unused_workflow(message: str, _session_id: str) -> AgentState:
    return AgentState(
        run_id="unused",
        query=message,
        status=AgentStatus.COMPLETED,
        final_answer="unused",
    )


def test_console_api_recovers_interrupted_session_after_runner_restart(
    tmp_path,
) -> None:
    database = tmp_path / "sessions" / "checkpoints.sqlite3"
    first_runner = LangGraphSessionRunner(database, research=lambda _query: "unused")
    first_client = TestClient(
        create_app(
            ResearchService(
                database, workflow=_unused_workflow, session_runner=first_runner
            )
        )
    )

    pending = first_client.post(
        "/api/sessions/acceptance/resume",
        json={"answer": "not yet"},
        headers={"Idempotency-Key": "wrong-order"},
    )
    started = first_client.post(
        "/api/sessions/acceptance/turns",
        json={"message": " "},
        headers={"Idempotency-Key": "start"},
    )
    first_runner.close()

    calls: list[str] = []
    second_runner = LangGraphSessionRunner(
        database, research=lambda query: calls.append(query) or "grounded answer"
    )
    second_client = TestClient(
        create_app(
            ResearchService(
                database, workflow=_unused_workflow, session_runner=second_runner
            )
        )
    )
    resumed = second_client.post(
        "/api/sessions/acceptance/resume",
        json={"answer": "具体研究问题"},
        headers={"Idempotency-Key": "resume"},
    )
    snapshot = second_client.get(f"/api/runs/{resumed.json()['run_id']}")
    second_runner.close()

    assert pending.status_code == 409
    assert started.json()["status"] == "waiting_for_input"
    assert resumed.status_code == 200
    assert snapshot.json()["status"] == "completed"
    assert snapshot.json()["answer"] == "grounded answer"
    assert calls == ["具体研究问题"]


def test_console_api_never_marks_empty_evidence_as_completed(tmp_path) -> None:
    def insufficient_workflow(message: str, _session_id: str, publish) -> AgentState:
        publish(
            "evidence_assessed",
            {"status": "insufficient", "gaps": ["no_successful_relevant_source"]},
        )
        return AgentState(
            run_id="empty-evidence",
            query=message,
            status=AgentStatus.COMPLETED,
            final_answer="没有成功读取任何候选来源。",
        )

    service = ResearchService(
        tmp_path / "checkpoints.sqlite3", workflow=insufficient_workflow
    )
    client = TestClient(create_app(service))

    response = client.post(
        "/api/sessions/empty/turns",
        json={"message": "question"},
        headers={"Idempotency-Key": "empty"},
    )

    assert response.status_code == 202
    assert response.json()["status"] == RunStatus.INSUFFICIENT_EVIDENCE.value
    assert response.json()["evidence"]["read_sources"] == []
    assert response.json()["evidence_status"] == "insufficient"
    assert response.json()["generation_mode"] != "llm_grounded"


def test_session_snapshot_exposes_multiple_persisted_turns(tmp_path) -> None:
    service = ResearchService(
        tmp_path / "checkpoints.sqlite3",
        workflow=lambda message, _session_id: AgentState(
            run_id=f"run-{message}",
            query=message,
            status=AgentStatus.COMPLETED,
            final_answer=f"answer for {message}",
        ),
    )
    client = TestClient(create_app(service))

    for key, message in (("one", "first"), ("two", "second")):
        response = client.post(
            "/api/sessions/history/turns",
            json={"message": message},
            headers={"Idempotency-Key": key},
        )
        assert response.status_code == 202

    snapshot = client.get("/api/sessions/history")

    assert snapshot.status_code == 200
    assert [run["question"] for run in snapshot.json()["runs"]] == [
        "first",
        "second",
    ]
