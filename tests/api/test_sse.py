from fastapi.testclient import TestClient

from researchflow.api.app import create_app
from researchflow.application.models import StartTurn
from researchflow.application.service import ResearchService
from researchflow.domain import AgentState, AgentStatus


def test_sse_replays_events_after_last_event_id(tmp_path) -> None:
    service = ResearchService(
        tmp_path / "checkpoints.sqlite3",
        workflow=lambda message, _: AgentState(
            run_id="workflow-run",
            query=message,
            status=AgentStatus.COMPLETED,
            final_answer="verified answer",
        ),
    )
    snapshot = service.start_turn(
        StartTurn(session_id="one", message="question", idempotency_key="key")
    )
    client = TestClient(create_app(service))

    response = client.get(
        f"/api/runs/{snapshot.run_id}/events", headers={"Last-Event-ID": "1"}
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "id: 2" in response.text
    assert "event: run_completed" in response.text
    assert "id: 1" not in response.text
