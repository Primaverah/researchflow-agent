from researchflow.application.models import RunStatus, StartTurn
from researchflow.application.service import ResearchService
from researchflow.domain import AgentState, AgentStatus


def test_start_turn_persists_completed_snapshot(tmp_path) -> None:
    def workflow(message: str, _session_id: str) -> AgentState:
        return AgentState(
            run_id="workflow-run",
            query=message,
            status=AgentStatus.COMPLETED,
            final_answer="verified answer",
        )

    service = ResearchService(tmp_path / "checkpoints.sqlite3", workflow=workflow)
    snapshot = service.start_turn(
        StartTurn(session_id="one", message="tool calling", idempotency_key="key-1")
    )

    assert snapshot.status is RunStatus.COMPLETED
    assert service.get_run(snapshot.run_id).status is RunStatus.COMPLETED


def test_duplicate_start_does_not_execute_workflow_twice(tmp_path) -> None:
    calls: list[str] = []

    def workflow(message: str, _session_id: str) -> AgentState:
        calls.append(message)
        return AgentState(
            run_id="workflow-run",
            query=message,
            status=AgentStatus.COMPLETED,
            final_answer="verified answer",
        )

    service = ResearchService(tmp_path / "checkpoints.sqlite3", workflow=workflow)
    request = StartTurn(
        session_id="one", message="tool calling", idempotency_key="key-1"
    )
    service.start_turn(request)
    service.start_turn(request)

    assert calls == ["tool calling"]
