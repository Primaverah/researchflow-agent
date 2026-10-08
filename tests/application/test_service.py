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
    assert snapshot.answer == "verified answer"
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


def test_start_turn_persists_lifecycle_events(tmp_path) -> None:
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
        StartTurn(session_id="one", message="tool calling", idempotency_key="key-1")
    )

    events = service.events_for_run(snapshot.run_id)
    assert [event.type for event in events] == ["run_started", "run_completed"]
    assert service.get_run(snapshot.run_id).last_event_id == events[-1].event_id


def test_published_evidence_events_update_persisted_run_snapshot(tmp_path) -> None:
    service = ResearchService(
        tmp_path / "checkpoints.sqlite3",
        workflow=lambda message, _: AgentState(
            run_id="workflow-run",
            query=message,
            status=AgentStatus.COMPLETED,
            final_answer="answer",
        ),
    )
    snapshot = service.start_turn(
        StartTurn(session_id="one", message="question", idempotency_key="key-1")
    )

    service.publish_event(
        snapshot.run_id,
        "candidate_selected",
        {
            "source_id": "https://example.test/source",
            "title": "搜索候选",
            "kind": "web",
        },
    )
    service.publish_event(
        snapshot.run_id,
        "source_read",
        {
            "source_id": "https://example.test/source",
            "title": "已读取正文",
            "kind": "web",
            "content_length": 42,
        },
    )
    updated = service.publish_event(
        snapshot.run_id,
        "evidence_assessed",
        {"status": "partial", "gaps": ["missing_date"]},
    )

    assert updated.evidence.candidates[0].title == "搜索候选"
    assert updated.evidence.read_sources[0].title == "已读取正文"
    assert updated.evidence_status == "partial"
    assert updated.evidence_gaps == ["missing_date"]
    assert service.get_run(snapshot.run_id) == updated


def test_event_aware_workflow_projects_evidence_while_it_runs(tmp_path) -> None:
    def workflow(message: str, _session_id: str, publish) -> AgentState:
        publish(
            "candidate_selected",
            {
                "source_id": "https://example.test/source",
                "title": "候选来源",
                "kind": "web",
            },
        )
        publish(
            "source_rejected",
            {
                "source_id": "https://example.test/rejected",
                "reason": "low_relevance",
            },
        )
        return AgentState(
            run_id="workflow-run",
            query=message,
            status=AgentStatus.COMPLETED,
            final_answer="answer",
        )

    service = ResearchService(tmp_path / "checkpoints.sqlite3", workflow=workflow)

    snapshot = service.start_turn(
        StartTurn(session_id="one", message="question", idempotency_key="key-1")
    )

    assert snapshot.evidence.candidates[0].title == "候选来源"
    assert snapshot.evidence.rejected_sources[0].reason == "low_relevance"
