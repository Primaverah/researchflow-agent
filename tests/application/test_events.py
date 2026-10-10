from researchflow.application.events import EventBroker
from researchflow.application.run_store import SqliteRunStore


def test_event_broker_persists_ordered_safe_events(tmp_path) -> None:
    store = SqliteRunStore(tmp_path / "checkpoints.sqlite3")
    broker = EventBroker(store)

    first = broker.publish(
        run_id="run-1",
        session_id="session-1",
        event_type="source_read",
        data={
            "source_id": "https://example.test/article",
            "content_length": 42,
            "content": "this must never leave the process",
        },
    )
    second = broker.publish(
        run_id="run-1",
        session_id="session-1",
        event_type="evidence_assessed",
        data={"status": "sufficient"},
    )

    assert first.event_id == 1
    assert second.event_id == 2
    assert "content" not in first.data
    assert store.list_events("run-1", after_event_id=0) == [first, second]


def test_event_broker_redacts_secret_like_event_values(tmp_path) -> None:
    broker = EventBroker(SqliteRunStore(tmp_path / "checkpoints.sqlite3"))

    event = broker.publish(
        run_id="run-1",
        session_id="session-1",
        event_type="run_failed",
        data={"diagnostic": "Authorization: Bearer sk-not-a-real-key"},
    )

    assert "sk-not-a-real-key" not in str(event.data)


def test_event_subscription_replays_then_receives_new_events(tmp_path) -> None:
    broker = EventBroker(SqliteRunStore(tmp_path / "checkpoints.sqlite3"))
    broker.publish(
        run_id="run-1",
        session_id="session-1",
        event_type="run_started",
    )
    subscription = broker.subscribe("run-1", after_event_id=0, keepalive_seconds=0.01)

    assert next(subscription).type == "run_started"
    broker.publish(
        run_id="run-1",
        session_id="session-1",
        event_type="search_completed",
        data={"candidate_count": 2},
    )

    event = next(subscription)
    assert event is not None
    assert event.type == "search_completed"
    subscription.close()
