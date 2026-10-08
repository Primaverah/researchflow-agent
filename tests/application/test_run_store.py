from researchflow.application.models import RunStatus
from researchflow.application.run_store import SqliteRunStore


def test_run_store_returns_same_run_for_same_session_and_idempotency_key(
    tmp_path,
) -> None:
    store = SqliteRunStore(tmp_path / "checkpoints.sqlite3")

    first = store.create_run("session-a", "key-1")
    second = store.create_run("session-a", "key-1")

    assert second.run_id == first.run_id
    assert second.status is RunStatus.CREATED


def test_run_store_isolates_session_ids(tmp_path) -> None:
    store = SqliteRunStore(tmp_path / "checkpoints.sqlite3")
    first = store.create_run("session-a", "key-1")
    second = store.create_run("session-b", "key-1")

    assert [run.run_id for run in store.list_session_runs("session-a")] == [
        first.run_id
    ]
    assert [run.run_id for run in store.list_session_runs("session-b")] == [
        second.run_id
    ]
