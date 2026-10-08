"""SQLite storage for application-level research run metadata."""

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from researchflow.application.models import RunSnapshot, RunStatus


def _utc_now() -> datetime:
    return datetime.now(UTC)


class SqliteRunStore:
    """Store run metadata without modifying LangGraph-owned checkpoint tables."""

    def __init__(self, database: Path) -> None:
        database.parent.mkdir(parents=True, exist_ok=True)
        self._database = database
        self._initialize()

    def create_run(self, session_id: str, idempotency_key: str) -> RunSnapshot:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT run_id FROM researchflow_idempotency "
                "WHERE session_id = ? AND idempotency_key = ?",
                (session_id, idempotency_key),
            ).fetchone()
            if row is not None:
                return self._get(connection, row[0])
            now = _utc_now()
            snapshot = RunSnapshot(
                run_id=uuid4().hex,
                session_id=session_id,
                status=RunStatus.CREATED,
                created_at=now,
                updated_at=now,
            )
            connection.execute(
                "INSERT INTO researchflow_runs "
                "(run_id, session_id, status, created_at, updated_at, payload) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    snapshot.run_id,
                    session_id,
                    snapshot.status.value,
                    now.isoformat(),
                    now.isoformat(),
                    snapshot.model_dump_json(),
                ),
            )
            connection.execute(
                "INSERT INTO researchflow_idempotency "
                "(session_id, idempotency_key, run_id) VALUES (?, ?, ?)",
                (session_id, idempotency_key, snapshot.run_id),
            )
            return snapshot

    def list_session_runs(self, session_id: str) -> list[RunSnapshot]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT run_id FROM researchflow_runs "
                "WHERE session_id = ? ORDER BY created_at",
                (session_id,),
            ).fetchall()
            return [self._get(connection, row[0]) for row in rows]

    def get_run(self, run_id: str) -> RunSnapshot:
        with self._connect() as connection:
            return self._get(connection, run_id)

    def save(self, snapshot: RunSnapshot) -> RunSnapshot:
        updated = snapshot.model_copy(update={"updated_at": _utc_now()})
        with self._connect() as connection:
            connection.execute(
                "UPDATE researchflow_runs SET status = ?, updated_at = ?, payload = ? "
                "WHERE run_id = ?",
                (
                    updated.status.value,
                    updated.updated_at.isoformat(),
                    updated.model_dump_json(),
                    updated.run_id,
                ),
            )
        return updated

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS researchflow_runs "
                "(run_id TEXT PRIMARY KEY, session_id TEXT NOT NULL, "
                "status TEXT NOT NULL, created_at TEXT NOT NULL, "
                "updated_at TEXT NOT NULL, payload TEXT NOT NULL)"
            )
            connection.execute(
                "CREATE TABLE IF NOT EXISTS researchflow_idempotency "
                "(session_id TEXT NOT NULL, idempotency_key TEXT NOT NULL, "
                "run_id TEXT NOT NULL, PRIMARY KEY (session_id, idempotency_key))"
            )
            connection.execute(
                "CREATE TABLE IF NOT EXISTS researchflow_events "
                "(run_id TEXT NOT NULL, event_id INTEGER NOT NULL, "
                "payload TEXT NOT NULL, "
                "PRIMARY KEY (run_id, event_id))"
            )

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._database)

    @staticmethod
    def _get(connection: sqlite3.Connection, run_id: str) -> RunSnapshot:
        row = connection.execute(
            "SELECT payload FROM researchflow_runs WHERE run_id = ?", (run_id,)
        ).fetchone()
        if row is None:
            raise KeyError(run_id)
        return RunSnapshot.model_validate(json.loads(row[0]))
