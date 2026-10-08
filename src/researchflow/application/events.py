"""Safe, persisted events for research-run observers."""

import re
from collections.abc import Mapping

from researchflow.application.models import EventEnvelope
from researchflow.application.run_store import SqliteRunStore

_SECRET = re.compile(r"(?i)(bearer\s+|sk-)[A-Za-z0-9_-]+")
_DROP_KEYS = frozenset({"content", "prompt", "api_key", "authorization"})


class EventBroker:
    """Assign per-run sequence IDs before exposing sanitized event metadata."""

    def __init__(self, store: SqliteRunStore) -> None:
        self._store = store

    def publish(
        self,
        *,
        run_id: str,
        session_id: str,
        event_type: str,
        data: Mapping[str, object] | None = None,
    ) -> EventEnvelope:
        return self._store.append_event(
            EventEnvelope(
                event_id=0,
                run_id=run_id,
                session_id=session_id,
                type=event_type,
                data=self._sanitize_mapping(data or {}),
            )
        )

    def replay(self, run_id: str, *, after_event_id: int = 0) -> list[EventEnvelope]:
        return self._store.list_events(run_id, after_event_id=after_event_id)

    @classmethod
    def _sanitize_mapping(cls, data: Mapping[str, object]) -> dict[str, object]:
        return {
            key: cls._sanitize(value)
            for key, value in data.items()
            if key.casefold() not in _DROP_KEYS
        }

    @classmethod
    def _sanitize(cls, value: object) -> object:
        if isinstance(value, str):
            return _SECRET.sub(r"\1[REDACTED]", value)
        if isinstance(value, Mapping):
            return cls._sanitize_mapping(value)
        if isinstance(value, list):
            return [cls._sanitize(item) for item in value]
        if isinstance(value, (int, float, bool)) or value is None:
            return value
        return str(value)
