"""Persisted LangGraph chat sessions with a deliberately small state schema."""

import sqlite3
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any, TypedDict

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from pydantic import Field

from researchflow.domain.models import DomainModel


def _append_messages(
    existing: list[dict[str, str]], updates: list[dict[str, str]]
) -> list[dict[str, str]]:
    """Append turn deltas; only the compactor may explicitly replace history."""
    if updates and updates[0].get("__replace__") == "true":
        return updates[1:]
    return [*existing, *updates]


class SessionMessage(DomainModel):
    role: str
    content: str


class SessionGraphState(DomainModel):
    """JSON-only persisted fields; runtime services never enter this model."""

    session_id: str
    messages: list[SessionMessage] = Field(default_factory=list)
    summary: str = ""
    current_input: str = ""
    resolved_subject: str = ""
    intent: str = ""
    standalone_query: str = ""
    required_facets: list[str] = Field(default_factory=list)
    rewritten_query: str = ""
    prior_query: str = ""
    turn_count: int = 0
    interrupt_status: str = ""
    response: str = ""


class SessionRecord(DomainModel):
    session_id: str
    updated_at: datetime


class SessionCatalog:
    """Small exact-match catalog kept beside LangGraph's checkpoint tables."""

    def __init__(self, database: Path) -> None:
        database.parent.mkdir(parents=True, exist_ok=True)
        self._database = database
        with self._connect() as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS session_catalog "
                "(session_id TEXT PRIMARY KEY, updated_at TEXT NOT NULL)"
            )

    def touch(self, session_id: str) -> None:
        now = datetime.now(UTC).isoformat()
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO session_catalog(session_id, updated_at) VALUES (?, ?) "
                "ON CONFLICT(session_id) DO UPDATE SET updated_at=excluded.updated_at",
                (session_id, now),
            )

    def list(self) -> list[SessionRecord]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT session_id, updated_at FROM session_catalog ORDER BY session_id"
            ).fetchall()
        return [SessionRecord(session_id=row[0], updated_at=row[1]) for row in rows]

    def delete(self, session_id: str) -> bool:
        with self._connect() as connection:
            deleted = connection.execute(
                "DELETE FROM session_catalog WHERE session_id = ?", (session_id,)
            ).rowcount
            for table in ("checkpoints", "writes"):
                exists = connection.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' AND name = ?",
                    (table,),
                ).fetchone()
                if exists:
                    connection.execute(
                        f"DELETE FROM {table} WHERE thread_id = ?", (session_id,)
                    )
        return deleted > 0

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._database)


class ChatResult(DomainModel):
    response: str = ""
    interrupted: bool = False
    prompt: str | None = None


class SessionStateData(TypedDict, total=False):
    session_id: str
    messages: Annotated[list[dict[str, str]], _append_messages]
    summary: str
    current_input: str
    resolved_subject: str
    intent: str
    standalone_query: str
    required_facets: list[str]
    rewritten_query: str
    prior_query: str
    turn_count: int
    interrupt_status: str
    response: str


class LangGraphSessionRunner:
    """A checkpointed chat graph whose only mutable data is JSON-safe state."""

    def __init__(self, database: Path, research: Callable[[str], str | None]) -> None:
        self._database = database
        self._catalog = SessionCatalog(database)
        self._research = research
        self._connection = sqlite3.connect(database, check_same_thread=False)
        self._checkpointer = SqliteSaver(
            self._connection, serde=JsonPlusSerializer(pickle_fallback=False)
        )
        self._graph = self._build_graph()

    def chat(self, message: str, *, session_id: str) -> ChatResult:
        self._catalog.touch(session_id)
        result = self._graph.invoke(
            {
                "session_id": session_id,
                "messages": [{"role": "user", "content": message}],
                "current_input": message,
            },
            config={"configurable": {"thread_id": session_id}},
        )
        return self._result(result)

    def resume(self, session_id: str, answer: str) -> ChatResult:
        self._catalog.touch(session_id)
        result = self._graph.invoke(
            Command(resume=answer), config={"configurable": {"thread_id": session_id}}
        )
        return self._result(result)

    def get_state(self, session_id: str) -> dict[str, Any]:
        snapshot = self._graph.get_state({"configurable": {"thread_id": session_id}})
        return dict(snapshot.values)

    def close(self) -> None:
        self._connection.close()

    def _build_graph(self):
        graph = StateGraph(SessionStateData)
        graph.add_node("load_context", self._load_context)
        graph.add_node("contextualize", self._contextualize)
        graph.add_node("clarify", self._clarify)
        graph.add_node("research", self._research_node)
        graph.add_node("respond", self._respond)
        graph.add_node("compact_history", self._compact_history)
        graph.add_edge(START, "load_context")
        graph.add_edge("load_context", "contextualize")
        graph.add_edge("contextualize", "clarify")
        graph.add_edge("clarify", "research")
        graph.add_edge("research", "respond")
        graph.add_edge("respond", "compact_history")
        graph.add_edge("compact_history", END)
        return graph.compile(checkpointer=self._checkpointer)

    @staticmethod
    def _load_context(_: dict[str, Any]) -> dict[str, Any]:
        return {}

    @staticmethod
    def _contextualize(state: dict[str, Any]) -> dict[str, Any]:
        latest = state.get("current_input", "").strip()
        subject = state.get("resolved_subject", "").strip()
        intent = LangGraphSessionRunner._intent(latest)
        if not subject:
            subject = LangGraphSessionRunner._explicit_subject(
                state.get("prior_query", "").strip()
            )
        explicit_subject = (
            ""
            if intent == "continuation" or LangGraphSessionRunner._has_reference(latest)
            else LangGraphSessionRunner._explicit_subject(latest)
        )
        if explicit_subject:
            subject = explicit_subject

        if LangGraphSessionRunner._has_reference(latest) and subject:
            standalone = LangGraphSessionRunner._replace_reference(latest, subject)
        else:
            standalone = latest

        if intent == "compiler" and subject:
            standalone = f"{subject}都使用哪些常见编译器"
            facets = ["common_compilers"]
        elif intent == "representative_works" and subject:
            standalone = f"{subject}的代表电影作品有哪些"
            facets = ["representative_works"]
        elif intent == "age" and subject:
            standalone = f"{subject}的出生日期及截至当前日期的年龄".strip()
            facets = ["birth_date", "age"]
        elif intent == "latest_update" and subject:
            standalone = f"{subject}最近什么时候更新"
            facets = ["latest_update"]
        elif intent == "continuation" and subject:
            standalone = f"{subject}更多信息"
            facets = []
        else:
            facets = []
        return {
            "resolved_subject": subject,
            "intent": intent,
            "standalone_query": standalone,
            "rewritten_query": standalone,
            "required_facets": facets,
            "turn_count": state.get("turn_count", 0) + 1,
            "interrupt_status": "awaiting_clarification" if not standalone else "",
        }

    @staticmethod
    def _explicit_subject(query: str) -> str:
        for suffix in ("有什么区别", "是谁", "是个什么游戏", "是什么游戏"):
            if query.endswith(suffix):
                return query.removesuffix(suffix).strip(" ，。？?")
        if query and not any(
            token in query for token in ("什么", "哪些", "怎么", "吗", "？", "?")
        ):
            return query
        return ""

    @staticmethod
    def _intent(query: str) -> str:
        if "编译器" in query:
            return "compiler"
        if "代表作" in query:
            return "representative_works"
        if "多大" in query or "年龄" in query:
            return "age"
        if "更新" in query:
            return "latest_update"
        if "区别" in query:
            return "comparison"
        if query.casefold() in {"more", "continue", "继续", "更多"}:
            return "continuation"
        return "overview"

    @staticmethod
    def _has_reference(query: str) -> bool:
        return any(
            token in query
            for token in (
                "它们",
                "他们",
                "她们",
                "两者",
                "该项目",
                "这个游戏",
                "它",
                "他",
                "她",
            )
        )

    @staticmethod
    def _replace_reference(query: str, subject: str) -> str:
        for token in (
            "它们",
            "他们",
            "她们",
            "两者",
            "该项目",
            "这个游戏",
            "它",
            "他",
            "她",
        ):
            if token in query:
                return query.replace(token, subject)
        return query

    @staticmethod
    def _clarify(state: dict[str, Any]) -> dict[str, Any]:
        if state.get("rewritten_query", "").strip():
            return {}
        answer = interrupt("请提供需要研究的具体问题。")
        query = answer.strip()
        return {
            "messages": [{"role": "user", "content": answer}],
            "current_input": query,
            "standalone_query": query,
            "rewritten_query": query,
            "interrupt_status": "resumed",
        }

    def _research_node(self, state: dict[str, Any]) -> dict[str, Any]:
        answer = self._research(state["standalone_query"])
        return {"response": answer or "未找到相关文档。"}

    @staticmethod
    def _respond(state: dict[str, Any]) -> dict[str, Any]:
        return {
            "messages": [{"role": "assistant", "content": state["response"]}]
        }

    @staticmethod
    def _compact_history(state: dict[str, Any]) -> dict[str, Any]:
        messages = state.get("messages", [])
        if len(messages) <= 12:
            return {}
        older = messages[:-6]
        summary = "\n".join(f"{item['role']}: {item['content']}" for item in older)[
            -4000:
        ]
        return {
            "summary": summary,
            "messages": [{"__replace__": "true"}, *messages[-6:]],
        }

    @staticmethod
    def _result(result: dict[str, Any]) -> ChatResult:
        interrupts = result.get("__interrupt__", [])
        if interrupts:
            value = interrupts[0].value
            return ChatResult(interrupted=True, prompt=str(value))
        return ChatResult(response=result.get("response", ""))
