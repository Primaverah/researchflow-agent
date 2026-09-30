"""Offline tests for persisted LangGraph research sessions."""

from pathlib import Path

from researchflow.agent.session import LangGraphSessionRunner, SessionCatalog


def test_catalog_exact_delete_and_session_isolation(tmp_path: Path) -> None:
    catalog = SessionCatalog(tmp_path / "checkpoints.sqlite3")
    catalog.touch("one")
    catalog.touch("one-extra")

    assert [item.session_id for item in catalog.list()] == ["one", "one-extra"]
    assert catalog.delete("one") is True
    assert [item.session_id for item in catalog.list()] == ["one-extra"]


def test_context_is_preserved_and_history_is_compacted(tmp_path: Path) -> None:
    runner = LangGraphSessionRunner(
        tmp_path / "checkpoints.sqlite3", research=lambda q: f"answer:{q}"
    )
    for index in range(13):
        result = runner.chat("topic" if index == 0 else "more", session_id="s1")

    state = runner.get_state("s1")
    assert result.response.startswith("answer:")
    assert state["summary"]
    assert len(state["messages"]) <= 12
    assert "topic" in state["rewritten_query"]


def test_follow_up_is_rewritten_with_the_previous_question(tmp_path: Path) -> None:
    seen: list[str] = []
    runner = LangGraphSessionRunner(
        tmp_path / "checkpoints.sqlite3", research=seen.append
    )

    runner.chat("agent tools", session_id="follow-up")
    runner.chat("more", session_id="follow-up")

    assert "agent tools" in seen[-1]


def test_cross_runner_follow_up_keeps_the_subject(tmp_path: Path) -> None:
    database = tmp_path / "checkpoints.sqlite3"
    LangGraphSessionRunner(database, research=lambda query: "first").chat(
        "重返未来1999是个什么游戏", session_id="cross-follow-up"
    )
    seen: list[str] = []
    LangGraphSessionRunner(database, research=seen.append).chat(
        "它最近什么时候更新", session_id="cross-follow-up"
    )

    assert "重返未来" in seen[-1]


def test_contextualizer_keeps_current_input_and_resolves_person_pronouns() -> None:
    works = LangGraphSessionRunner._contextualize(
        {
            "current_input": "他的代表作有哪些",
            "prior_query": "成龙是谁",
            "messages": [],
        }
    )
    age = LangGraphSessionRunner._contextualize(
        {
            "current_input": "他现在多大了",
            "prior_query": "成龙是谁",
            "messages": [],
        }
    )

    assert works["standalone_query"] == "成龙的代表电影作品有哪些"
    assert works["required_facets"] == ["representative_works"]
    assert age["standalone_query"] == "成龙的出生日期及截至当前日期的年龄"
    assert age["required_facets"] == ["birth_date", "age"]


def test_clarification_interrupt_can_resume_without_research(tmp_path: Path) -> None:
    calls: list[str] = []
    runner = LangGraphSessionRunner(
        tmp_path / "checkpoints.sqlite3", research=calls.append
    )

    pending = runner.chat("", session_id="clarify")
    assert pending.interrupted is True
    assert calls == []
    resumed = runner.resume("clarify", "research local tools")

    assert resumed.interrupted is False
    assert calls == ["research local tools"]


def test_checkpoint_can_be_loaded_by_a_new_runner(tmp_path: Path) -> None:
    database = tmp_path / "checkpoints.sqlite3"
    LangGraphSessionRunner(database, research=lambda q: f"answer:{q}").chat(
        "first", session_id="persist"
    )

    state = LangGraphSessionRunner(
        database, research=lambda q: f"answer:{q}"
    ).get_state("persist")
    assert state["messages"][0]["content"] == "first"
