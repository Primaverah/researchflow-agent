"""Offline tests for persisted LangGraph research sessions."""

from pathlib import Path

from researchflow.agent.session import (
    LangGraphSessionRunner,
    SessionCatalog,
    SessionResearchRequest,
    SessionResearchResult,
)


def test_catalog_exact_delete_and_session_isolation(tmp_path: Path) -> None:
    catalog = SessionCatalog(tmp_path / "checkpoints.sqlite3")
    catalog.touch("one")
    catalog.touch("one-extra")

    assert [item.session_id for item in catalog.list()] == ["one", "one-extra"]
    assert catalog.delete("one") is True
    assert [item.session_id for item in catalog.list()] == ["one-extra"]


def test_research_result_preserves_graph_budget_terminal_metadata() -> None:
    result = SessionResearchResult.from_events(
        "Agent 已达到最大步骤限制，研究流程已停止。",
        [
            (
                "run_completed",
                {"end_reason": "max_steps", "node_steps": 1, "max_steps": 1},
            )
        ],
    )

    assert result.end_reason == "max_steps"
    assert result.node_steps == 1
    assert result.max_steps == 1


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


def test_compiler_follow_up_uses_resolved_subject_not_previous_question(
    tmp_path: Path,
) -> None:
    seen: list[str] = []
    runner = LangGraphSessionRunner(
        tmp_path / "checkpoints.sqlite3",
        research=lambda query: seen.append(query) or query,
    )

    runner.chat("C和C++有什么区别", session_id="languages")
    runner.chat("它们都用什么编译器", session_id="languages")

    state = runner.get_state("languages")
    assert state["resolved_subject"] == "C和C++"
    assert seen[-1] == "C和C++都使用哪些常见编译器"


def test_person_follow_ups_keep_subject_and_change_required_facets(
    tmp_path: Path,
) -> None:
    seen: list[str] = []
    runner = LangGraphSessionRunner(
        tmp_path / "checkpoints.sqlite3",
        research=lambda query: seen.append(query) or query,
    )

    runner.chat("成龙是谁", session_id="jackie")
    runner.chat("他的代表作有哪些", session_id="jackie")
    works_state = runner.get_state("jackie")
    runner.chat("他现在多大了", session_id="jackie")
    age_state = runner.get_state("jackie")

    assert works_state["resolved_subject"] == "成龙"
    assert works_state["intent"] == "representative_works"
    assert works_state["required_facets"] == ["representative_works"]
    assert age_state["standalone_query"] == "成龙的出生日期及截至当前日期的年龄"
    assert age_state["intent"] == "age"
    assert age_state["required_facets"] == ["birth_date", "age"]
    assert seen[-1] == "成龙的出生日期及截至当前日期的年龄"


def test_sessions_do_not_share_resolved_subject_or_messages(tmp_path: Path) -> None:
    runner = LangGraphSessionRunner(
        tmp_path / "checkpoints.sqlite3", research=lambda query: query
    )

    runner.chat("重返未来1999是个什么游戏", session_id="game")
    runner.chat("成龙是谁", session_id="person")

    game = runner.get_state("game")
    person = runner.get_state("person")
    assert game["resolved_subject"] == "重返未来1999"
    assert person["resolved_subject"] == "成龙"
    assert [message["content"] for message in game["messages"]] != [
        message["content"] for message in person["messages"]
    ]


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


def test_language_only_follow_up_reuses_prior_research_and_passes_context(
    tmp_path: Path,
) -> None:
    requests: list[SessionResearchRequest] = []
    runner = LangGraphSessionRunner(
        tmp_path / "checkpoints.sqlite3",
        research=lambda query: query,
        research_with_context=lambda request: (
            requests.append(request)
            or f"answer:{request.answer_target}:{request.answer_language}"
        ),
    )

    runner.chat("什么是RAG？帮我查找3篇相关论文", session_id="rag")
    result = runner.chat("用中文回答", session_id="rag")

    state = runner.get_state("rag")
    assert state["current_input"] == "用中文回答"
    assert state["resolved_subject"] == "RAG"
    assert state["standalone_query"] == "什么是RAG？帮我查找3篇相关论文"
    assert state["answer_target"] == "什么是RAG？帮我查找3篇相关论文"
    assert state["answer_language"] == "zh"
    assert requests[-1].current_input == "用中文回答"
    assert requests[-1].standalone_query == "什么是RAG？帮我查找3篇相关论文"
    assert requests[-1].answer_target == "什么是RAG？帮我查找3篇相关论文"
    assert requests[-1].answer_language == "zh"
    assert result.response.endswith(":zh")
