"""Tests for web selection domain restrictions."""

from researchflow.agent.web import WebRulePlanner, WebStateSelector
from researchflow.domain import AgentState, AgentStatus, ToolResult


def test_web_selector_fetches_only_allowed_domains() -> None:
    state = AgentState(
        run_id="run-1",
        query="Python 3.13",
        status=AgentStatus.RUNNING,
        plan=WebRulePlanner().create_plan("Python 3.13"),
        tool_results=[
            ToolResult(
                call_id="search",
                tool_name="search_documents",
                success=True,
                output={"hits": []},
            ),
            ToolResult(
                call_id="web",
                tool_name="web_search",
                success=True,
                output={
                    "results": [
                        {
                            "url": "https://blog.csdn.net/x",
                            "title": "CSDN",
                            "summary": "",
                        },
                        {
                            "url": "https://docs.python.org/3/whatsnew/3.13.html",
                            "title": "Docs",
                            "summary": "",
                        },
                    ]
                },
            ),
        ],
    )

    action = WebStateSelector(allowed_domains=("docs.python.org",)).select(state)

    assert action.arguments["url"] == "https://docs.python.org/3/whatsnew/3.13.html"
