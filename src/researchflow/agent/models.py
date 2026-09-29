"""Small internal action types used by the rule-driven agent."""

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class AgentActionType(StrEnum):
    """Actions understood by the first rule-driven runner."""

    SEARCH = "search"
    READ = "read"
    WEB_SEARCH = "web_search"
    FETCH_URL = "fetch_url"
    SUMMARIZE = "summarize"
    SAVE = "save"
    FINISH = "finish"


@dataclass(frozen=True, slots=True)
class AgentAction:
    """One deterministic selector decision."""

    action_type: AgentActionType
    tool_name: str | None = None
    arguments: dict[str, Any] = field(default_factory=dict)
