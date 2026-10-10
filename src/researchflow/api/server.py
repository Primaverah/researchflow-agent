"""Local server assembly for the ResearchFlow web console."""

from collections.abc import Callable
from pathlib import Path

from researchflow.api.app import create_app
from researchflow.application.service import ResearchService
from researchflow.domain import AgentState


def create_local_app(
    database: Path,
    workflow: Callable[[str, str], AgentState],
    *,
    session_runner: object | None = None,
):
    return create_app(
        ResearchService(database, workflow=workflow, session_runner=session_runner)
    )
