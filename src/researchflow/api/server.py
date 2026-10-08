"""Local server assembly for the ResearchFlow web console."""

from collections.abc import Callable
from pathlib import Path

from researchflow.api.app import create_app
from researchflow.application.service import ResearchService
from researchflow.domain import AgentState


def create_local_app(
    database: Path,
    workflow: Callable[[str, str], AgentState],
):
    return create_app(ResearchService(database, workflow=workflow))
