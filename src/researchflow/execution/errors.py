"""Execution and trace recording exceptions."""

from researchflow.tools.errors import ResearchFlowError


class TraceRecordingError(ResearchFlowError):
    """Raised when an execution trace cannot be safely persisted."""
