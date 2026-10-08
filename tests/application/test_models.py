from datetime import UTC, datetime

from researchflow.agent.graph import AgentGraphState, GraphCandidate
from researchflow.application.models import EvidenceSnapshot, RunSnapshot, RunStatus
from researchflow.domain import AgentState, AgentStatus
from researchflow.tools.offline import ReadDocumentOutput


def test_evidence_snapshot_keeps_candidates_read_and_rejected_separate() -> None:
    graph_state = AgentGraphState(
        run_id="run-1",
        query="C++ compilers",
        agent=AgentState(
            run_id="run-1",
            query="C++ compilers",
            status=AgentStatus.RUNNING,
        ),
        candidates=[
            GraphCandidate(
                source_type="web",
                locator="https://example.test/cpp",
                title="C++ compilers",
                summary="Candidate only",
            )
        ],
        documents=[
            ReadDocumentOutput(
                path="compiler.md",
                title="Compiler guide",
                content="GCC and Clang compile C++.",
                char_count=30,
            )
        ],
        rejected_sources=["https://bad.test — web_low_quality_content"],
    )

    snapshot = EvidenceSnapshot.from_graph_state(graph_state)

    assert snapshot.candidates[0].read is False
    assert snapshot.read_sources[0].source_id == "compiler.md"
    assert snapshot.rejected_sources[0].reason == "web_low_quality_content"


def test_insufficient_evidence_is_not_answered_status() -> None:
    snapshot = RunSnapshot(
        run_id="run-1",
        session_id="session-1",
        status=RunStatus.INSUFFICIENT_EVIDENCE,
        created_at=datetime(2026, 10, 8, tzinfo=UTC),
        updated_at=datetime(2026, 10, 8, tzinfo=UTC),
    )

    assert snapshot.status is not RunStatus.COMPLETED
