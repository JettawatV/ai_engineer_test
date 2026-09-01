from agentic_rag.models import GeneratedReport, RetrievalResult, RetrievedChunk, UserQuery
from agentic_rag.workflow import build_workflow, invoke_workflow


class FakeRetrieverAgent:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def retrieve(self, user_query: UserQuery) -> RetrievalResult:
        self.events.append("retrieve")
        return RetrievalResult(
            search_query=user_query.query,
            chunks=[
                RetrievedChunk(
                    chunk_id="POLICY-001",
                    text="International travel requires division-head approval.",
                    source="fixture.txt",
                    score=2.5,
                )
            ],
        )


class FakeReportAgent:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def generate(self, user_query: UserQuery, retrieval: RetrievalResult) -> GeneratedReport:
        self.events.append("report")
        assert user_query.query == "What approval is required?"
        assert retrieval.chunks[0].chunk_id == "POLICY-001"
        return GeneratedReport(
            answer="International travel requires division-head approval.",
            used_chunk_ids=["POLICY-001"],
            insufficient_context=False,
        )


def test_workflow_runs_agents_in_required_sequence() -> None:
    events: list[str] = []
    graph = build_workflow(FakeRetrieverAgent(events), FakeReportAgent(events))

    result = invoke_workflow(graph, "What approval is required?")

    assert events == ["retrieve", "report"]
    assert result.report.used_chunk_ids == ["POLICY-001"]
