from pathlib import Path

import pytest
from langchain_core.messages import AIMessage

from agentic_rag.agents import (
    REPORT_PROMPT,
    AgentOutputError,
    DataRetrieverAgent,
    ReportGeneratorAgent,
    build_search_tool,
)
from agentic_rag.models import GeneratedReport, RetrievalResult, RetrievedChunk, UserQuery


class FakeToolCallingModel:
    def __init__(self, response: AIMessage) -> None:
        self.response = response
        self.bound_kwargs: dict | None = None

    def bind_tools(self, tools, **kwargs):
        self.bound_kwargs = {"tools": tools, **kwargs}
        return self

    def invoke(self, messages):
        assert len(messages) == 2
        return self.response


class FakeStructuredModel:
    def __init__(self, response) -> None:
        self.response = response
        self.structured_kwargs: dict | None = None

    def with_structured_output(self, schema, **kwargs):
        self.structured_kwargs = {"schema": schema, **kwargs}
        return self

    def invoke(self, messages):
        assert len(messages) == 2
        return self.response


def make_search_tool(embeddings):
    return build_search_tool(
        Path("tests/fixtures/sample_travel_policy.txt"), embeddings, min_score=0.3
    )


def make_retrieval() -> RetrievalResult:
    return RetrievalResult(
        search_query="hotel receipts",
        chunks=[
            RetrievedChunk(
                chunk_id="POLICY-001",
                text="Hotel claims require an itemized receipt.",
                source="fixture.txt",
                score=0.82,
            )
        ],
    )


def test_search_tool_returns_validated_contract(travel_embeddings) -> None:
    raw_result = make_search_tool(travel_embeddings).invoke(
        {"query": "When must expenses be submitted?"}
    )
    result = RetrievalResult.model_validate(raw_result)

    assert result.chunks[0].chunk_id == "TEST-TRAVEL-003"


def test_data_retriever_forces_and_executes_approved_tool(travel_embeddings) -> None:
    model = FakeToolCallingModel(
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "search_knowledge_base",
                    "args": {"query": "Who approves international travel?"},
                    "id": "call-1",
                    "type": "tool_call",
                }
            ],
        )
    )
    agent = DataRetrieverAgent(model, make_search_tool(travel_embeddings))

    result = agent.retrieve(UserQuery(query="Who approves international travel?"))
    agent.retrieve(UserQuery(query="Who approves international travel?"))

    assert result.chunks[0].chunk_id == "TEST-TRAVEL-001"
    assert travel_embeddings.document_calls == 1
    assert travel_embeddings.query_calls == [
        "Who approves international travel?",
        "Who approves international travel?",
    ]
    assert model.bound_kwargs["tool_choice"] == "search_knowledge_base"
    assert model.bound_kwargs["parallel_tool_calls"] is False


def test_data_retriever_rejects_missing_tool_call(travel_embeddings) -> None:
    agent = DataRetrieverAgent(
        FakeToolCallingModel(AIMessage(content="No tool")),
        make_search_tool(travel_embeddings),
    )

    with pytest.raises(AgentOutputError, match="exactly one tool call"):
        agent.retrieve(UserQuery(query="Who approves travel?"))


def test_report_generator_returns_grounded_report() -> None:
    expected = GeneratedReport(
        answer="Hotel claims require an itemized receipt.",
        used_chunk_ids=["POLICY-001"],
        insufficient_context=False,
    )
    model = FakeStructuredModel(expected)

    result = ReportGeneratorAgent(model).generate(
        UserQuery(query="Do hotel claims need receipts?"), make_retrieval()
    )

    assert result == expected
    assert model.structured_kwargs["schema"] is GeneratedReport
    assert model.structured_kwargs["method"] == "json_schema"


def test_report_prompt_matches_required_output_quality() -> None:
    for requirement in ("comprehensive", "cohesive", "non-redundant", "well-formatted"):
        assert requirement in REPORT_PROMPT


def test_report_generator_rejects_unknown_source() -> None:
    model = FakeStructuredModel(
        GeneratedReport(
            answer="Receipts are required.",
            used_chunk_ids=["POLICY-999"],
            insufficient_context=False,
        )
    )

    with pytest.raises(AgentOutputError, match="unknown chunk IDs"):
        ReportGeneratorAgent(model).generate(
            UserQuery(query="Are receipts required?"), make_retrieval()
        )
