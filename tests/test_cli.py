from pathlib import Path

import pytest
from pydantic import ValidationError

from agentic_rag import cli
from agentic_rag.config import Settings
from agentic_rag.models import (
    GeneratedReport,
    RetrievalResult,
    RetrievedChunk,
    UserQuery,
    WorkflowResult,
)
from agentic_rag.retrieval import KnowledgeBaseError


def make_result() -> WorkflowResult:
    return WorkflowResult(
        query="Are receipts required?",
        retrieval=RetrievalResult(
            search_query="Are receipts required?",
            chunks=[
                RetrievedChunk(
                    chunk_id="POLICY-001",
                    text="Receipts are required.",
                    source="fixture.txt",
                    score=0.82,
                )
            ],
        ),
        report=GeneratedReport(
            answer="Receipts are required.",
            used_chunk_ids=["POLICY-001"],
            insufficient_context=False,
        ),
    )


def test_models_strip_text_and_enforce_evidence_rules() -> None:
    assert UserQuery(query="  What is allowed?  ").query == "What is allowed?"

    with pytest.raises(ValidationError):
        UserQuery(query="   ")
    with pytest.raises(ValidationError, match="at least one supporting chunk"):
        GeneratedReport(answer="Travel is allowed.", used_chunk_ids=[], insufficient_context=False)
    with pytest.raises(ValidationError, match="cannot claim supporting chunks"):
        GeneratedReport(
            answer="There is not enough information.",
            used_chunk_ids=["POLICY-001"],
            insufficient_context=True,
        )


def test_settings_keep_api_key_secret() -> None:
    settings = Settings(
        openai_api_key="sk-test-value",
        knowledge_base_path=Path("tests/fixtures/sample_travel_policy.txt"),
    )

    assert settings.openai_model == "gpt-5-mini"
    assert settings.openai_embedding_model == "text-embedding-3-small"
    assert "sk-test-value" not in repr(settings)


def test_print_result_can_show_sources(capsys) -> None:
    cli.print_result(make_result(), show_sources=True)

    output = capsys.readouterr().out
    assert "Receipts are required." in output
    assert "POLICY-001 | score=0.8200 | used" in output


def test_build_application_constructs_graph_without_api_call() -> None:
    settings = Settings(
        openai_api_key="sk-test-value",
        knowledge_base_path=Path("tests/fixtures/sample_travel_policy.txt"),
    )

    graph = cli.build_application(settings)

    assert "data_retriever_agent" in graph.get_graph().nodes
    assert "report_generator_agent" in graph.get_graph().nodes


def test_main_runs_one_query_with_injected_dependencies(monkeypatch) -> None:
    calls: list[tuple] = []
    monkeypatch.setattr(cli, "Settings", lambda: object())
    monkeypatch.setattr(cli, "build_application", lambda settings: "compiled-graph")
    monkeypatch.setattr(
        cli,
        "ask_once",
        lambda graph, query, show_sources: calls.append((graph, query, show_sources)),
    )

    exit_code = cli.main(["Are", "receipts", "required?", "--show-sources"])

    assert exit_code == 0
    assert calls == [("compiled-graph", "Are receipts required?", True)]


def test_main_returns_safe_error_code(monkeypatch) -> None:
    monkeypatch.setattr(cli, "Settings", lambda: object())

    def fail_to_build(settings):
        raise KnowledgeBaseError("No policy chunks")

    monkeypatch.setattr(cli, "build_application", fail_to_build)

    assert cli.main(["Question"]) == 2
