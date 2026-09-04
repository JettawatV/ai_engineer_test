import json
from pathlib import Path

import pytest

from agentic_rag.models import GeneratedReport, RetrievalResult, RetrievedChunk, WorkflowResult
from evals.run_live import EvalCase, EvalConfigurationError, evaluate_case, load_cases


def make_result(
    *,
    answer: str = "A Master\u2019s Degree and at least 3 years of experience are required.",
    retrieved_ids: tuple[str, ...] = ("KB-003",),
    used_ids: tuple[str, ...] = ("KB-003",),
    insufficient_context: bool = False,
) -> WorkflowResult:
    return WorkflowResult(
        query="What qualifications are required?",
        retrieval=RetrievalResult(
            search_query="What qualifications are required?",
            chunks=[
                RetrievedChunk(
                    chunk_id=chunk_id,
                    text="Relevant evidence.",
                    source="knowledge_base.txt",
                    score=0.8,
                )
                for chunk_id in retrieved_ids
            ],
        ),
        report=GeneratedReport(
            answer=answer,
            used_chunk_ids=list(used_ids),
            insufficient_context=insufficient_context,
        ),
    )


def test_evaluate_case_accepts_grounded_answer_with_normalized_punctuation() -> None:
    case = EvalCase(
        id="qualifications",
        query="What qualifications are required?",
        expected_chunk_ids=["KB-003"],
        must_include=["master's degree", "AT LEAST 3 YEARS"],
        expected_insufficient_context=False,
    )

    assert evaluate_case(case, make_result()) == []


def test_evaluate_case_reports_missing_evidence_and_phrase() -> None:
    case = EvalCase(
        id="requirements",
        query="What qualifications are required?",
        expected_chunk_ids=["KB-002", "KB-003"],
        must_include=["financial analysis"],
        expected_insufficient_context=False,
    )

    failures = evaluate_case(case, make_result())

    assert "required chunks were not retrieved: KB-002" in failures
    assert "required chunks were not used: KB-002" in failures
    assert "required phrases were absent: financial analysis" in failures


def test_evaluate_case_accepts_expected_insufficient_context() -> None:
    case = EvalCase(
        id="salary",
        query="What is the salary?",
        expected_chunk_ids=[],
        must_include=[],
        expected_insufficient_context=True,
    )
    result = make_result(
        answer="The provided information does not specify a salary.",
        retrieved_ids=(),
        used_ids=(),
        insufficient_context=True,
    )

    assert evaluate_case(case, result) == []


def test_load_cases_rejects_duplicate_ids(tmp_path) -> None:
    path = tmp_path / "cases.json"
    case = {
        "id": "duplicate",
        "query": "Question?",
        "expected_chunk_ids": [],
        "must_include": [],
        "expected_insufficient_context": True,
    }
    path.write_text(json.dumps([case, case]), encoding="utf-8")

    with pytest.raises(EvalConfigurationError, match="must be unique"):
        load_cases(path)


def test_repository_live_cases_are_valid() -> None:
    cases = load_cases(Path("evals/cases.json"))

    assert len(cases) == 6
