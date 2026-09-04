from __future__ import annotations

import argparse
import json
import sys
import time
import unicodedata
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, TypeAdapter, ValidationError

from agentic_rag.cli import build_application
from agentic_rag.config import Settings
from agentic_rag.models import ChunkId, QueryText, WorkflowResult
from agentic_rag.workflow import invoke_workflow

DEFAULT_CASES_PATH = Path("evals/cases.json")
DEFAULT_OUTPUT_PATH = Path("evals/results/latest.json")
RequiredPhrase = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class EvalConfigurationError(ValueError):
    """Raised when the live-evaluation case file is invalid."""


class EvalCase(BaseModel):
    """One expected behavior for the live agent workflow."""

    model_config = ConfigDict(extra="forbid")

    id: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=100)]
    query: QueryText
    expected_chunk_ids: list[ChunkId] = Field(default_factory=list)
    must_include: list[RequiredPhrase] = Field(default_factory=list)
    expected_insufficient_context: bool


def normalize_for_match(value: str) -> str:
    """Normalize harmless punctuation and whitespace differences for phrase checks."""

    normalized = unicodedata.normalize("NFKC", value)
    normalized = normalized.translate(
        str.maketrans(
            {
                "\u2018": "'",
                "\u2019": "'",
                "\u2010": "-",
                "\u2011": "-",
                "\u2013": "-",
                "\u2014": "-",
            }
        )
    )
    return " ".join(normalized.casefold().split())


def load_cases(path: Path) -> list[EvalCase]:
    """Load and validate a non-empty set of uniquely named evaluation cases."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise EvalConfigurationError(f"Could not read evaluation cases: {path}") from exc
    except json.JSONDecodeError as exc:
        raise EvalConfigurationError(f"Evaluation cases are not valid JSON: {exc}") from exc

    try:
        cases = TypeAdapter(list[EvalCase]).validate_python(payload)
    except ValidationError as exc:
        raise EvalConfigurationError(f"Evaluation cases failed validation:\n{exc}") from exc

    if not cases:
        raise EvalConfigurationError("Evaluation case file must contain at least one case")

    case_ids = [case.id for case in cases]
    if len(case_ids) != len(set(case_ids)):
        raise EvalConfigurationError("Evaluation case IDs must be unique")
    return cases


def evaluate_case(case: EvalCase, result: WorkflowResult) -> list[str]:
    """Return human-readable reasons that a completed workflow result failed."""

    failures: list[str] = []
    report = result.report
    retrieved_ids = {chunk.chunk_id for chunk in result.retrieval.chunks}
    used_ids = set(report.used_chunk_ids)
    expected_ids = set(case.expected_chunk_ids)

    if report.insufficient_context != case.expected_insufficient_context:
        failures.append(
            "expected insufficient_context="
            f"{case.expected_insufficient_context}, got {report.insufficient_context}"
        )

    missing_retrieved = expected_ids - retrieved_ids
    if missing_retrieved:
        failures.append(
            "required chunks were not retrieved: " + ", ".join(sorted(missing_retrieved))
        )

    missing_used = expected_ids - used_ids
    if missing_used:
        failures.append("required chunks were not used: " + ", ".join(sorted(missing_used)))

    normalized_answer = normalize_for_match(report.answer)
    missing_phrases = [
        phrase
        for phrase in case.must_include
        if normalize_for_match(phrase) not in normalized_answer
    ]
    if missing_phrases:
        failures.append("required phrases were absent: " + ", ".join(missing_phrases))

    return failures


def result_record(
    case: EvalCase,
    result: WorkflowResult,
    failures: list[str],
    duration_seconds: float,
) -> dict[str, Any]:
    """Create an inspectable, secret-free record for one completed case."""

    return {
        "id": case.id,
        "query": case.query,
        "passed": not failures,
        "duration_seconds": round(duration_seconds, 3),
        "failures": failures,
        "answer": result.report.answer,
        "insufficient_context": result.report.insufficient_context,
        "expected_chunk_ids": case.expected_chunk_ids,
        "used_chunk_ids": result.report.used_chunk_ids,
        "retrieved_chunks": [
            {"chunk_id": chunk.chunk_id, "score": round(chunk.score, 6)}
            for chunk in result.retrieval.chunks
        ],
    }


def run_live_evaluations(
    cases_path: Path,
    output_path: Path,
    *,
    show_answers: bool,
) -> int:
    """Run every case through one shared live workflow and write a JSON report."""

    cases = load_cases(cases_path)
    settings = Settings()
    graph = build_application(settings)
    records: list[dict[str, Any]] = []

    print(f"Running {len(cases)} live evaluations (external API calls are enabled).")
    for case in cases:
        started = time.perf_counter()
        try:
            result = invoke_workflow(graph, case.query)
            duration = time.perf_counter() - started
            failures = evaluate_case(case, result)
            record = result_record(case, result, failures, duration)
        except Exception as exc:  # Continue so one provider failure does not hide later cases.
            duration = time.perf_counter() - started
            record = {
                "id": case.id,
                "query": case.query,
                "passed": False,
                "duration_seconds": round(duration, 3),
                "failures": [f"{type(exc).__name__}: {exc}"],
            }

        records.append(record)
        status = "PASS" if record["passed"] else "FAIL"
        print(f"{status} {case.id} ({record['duration_seconds']:.3f}s)")
        for failure in record["failures"]:
            print(f"  - {failure}")
        if show_answers and "answer" in record:
            print("  Answer:")
            for line in str(record["answer"]).splitlines():
                print(f"    {line}")

    passed = sum(bool(record["passed"]) for record in records)
    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "cases_path": str(cases_path),
        "model": settings.openai_model,
        "embedding_model": settings.openai_embedding_model,
        "passed": passed,
        "failed": len(records) - passed,
        "total": len(records),
        "results": records,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(f"\n{passed}/{len(records)} live evaluations passed.")
    print(f"Detailed report: {output_path}")
    return 0 if passed == len(records) else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run opt-in live evaluations against the configured agent workflow."
    )
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--show-answers", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return run_live_evaluations(args.cases, args.output, show_answers=args.show_answers)
    except (EvalConfigurationError, ValidationError) as exc:
        print(f"Live evaluation could not start: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
