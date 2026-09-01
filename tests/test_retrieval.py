from pathlib import Path

import pytest

from agentic_rag.retrieval import (
    BM25Retriever,
    KnowledgeBaseError,
    load_knowledge_chunks,
    normalize_text,
    tokenize,
)

FIXTURE = Path("tests/fixtures/sample_travel_policy.txt")


def make_retriever() -> BM25Retriever:
    return BM25Retriever(load_knowledge_chunks(FIXTURE))


def test_normalizes_and_tokenizes_english_text() -> None:
    assert normalize_text("  TRAVEL\u00a0Policy  ") == "travel policy"
    assert tokenize("Traveler's HOTEL: approved!") == ["traveler's", "hotel", "approved"]


def test_loads_stable_policy_chunks() -> None:
    chunks = load_knowledge_chunks(FIXTURE)

    assert len(chunks) == 5
    assert chunks[0].chunk_id == "TEST-TRAVEL-001"
    assert chunks[0].source == FIXTURE.name


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("# metadata only\n", "no policy chunks"),
        ("[POLICY-001]\nFirst.\n\n[POLICY-001]\nSecond.\n", "Duplicate knowledge chunk ID"),
    ],
)
def test_rejects_invalid_knowledge_bases(tmp_path: Path, content: str, message: str) -> None:
    path = tmp_path / "invalid.txt"
    path.write_text(content, encoding="utf-8")

    with pytest.raises(KnowledgeBaseError, match=message):
        load_knowledge_chunks(path)


def test_retrieves_relevant_policy_first() -> None:
    result = make_retriever().search(
        "Who must approve international business travel?", top_k=3, min_score=0.0
    )

    assert result.chunks[0].chunk_id == "TEST-TRAVEL-001"
    assert all(chunk.score > 0 for chunk in result.chunks)


def test_returns_no_chunks_for_unrelated_query() -> None:
    result = make_retriever().search(
        "What is the password rotation frequency?", top_k=3, min_score=0.0
    )

    assert result.chunks == []


def test_respects_top_k() -> None:
    result = make_retriever().search("travel hotel expenses", top_k=2, min_score=0.0)
    assert len(result.chunks) <= 2
