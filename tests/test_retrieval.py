from pathlib import Path

import pytest
from langchain_core.embeddings import Embeddings

from agentic_rag.models import KnowledgeChunk
from agentic_rag.retrieval import (
    KnowledgeBaseError,
    SemanticRetriever,
    load_knowledge_chunks,
    normalize_text,
)

FIXTURE = Path("tests/fixtures/sample_travel_policy.txt")


def make_retriever(embeddings: Embeddings) -> SemanticRetriever:
    return SemanticRetriever(load_knowledge_chunks(FIXTURE), embeddings)


def test_normalizes_text() -> None:
    assert normalize_text("  TRAVEL\u00a0Policy  ") == "travel policy"


def test_loads_stable_policy_chunks() -> None:
    chunks = load_knowledge_chunks(FIXTURE)

    assert len(chunks) == 5
    assert chunks[0].chunk_id == "TEST-TRAVEL-001"
    assert chunks[0].source == FIXTURE.name


def test_plain_paragraphs_receive_deterministic_ids(tmp_path: Path) -> None:
    path = tmp_path / "plain.txt"
    path.write_text(
        "# Source: public example\n\nQualifications:\n\n"
        "Master's degree required.\nThree years of experience required.\n\n"
        "The role is based in Bangkok.\n",
        encoding="utf-8",
    )

    chunks = load_knowledge_chunks(path)

    assert [chunk.chunk_id for chunk in chunks] == ["KB-001", "KB-002"]
    assert chunks[0].text == (
        "Qualifications:\nMaster's degree required.\nThree years of experience required."
    )
    assert chunks[1].text == "The role is based in Bangkok."


def test_production_knowledge_base_is_loadable() -> None:
    chunks = load_knowledge_chunks(Path("knowledge_base.txt"))

    assert chunks
    assert chunks[0].chunk_id == "KB-001"


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("# metadata only\n", "no searchable paragraphs"),
        ("[POLICY-001]\nFirst.\n\n[POLICY-001]\nSecond.\n", "Duplicate knowledge chunk ID"),
        ("Qualifications:\n", "heading has no following content"),
    ],
)
def test_rejects_invalid_knowledge_bases(tmp_path: Path, content: str, message: str) -> None:
    path = tmp_path / "invalid.txt"
    path.write_text(content, encoding="utf-8")

    with pytest.raises(KnowledgeBaseError, match=message):
        load_knowledge_chunks(path)


def test_semantic_search_handles_a_paraphrased_query(travel_embeddings) -> None:
    result = make_retriever(travel_embeddings).search(
        "Who authorizes overseas trips?", min_score=0.3
    )

    assert result.chunks[0].chunk_id == "TEST-TRAVEL-001"
    assert len(result.chunks) == 1


def test_returns_no_chunks_for_unrelated_query(travel_embeddings) -> None:
    result = make_retriever(travel_embeddings).search(
        "What is the password rotation frequency?", min_score=0.3
    )

    assert result.chunks == []


def test_embeds_documents_once_but_embeds_every_query(travel_embeddings) -> None:
    retriever = make_retriever(travel_embeddings)

    retriever.search("Who authorizes overseas trips?", min_score=0.3)
    retriever.search("When must expenses be submitted?", min_score=0.3)

    assert travel_embeddings.document_calls == 1
    assert travel_embeddings.query_calls == [
        "Who authorizes overseas trips?",
        "When must expenses be submitted?",
    ]


def test_returns_every_match_above_similarity_floor() -> None:
    chunks = [
        KnowledgeChunk(chunk_id=f"MATCH-{index}", text=text, source="fixture.txt")
        for index, text in enumerate(["first", "second", "third"], start=1)
    ]

    class ControlledEmbeddings(Embeddings):
        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return [[1.0, 0.0], [0.8, 0.6], [0.2, 0.98]]

        def embed_query(self, text: str) -> list[float]:
            return [1.0, 0.0]

    result = SemanticRetriever(chunks, ControlledEmbeddings()).search("query", min_score=0.5)

    assert [chunk.chunk_id for chunk in result.chunks] == ["MATCH-1", "MATCH-2"]
