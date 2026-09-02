import pytest
from langchain_core.embeddings import Embeddings


class TravelSemanticEmbeddings(Embeddings):
    """Small deterministic embedding double for the travel-policy fixture."""

    def __init__(self) -> None:
        self.document_calls = 0
        self.query_calls: list[str] = []

    @staticmethod
    def _embed(text: str) -> list[float]:
        normalized = text.casefold()
        if any(
            phrase in normalized
            for phrase in (
                "division head",
                "international travel",
                "authorizes overseas",
            )
        ):
            return [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        if any(phrase in normalized for phrase in ("corporate travel provider", "flights")):
            return [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
        if any(phrase in normalized for phrase in ("30 calendar days", "expenses")):
            return [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
        if any(phrase in normalized for phrase in ("itemized receipt", "receipts")):
            return [0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
        if any(phrase in normalized for phrase in ("public transport", "taxis")):
            return [0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        return [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.document_calls += 1
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        self.query_calls.append(text)
        return self._embed(text)


@pytest.fixture
def travel_embeddings() -> TravelSemanticEmbeddings:
    return TravelSemanticEmbeddings()
