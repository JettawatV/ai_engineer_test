import re
import unicodedata
from math import isfinite, sqrt
from pathlib import Path

from langchain_core.embeddings import Embeddings

from .models import KnowledgeChunk, RetrievalResult, RetrievedChunk

CHUNK_HEADER = re.compile(
    r"^\[(?P<chunk_id>[A-Z0-9][A-Z0-9_-]{1,63})(?::[^\]]+)?\]\s*(?P<inline>.*)$"
)


class KnowledgeBaseError(ValueError):
    pass


def normalize_text(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def load_knowledge_chunks(path: Path) -> list[KnowledgeChunk]:
    """Load validated knowledge chunks from a UTF-8 text file.

    Plain paragraphs receive deterministic IDs based on document order. Explicit
    bracketed IDs remain supported when stable, human-readable IDs are preferred.

    Raises:
        KnowledgeBaseError: If the file is missing, unreadable, empty, or malformed.
    """

    if not path.is_file():
        raise KnowledgeBaseError(f"Knowledge base file was not found: {path}")

    try:
        raw_text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise KnowledgeBaseError("Knowledge base must be valid UTF-8 text") from exc
    except OSError as exc:
        raise KnowledgeBaseError(f"Knowledge base could not be read: {path}") from exc

    chunks: list[KnowledgeChunk] = []
    seen_ids: set[str] = set()
    seen_text: set[str] = set()
    pending_heading: str | None = None
    auto_id = 1

    for block in re.split(r"\n\s*\n", raw_text.replace("\r\n", "\n")):
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if not lines or all(line.startswith("#") for line in lines):
            continue

        match = CHUNK_HEADER.match(lines[0])
        if match:
            chunk_id = match.group("chunk_id")
            content_lines = [match.group("inline"), *lines[1:]]
        else:
            content_lines = [line for line in lines if not line.startswith("#")]
            if len(content_lines) == 1 and content_lines[0].endswith(":"):
                pending_heading = content_lines[0]
                continue

            while f"KB-{auto_id:03d}" in seen_ids:
                auto_id += 1
            chunk_id = f"KB-{auto_id:03d}"
            auto_id += 1

        body_lines = [line for line in content_lines if line and not line.startswith("#")]
        if pending_heading:
            body_lines.insert(0, pending_heading)
            pending_heading = None
        body = "\n".join(body_lines).strip()

        if not body:
            raise KnowledgeBaseError(f"Knowledge chunk {chunk_id} has no content")
        if chunk_id in seen_ids:
            raise KnowledgeBaseError(f"Duplicate knowledge chunk ID: {chunk_id}")
        if normalize_text(body) in seen_text:
            raise KnowledgeBaseError(f"Duplicate knowledge content found at chunk {chunk_id}")

        chunks.append(KnowledgeChunk(chunk_id=chunk_id, text=body, source=path.name))
        seen_ids.add(chunk_id)
        seen_text.add(normalize_text(body))

    if pending_heading:
        raise KnowledgeBaseError(f"Knowledge heading has no following content: {pending_heading}")
    if not chunks:
        raise KnowledgeBaseError("Knowledge base contains no searchable paragraphs.")
    return chunks


def _validate_vector(vector: list[float], *, label: str) -> tuple[float, ...]:
    if not vector:
        raise KnowledgeBaseError(f"{label} embedding is empty")

    values = tuple(float(value) for value in vector)
    if not all(isfinite(value) for value in values):
        raise KnowledgeBaseError(f"{label} embedding contains a non-finite value")
    if not any(values):
        raise KnowledgeBaseError(f"{label} embedding has zero magnitude")
    return values


def cosine_similarity(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    """Calculate cosine similarity for two validated embedding vectors."""

    if len(left) != len(right):
        raise KnowledgeBaseError("Embedding vectors have inconsistent dimensions")

    dot_product = sum(a * b for a, b in zip(left, right, strict=True))
    left_magnitude = sqrt(sum(value * value for value in left))
    right_magnitude = sqrt(sum(value * value for value in right))
    similarity = dot_product / (left_magnitude * right_magnitude)
    return max(-1.0, min(1.0, similarity))


class SemanticRetriever:
    """Rank local knowledge chunks by embedding cosine similarity."""

    def __init__(self, chunks: list[KnowledgeChunk], embeddings: Embeddings) -> None:
        if not chunks:
            raise KnowledgeBaseError("At least one knowledge chunk is required")

        self._chunks = list(chunks)
        self._embeddings = embeddings
        self._document_vectors: list[tuple[float, ...]] | None = None

    def _get_document_vectors(self) -> list[tuple[float, ...]]:
        if self._document_vectors is not None:
            return self._document_vectors

        raw_vectors = self._embeddings.embed_documents([chunk.text for chunk in self._chunks])
        if len(raw_vectors) != len(self._chunks):
            raise KnowledgeBaseError("Embedding provider returned an unexpected vector count")

        vectors = [
            _validate_vector(vector, label=f"Knowledge chunk {chunk.chunk_id}")
            for chunk, vector in zip(self._chunks, raw_vectors, strict=True)
        ]
        dimensions = {len(vector) for vector in vectors}
        if len(dimensions) != 1:
            raise KnowledgeBaseError("Knowledge embeddings have inconsistent dimensions")

        self._document_vectors = vectors
        return vectors

    def search(self, query: str, *, min_score: float) -> RetrievalResult:
        """Embed one query and return all chunks above the similarity floor."""

        cleaned_query = " ".join(query.split())
        if not cleaned_query:
            return RetrievalResult(search_query=cleaned_query or query, chunks=[])

        query_vector = _validate_vector(self._embeddings.embed_query(cleaned_query), label="Query")
        document_vectors = self._get_document_vectors()
        ranked = sorted(
            (
                (chunk, cosine_similarity(query_vector, vector))
                for chunk, vector in zip(self._chunks, document_vectors, strict=True)
            ),
            key=lambda item: (-item[1], item[0].chunk_id),
        )
        chunks = [
            RetrievedChunk(
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                source=chunk.source,
                score=score,
            )
            for chunk, score in ranked
            if score > min_score
        ]
        return RetrievalResult(search_query=cleaned_query, chunks=chunks)
