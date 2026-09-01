import re
import unicodedata
from pathlib import Path
from rank_bm25 import BM25Okapi
from .models import KnowledgeChunk, RetrievalResult, RetrievedChunk

TOKEN_PATTERN = re.compile(r"[^\W_]+(?:'[^\W_]+)?", flags=re.UNICODE)
CHUNK_HEADER = re.compile(
    r"^\[(?P<chunk_id>[A-Z0-9][A-Z0-9_-]{1,63})(?::[^\]]+)?\]\s*(?P<inline>.*)$"
)
ENGLISH_FUNCTION_WORDS = frozenset(
    "a an and are as at be by for from has have how in is it of on or that the this to was "
    "were what when where which who why with".split()
)


class KnowledgeBaseError(ValueError):
    pass


def normalize_text(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def tokenize(text: str) -> list[str]:
    return [
        token
        for token in TOKEN_PATTERN.findall(normalize_text(text))
        if token not in ENGLISH_FUNCTION_WORDS
    ]


def load_knowledge_chunks(path: Path) -> list[KnowledgeChunk]:
    """Load validated knowledge chunks from a UTF-8 text file.
    Chunks must be separated by blank lines and begin with a stable bracketed ID.
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

    for block in re.split(r"\n\s*\n", raw_text.replace("\r\n", "\n")):
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if not lines or all(line.startswith("#") for line in lines):
            continue

        match = CHUNK_HEADER.match(lines[0])
        if match is None:
            raise KnowledgeBaseError(
                "Every knowledge chunk must start with an ID such as [POLICY-001: SHORT TITLE]"
            )

        chunk_id = match.group("chunk_id")
        body = " ".join(
            line
            for line in [match.group("inline"), *lines[1:]]
            if line and not line.startswith("#")
        )
        body = " ".join(body.split())

        if not body:
            raise KnowledgeBaseError(f"Knowledge chunk {chunk_id} has no content")
        if chunk_id in seen_ids:
            raise KnowledgeBaseError(f"Duplicate knowledge chunk ID: {chunk_id}")
        if normalize_text(body) in seen_text:
            raise KnowledgeBaseError(f"Duplicate knowledge content found at chunk {chunk_id}")

        chunks.append(KnowledgeChunk(chunk_id=chunk_id, text=body, source=path.name))
        seen_ids.add(chunk_id)
        seen_text.add(normalize_text(body))

    if not chunks:
        raise KnowledgeBaseError(
            "Knowledge base contains no policy chunks. Add content using the documented format."
        )
    return chunks


class BM25Retriever:
    """Rank local knowledge chunks using deterministic lexical search."""

    def __init__(self, chunks: list[KnowledgeChunk]) -> None:
        if not chunks:
            raise KnowledgeBaseError("At least one knowledge chunk is required")

        tokenized_corpus = [tokenize(chunk.text) for chunk in chunks]
        empty_ids = [
            chunk.chunk_id
            for chunk, tokens in zip(chunks, tokenized_corpus, strict=True)
            if not tokens
        ]
        if empty_ids:
            raise KnowledgeBaseError(
                f"Knowledge chunks contain no searchable terms: {', '.join(empty_ids)}"
            )

        self._chunks = list(chunks)
        self._index = BM25Okapi(tokenized_corpus)

    def search(self, query: str, *, top_k: int, min_score: float) -> RetrievalResult:
        """Return the highest-scoring chunks above the configured score floor."""

        cleaned_query = " ".join(query.split())
        query_tokens = tokenize(cleaned_query)
        if not query_tokens:
            return RetrievalResult(search_query=cleaned_query or query, chunks=[])

        ranked = sorted(
            zip(self._chunks, self._index.get_scores(query_tokens), strict=True),
            key=lambda item: (-float(item[1]), item[0].chunk_id),
        )
        chunks = [
            RetrievedChunk(
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                source=chunk.source,
                score=max(0.0, float(score)),
            )
            for chunk, score in ranked
            if float(score) > min_score
        ][:top_k]
        return RetrievalResult(search_query=cleaned_query, chunks=chunks)
