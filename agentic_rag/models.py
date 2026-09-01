from math import isfinite
from typing import Annotated

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

QueryText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=1_000)]
AnswerText = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=8_000)
]
ChunkId = Annotated[str, StringConstraints(pattern=r"^[A-Z0-9][A-Z0-9_-]{1,63}$")]


class AppModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class UserQuery(AppModel):
    query: QueryText


class KnowledgeChunk(AppModel):
    chunk_id: ChunkId
    text: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=12_000)]
    source: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=255)]


class RetrievedChunk(KnowledgeChunk):
    score: float = Field(ge=0.0)

    @field_validator("score")
    @classmethod
    def score_must_be_finite(cls, value: float) -> float:
        if not isfinite(value):
            raise ValueError("score must be finite")
        return value


class RetrievalResult(AppModel):
    search_query: QueryText
    chunks: list[RetrievedChunk] = Field(default_factory=list, max_length=10)

    @model_validator(mode="after")
    def chunk_ids_must_be_unique(self) -> "RetrievalResult":
        chunk_ids = [chunk.chunk_id for chunk in self.chunks]
        if len(chunk_ids) != len(set(chunk_ids)):
            raise ValueError("retrieval result contains duplicate chunk IDs")
        return self


class GeneratedReport(AppModel):
    answer: AnswerText
    used_chunk_ids: list[ChunkId] = Field(default_factory=list, max_length=10)
    insufficient_context: bool

    @model_validator(mode="after")
    def evidence_status_must_be_consistent(self) -> "GeneratedReport":
        if len(self.used_chunk_ids) != len(set(self.used_chunk_ids)):
            raise ValueError("used_chunk_ids must not contain duplicates")
        if self.insufficient_context and self.used_chunk_ids:
            raise ValueError("an insufficient-context report cannot claim supporting chunks")
        if not self.insufficient_context and not self.used_chunk_ids:
            raise ValueError("a grounded report must identify at least one supporting chunk")
        return self


class WorkflowResult(AppModel):
    query: QueryText
    retrieval: RetrievalResult
    report: GeneratedReport
