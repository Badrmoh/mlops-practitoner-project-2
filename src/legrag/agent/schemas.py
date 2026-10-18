from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field


class RetrievedChunk(BaseModel):
    text: str = Field(description="Retrieved legal text chunk.")
    article_number: int | None = Field(default=None, description="Egyptian Civil Code article number for this chunk.")
    citation: str | None = Field(default=None, description="Canonical citation for this chunk.")
    language: str | None = Field(default=None, description="Chunk language, typically en or ar.")
    chunk_id: str | None = Field(default=None, description="Stable chunk identifier.")
    referenced_articles: list[int] = Field(default_factory=list, description="Article numbers referenced by this chunk.")
    distance: float | None = Field(default=None, description="Vector distance returned by retrieval; lower is more similar.")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Original vector-store metadata.")


class LegalAnswer(BaseModel):
    answered: bool = Field(description="True only when retrieved articles directly answer the question.")
    answer: str = Field(description="Final user-facing answer. Do not mention retrieval, context, chunks, or excerpts.")
    cited_article_numbers: list[int] = Field(
        default_factory=list,
        description="Article numbers used to support the answer. Must come from retrieved chunks.",
    )
    citations: list[str] = Field(default_factory=list, description="Canonical citations generated from cited_article_numbers.")
    contexts: list[RetrievedChunk] = Field(default_factory=list, description="Retrieved chunks used by the agent.")


@dataclass
class RagDeps:
    retriever: Any
    top_k: int = 5
    distance_threshold: float | None = None
    retrieved_contexts: list[RetrievedChunk] = field(default_factory=list)
