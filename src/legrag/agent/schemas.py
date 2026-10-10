from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field


class RetrievedChunk(BaseModel):
    text: str
    article_number: int | None = None
    citation: str | None = None
    language: str | None = None
    chunk_id: str | None = None
    referenced_articles: list[int] = Field(default_factory=list)
    distance: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class LegalAnswer(BaseModel):
    answered: bool = False
    answer: str
    cited_article_numbers: list[int] = Field(default_factory=list)
    citations: list[str] = Field(default_factory=list)
    contexts: list[RetrievedChunk] = Field(default_factory=list)


@dataclass
class RagDeps:
    retriever: Any
    top_k: int = 5
    distance_threshold: float | None = None
    retrieved_contexts: list[RetrievedChunk] = field(default_factory=list)
