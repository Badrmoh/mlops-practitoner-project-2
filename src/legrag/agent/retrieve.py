from typing import Any

import requests

from legrag.agent.schemas import RetrievedChunk
from legrag.store import ChromaVectorStore


def embedding_request_url(embedding_url: str) -> str:
    embedding_url = embedding_url.rstrip("/")
    if embedding_url.endswith("/api/embed") or embedding_url.endswith("/embeddings"):
        return embedding_url
    return f"{embedding_url}/embeddings"


def parse_embedding_response(body: dict[str, Any]) -> list[float]:
    if "embeddings" in body:
        embedding = body["embeddings"][0]
    elif "data" in body:
        embedding = body["data"][0]["embedding"]
    else:
        raise ValueError("Embedding response did not include 'embeddings' or 'data'")

    return [float(value) for value in embedding]


class LegalRetriever:
    def __init__(
        self,
        collection_name: str,
        persist_path: str,
        embedding_url: str,
        embedding_model: str,
        embedding_api_key: str = "",
    ) -> None:
        self._embedding_url = embedding_request_url(embedding_url)
        self._embedding_model = embedding_model
        self._embedding_api_key = embedding_api_key
        self._vector_store = ChromaVectorStore(collection_name, persist_path)

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        distance_threshold: float | None = None,
    ) -> list[RetrievedChunk]:
        embedding = self._embed_query(query)
        results = self._vector_store.search_with_metadata(embedding, n_results=top_k)
        chunks = [self._to_retrieved_chunk(result) for result in results]
        if distance_threshold is None:
            return chunks
        return [
            chunk for chunk in chunks
            if chunk.distance is not None and chunk.distance <= distance_threshold
        ]

    def _embed_query(self, query: str) -> list[float]:
        response = requests.post(
            self._embedding_url,
            headers={"Authorization": f"Bearer {self._embedding_api_key}"} if self._embedding_api_key else None,
            json={"model": self._embedding_model.removeprefix("ollama:"), "input": query},
            timeout=60,
        )
        response.raise_for_status()
        return parse_embedding_response(response.json())

    def _to_retrieved_chunk(self, result: dict[str, Any]) -> RetrievedChunk:
        metadata = result["metadata"]
        referenced_articles = metadata.get("referenced_articles") or []
        if referenced_articles == "none":
            referenced_articles = []
        return RetrievedChunk(
            text=result["text"],
            article_number=metadata.get("article_number"),
            citation=metadata.get("citation"),
            language=metadata.get("language"),
            chunk_id=metadata.get("chunk_id") or result.get("id"),
            referenced_articles=referenced_articles,
            distance=result.get("distance"),
            metadata=metadata,
        )
