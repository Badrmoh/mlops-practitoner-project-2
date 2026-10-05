from typing import Protocol

import chromadb
from chromadb.api.types import Embeddings, metadatas


class VectorStore(Protocol):
    def upsert(self,
                ids: list[str],
                texts: list[str],
                embeddings: Embeddings,
                metadatas: list[metadatas]) -> None:
        ...

    def search(
        self, query_embedding: list[float],
        n_results: int = 5) -> list[str] | None:
        ...


class ChromaVectorStore:
    def __init__(self, collection_name: str):
        self.client = chromadb.Client()
        self.collection = self.client.get_or_create_collection(collection_name)

    def upsert(self,
                ids: list[str],
                texts: list[str],
                embeddings: Embeddings,
                metadatas: list[metadatas]) -> None:
        self.collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
        )

    def search(self, query_embedding: list[float], n_results: int = 5) -> list[str] | None:
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
        )
        return results["documents"][0] if results["documents"] else None
