from typing import Protocol, cast

import chromadb
from chromadb.api.types import Embeddings, Metadatas

Metadata = dict[str, str | int | bool | float | None]


class VectorStore(Protocol):
    def upsert(self,
                ids: list[str],
                texts: list[str],
                embeddings: list[list[float]],
                metadatas: list[Metadata]) -> None:
        ...

    def search(
        self, query_embedding: list[float],
        n_results: int = 5) -> list[str] | None:
        ...


class ChromaVectorStore:
    def __init__(self, collection_name: str, persist_path: str):
        self.collection_name = collection_name
        self.client = chromadb.PersistentClient(path=persist_path)
        self.collection = self.client.get_or_create_collection(collection_name)

    def reset_collection(self) -> None:
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(self.collection_name)

    def count(self) -> int:
        return self.collection.count()

    def upsert(self,
                ids: list[str],
                texts: list[str],
                embeddings: list[list[float]],
                metadatas: list[Metadata]) -> None:
        self.collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=cast(Embeddings, embeddings),
            metadatas=cast(Metadatas, metadatas),
        )

    def search(self, query_embedding: list[float], n_results: int = 5) -> list[str] | None:
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
        )
        return results["documents"][0] if results["documents"] else None

    def search_with_metadata(self, query_embedding: list[float], n_results: int = 5) -> list[dict]:
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )
        documents = results.get("documents") or [[]]
        metadatas = results.get("metadatas") or [[]]
        distances = results.get("distances") or [[]]
        ids = results.get("ids") or [[]]

        return [
            {"id": chunk_id, "text": text, "metadata": metadata or {}, "distance": distance}
            for chunk_id, text, metadata, distance in zip(ids[0], documents[0], metadatas[0], distances[0])
        ]
