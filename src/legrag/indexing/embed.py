import logging

import requests
from pydantic_ai import Embedder as pyem
from pydantic_ai.embeddings.openai import OpenAIEmbeddingModel
from pydantic_ai.providers.openai import OpenAIProvider

from legrag.indexing.chunk import Chunk
from legrag.store import ChromaVectorStore

_log = logging.getLogger(__name__)


def openai_embedding_base_url(embedding_url: str) -> str:
    return embedding_url.removesuffix("/embeddings").rstrip("/")


class Embedder:
    def __init__(
        self,
        collection_name: str,
        persist_path: str,
        embedding_url: str,
        embedding_model: str,
        embedding_api_key: str = "",
    ):
        self._vector_store = ChromaVectorStore(collection_name, persist_path)
        self._embedding_model = embedding_model
        self._embedding_api_key = embedding_api_key
        self._uses_ollama_embed_api = "/api/embed" in embedding_url
        self._embedding_url = (
            embedding_url.rstrip("/") + "/api/embed"
            if self._uses_ollama_embed_api and "/api/embed" not in embedding_url
            else embedding_url
        )
        self._pyem: pyem | None = None

        if self._uses_ollama_embed_api:
            return
        provider = OpenAIProvider(
            base_url=openai_embedding_base_url(embedding_url),
            api_key=embedding_api_key,
        )
        self._pyem = pyem(model=OpenAIEmbeddingModel(embedding_model, provider=provider))

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if self._uses_ollama_embed_api:
            response = requests.post(
                self._embedding_url,
                headers={"Authorization": f"Bearer {self._embedding_api_key}"} if self._embedding_api_key else None,
                json={"model": self._embedding_model.removeprefix("ollama:"), "input": texts},
                timeout=120,
            )
            response.raise_for_status()
            return [[float(value) for value in embedding] for embedding in response.json()["embeddings"]]

        assert self._pyem is not None
        embeddings = self._pyem.embed_documents_sync(texts)
        return [list(embedding) for embedding in embeddings.embeddings]

    def store_in_batches(self, chunks: list[Chunk],
                        batch_size: int = 128):
        _log.info(f"Storing {len(chunks)} chunks in batches of {batch_size}")
        for start in range(0, len(chunks), batch_size):
            batch = []
            for b in chunks[start:start + batch_size]:
                if b.text:
                    batch.append(b)
            batch_embeddings = self.embed_batch([chunk.text for chunk in batch])
            batch_texts = [chunk.text for chunk in batch]
            batch_metadatas = [chunk.metadata for chunk in batch]
            batch_ids = [m["chunk_id"] for m in batch_metadatas]
            _log.info(f"Storing batch {start // batch_size + 1}")
            self._vector_store.upsert(
                ids=batch_ids,
                texts=batch_texts,
                embeddings=batch_embeddings,
                metadatas=batch_metadatas
            )

    def reset_collection(self) -> None:
        self._vector_store.reset_collection()
