import logging

from pydantic_ai import Embedder as pyem
from pydantic_ai.embeddings.openai import OpenAIEmbeddingModel
from pydantic_ai.providers.ollama import OllamaProvider

from legrag.indexing.chunk import Chunk
from legrag.store import ChromaVectorStore

_log = logging.getLogger(__name__)

class Embedder:
    def __init__(self, collection_name: str, persist_path: str, embedding_url: str, embedding_model: str):
        self._vector_store = ChromaVectorStore(collection_name, persist_path)
        self._embedding_url = embedding_url
        self._embedding_model = embedding_model
        model_name = embedding_model.removeprefix("ollama:")
        provider = OllamaProvider(base_url=embedding_url.removesuffix("/api/embed") + "/v1")
        self._pyem: pyem = pyem(model=OpenAIEmbeddingModel(model_name, provider=provider))

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        embeddings = self._pyem.embed_documents_sync(texts)
        return [list(embedding) for embedding in embeddings.embeddings]

    def store_in_batches(self, chunks: list[Chunk],
                        batch_size: int = 128):
        _log.info(f"Storing {len(chunks)} chunks in batches of {batch_size}")
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start:start + batch_size]
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
