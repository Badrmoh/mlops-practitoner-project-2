from dataclasses import dataclass

from legrag.store import VectorStore


@dataclass
class Deps:
    vector_store: VectorStore
