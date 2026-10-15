import json
from dataclasses import asdict

import yaml

from legrag.config import REPO_ROOT, Settings
from legrag.indexing.embed import Embedder
from legrag.indexing.chunk import Chunk

def main():
    with (REPO_ROOT / "dvc/params.yaml").open("r") as fp:
        params = yaml.safe_load(fp)
    settings = Settings()
    collection_name = params["embed"]["collection_name"]
    chroma_persist_path = params["embed"]["chroma_persist_path"]
    embedding_model = params["embed"]["embedding_model"]
    embedding_base_url = settings.embedding_base_url
    embedding_api_key = settings.embedding_api_key

    embedder = Embedder(
        collection_name,
        chroma_persist_path,
        embedding_base_url,
        embedding_model,
        embedding_api_key,
    )

    with (REPO_ROOT / params["embed"]["chunks_jsonl_path"]).open("r", encoding="utf-8") as fp:
        chunks: list[Chunk] = []
        texts: list[str] = []
        metadatas: list[dict[str, str | int]] = []
        ids: list[str] = []
        for chunk in fp.readlines():
            chunks.append(Chunk(**json.loads(chunk)))

    if len(chunks) > 0:
        embedder.reset_collection()
        embedder.store_in_batches(chunks)

if __name__ == "__main__":
    main()
