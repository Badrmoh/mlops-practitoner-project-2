import json
from dataclasses import asdict

import yaml

from legrag.config import REPO_ROOT, Settings
from legrag.indexing.chunk import Chunker


def main() -> None:
    with (REPO_ROOT / "dvc/params.yaml").open("r", encoding="utf-8") as fp:
        params = yaml.safe_load(fp) or {}

    chunk_params = params["chunk"]
    with (REPO_ROOT / chunk_params["articles_json_path"]).open("r", encoding="utf-8") as fp:
        articles: list[dict] = json.load(fp)

    settings = Settings(
        max_tokens_per_chunk=chunk_params["max_tokens_per_chunk"],
        chunk_overlap_tokens=chunk_params["chunk_overlap_tokens"],
    )

    chunks = Chunker(settings).create_chunks(articles)

    with (REPO_ROOT / chunk_params["chunks_jsonl_path"]).open("w", encoding="utf-8") as fp:
        for chunk in chunks:
            fp.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
