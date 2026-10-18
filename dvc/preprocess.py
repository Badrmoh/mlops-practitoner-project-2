from pathlib import Path

import yaml

from legrag.config import REPO_ROOT
from legrag.preprocess.processor import Preprocessor


def resolve_repo_path(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else REPO_ROOT / candidate


def main() -> None:
    with (REPO_ROOT / "dvc/params.yaml").open("r", encoding="utf-8") as fp:
        params = yaml.safe_load(fp) or {}

    preprocess_params = params["preprocess"]
    Preprocessor(
        raw_pdf_path=resolve_repo_path(preprocess_params["raw_pdf_path"]),
        docling_json_path=resolve_repo_path(preprocess_params["docling_json_path"]),
        articles_json_path=resolve_repo_path(preprocess_params["articles_json_path"]),
    ).run()


if __name__ == "__main__":
    main()
