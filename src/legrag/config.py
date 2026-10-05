from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="LEGRAG_",
        env_file=REPO_ROOT / ".env",
        extra="ignore",
    )
    raw_pdf_url: str = "https://sadanykhalifa.com/uploads/Laws/1576751803.pdf"
    raw_pdf_path: Path = REPO_ROOT / "data/raw/law.pdf"
    docling_json_path: Path = REPO_ROOT / "data/interim/law.docling.json"
    articles_json_path: Path = REPO_ROOT / "data/output/articles.json"
    embedder_model: str = "nomic-embed-text-v2-moe"
    embedder_url: str = "http://localhost:11434/api/embed"
    llm_base_url: str = "http://localhost:11434"
    chunk_overlap_tokens: int = 20
    collection_name: str = "egyptian_civil_code"
    approx_chars_per_token: float = 4.0
    max_tokens_per_chunk: int = 512
