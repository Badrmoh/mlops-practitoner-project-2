from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="LEGRAG_",
        env_file=REPO_ROOT / ".env",
        extra="ignore",
    )
    # Preprocess defaults
    raw_pdf_url: str = "https://sadanykhalifa.com/uploads/Laws/1576751803.pdf"
    raw_pdf_path: Path = REPO_ROOT / "data/raw/law.pdf"
    docling_json_path: Path = REPO_ROOT / "data/interim/law.docling.json"
    articles_json_path: Path = REPO_ROOT / "data/output/articles.json"

    # Embedding / vector store
    embedding_model: str = "qwen3-embedding:4b"
    embedding_base_url: str = "http://localhost:11434/api/embed"
    embedding_api_key: str = ""
    collection_name: str = "egyptian_civil_code"
    chroma_persist_path: Path = REPO_ROOT / "data/chroma"

    # LLM / judge provider
    llm_base_url: str = "http://localhost:11434/v1"
    llm_model: str = "qwen3.5:4b"
    llm_api_key: str = "ollama"
    retrieval_top_k: int = 27
    reports_dir: Path = REPO_ROOT / "reports"

    # Chunking defaults
    chunk_overlap_tokens: int = 20
    approx_chars_per_token: float = 5.0
    max_tokens_per_chunk: int = 512
