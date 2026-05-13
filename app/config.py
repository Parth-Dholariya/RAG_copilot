import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    index_dir: Path = Path(os.getenv("INDEX_DIR", "storage/index"))
    document_dir: Path = Path(os.getenv("DOCUMENT_DIR", "data/sample_docs"))
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "900"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "160"))
    top_k: int = int(os.getenv("TOP_K", "5"))
    rerank_top_n: int = int(os.getenv("RERANK_TOP_N", "5"))
    llm_provider: str = os.getenv("LLM_PROVIDER", "extractive")
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    gemini_api_key: str | None = os.getenv("GEMINI_API_KEY")
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")


settings = Settings()
