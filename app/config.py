"""Centralized application settings.

Single source of truth for configuration. Every adapter/service receives
value from here via dependency injection rather than reding os.environ
directly, which keeps them testable in isolation.
"""
from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Database
    database_url: str = "postgresql+psycopg://pipeline:pipeline@db:5432/pipeline"

    # arXiv ingestion
    arxiv_categories: str = "cs.AI,cs.CL,cs.LG"
    arxiv_max_papers: int = 3000
    arxiv_page_size: int = 100
    arxiv_min_request_interval_seconds: float = 3.0
    arxiv_base_url: str = "https://export.arxiv.org/api/query"

    # Embedding / Vector store
    embed_model: str = "BAAI/bge-small-en-v1.5"
    embed_dim: int = 384
    faiss_dir: str = "./data"
    faiss_index_name: str = "papers.index"

    # LLM
    llm_provider: str = "anthropic"
    llm_api_key: str = ""
    llm_model: str = "claude-sonnet-4-6"

    # RAG retrieval knobs
    retrieval_top_k: int = 20
    retrieval_top_n: int = 5
    retrieval_min_score: float = 0.35
    rag_max_context_tokens: int = 3000
    rerank_enabled: bool = False

    # App
    log_level: str = "INFO"

    @property
    def arxiv_category_list(self) -> list[str]:
        return [c.strip() for c in self.arxiv_categories.split(",") if c.strip()]


@lru_cache
def get_settings() -> Settings:
    """Cached settings singleton."""
    return Settings()
