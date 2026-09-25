"""Ports: the interfaces services depends on, implemented by adapters.

Using typing.Protocol (Structural typing) rather than ABCs. Services
declare "I need something with this shape" and adapters or test fakes
satisfy it without any inheritance relationship. This is the Dependency
Inversion (DI) half of SOLID: services import only from this module,
never from app.adapters.*
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Protocol
from datetime import datetime

from app.domain.models import PaperRecord, RetrievedChunk


class PaperSource(Protocol):
    """Fetches raw paper records from an external source (e.g. arXiv)."""
    def fetch(self, categories: list[str], max_results: int, page_size: int) -> Iterable[PaperRecord]:
        """Yield PaperRecords, newest-updated first, self-throttled internally"""
        ...


class PaperRepository(Protocol):
    """Persistence boundary for papers/authors/categories."""
    def upsert_papers(self, records: Iterable[PaperRecord]) -> "UpsertResult":
        ...

    def get_papers_needing_embedding(self, model: str) -> list[tuple[int, str, str]]:
        """Return (paper_id, arxiv_id, content_hash) for papers with no 
        embedding row, or whose stored hash/model differs the current one.
        """
        ...

    def mark_embedded(self, paper_id: int, content_hash: str, model: str) -> None:
        ...

    def get_papers_by_ids(self, paper_ids: list[int]) -> dict[int, "PaperView"]:
        ...

    def wipe(self) -> None:
        ...


class Embedder(Protocol):
    dim: int
    model_name: str

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        ...

    def embed_query(self, text: str) -> list[float]:
        ...


class VectorStore(Protocol):
    def add(self, ids: list[int], vectors: list[list[float]]) -> None:
        ...

    def remove(self, ids: list[int]) -> None:
        ...

    def search(self, vector: list[float], top_k: int) -> list[tuple[int, float]]:
        """Return [(id, similarity_score), ...] sorted best-first"""
        ...

    def save(self) -> None:
        ...

    def reload_if_stale(self) -> None:
        """Reload from disk if the underlying file changed 
        since the last load"""
        ...


class LLMClient(Protocol):
    def complete_json(self, system: str, user: str) -> str:
        """Return the raw text of the model's response 
        (expected to be JSON)"""
        ...


class Reranker(Protocol):
    """Cross-encoder reranking: scores (query, candidate) pairs directly,
    more accurate than embedding-similarity alone but too slow to run over
    the whole corpus - hence applied only to the small candidate set the
    vector store already returned."""

    def rerank(self, query: str, candidates: list[RetrievedChunk]) -> list[tuple[RetrievedChunk, float]]:
        """Return (chunk, score) pairs sorted best-first."""
        ...


# Value objects shared by above ports
@dataclass(frozen=True)
class UpsertResult:
    inserted: int
    updated: int
    unchanged: int
    fetched: int


@dataclass(frozen=True)
class PaperView:
    id: int
    arxiv_id: str
    title: str
    abstract: str
    authors: list[str]
    url: str
    published_at: datetime
