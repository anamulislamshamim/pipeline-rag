"""Turns a question into scored, hydrated RetrievedChunks.

Depends only on the Embedder / VectorStore / PaperRepository ports - single
responsibility is "fetch candidates", nothing about scoring thresholds or
prompt construction lives here (that's RetrievingState and ContextBuilder).
"""
from __future__ import annotations

from app.domain.models import RetrievedChunk
from app.ports import Embedder, PaperRepository, VectorStore


class Retriever:
    def __init__(self, embedder: Embedder, vector_store: VectorStore, repository: PaperRepository):
        self._embedder = embedder
        self._vector_store = vector_store
        self._repository = repository

    def retrieve(self, question: str, top_k: int) -> list[RetrievedChunk]:
        # Picks up an index rebuilt by a separate ingestion process, if any.
        self._vector_store.reload_if_stale()

        query_vector = self._embedder.embed_query(question)
        hits = self._vector_store.search(query_vector, top_k)
        if not hits:
            return []

        paper_ids = [paper_id for paper_id, _ in hits]
        views = self._repository.get_papers_by_ids(paper_ids)

        chunks: list[RetrievedChunk] = []
        for paper_id, score in hits:
            view = views.get(paper_id)
            if view is None:
                # FAISS entry pointing at a paper that's since been deleted
                # from Postgres (e.g. wiped without a matching reindex).
                # Skip defensively rather than error the whole query.
                continue
            chunks.append(
                RetrievedChunk(
                    paper_id=view.id,
                    arxiv_id=view.arxiv_id,
                    title=view.title,
                    abstract=view.abstract,
                    authors=view.authors,
                    url=view.url,
                    published_at=view.published_at,
                    score=score,
                )
            )
        return chunks
