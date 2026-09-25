"""Keeps the FAISS index in sync with Postgres.

Diffs papers against `paper_embeddings` (by content_hash + embedding model)
to find exactly which papers need (re-)embedding - new papers, changed
abstracts, or a switched embedding model all show up here. Only records a
paper as embedded *after* it's actually in the vector store and saved to
disk, so a crash mid-run just means the next sync retries those papers
rather than silently believing stale work is done.
"""
from __future__ import annotations

import logging

from app.ports import Embedder, PaperRepository, VectorStore

logger = logging.getLogger(__name__)


class IndexingService:
    def __init__(
        self,
        repository: PaperRepository,
        embedder: Embedder,
        vector_store: VectorStore,
        batch_size: int = 64,
    ):
        self._repository = repository
        self._embedder = embedder
        self._vector_store = vector_store
        self._batch_size = batch_size

    def sync(self) -> int:
        stale = self._repository.get_papers_needing_embedding(self._embedder.model_name)
        if not stale:
            logger.info("Index already up to date, nothing to embed.")
            return 0

        logger.info("Embedding %d paper(s) with model=%s", len(stale), self._embedder.model_name)
        total = 0
        for i in range(0, len(stale), self._batch_size):
            batch = stale[i : i + self._batch_size]
            ids = [paper_id for paper_id, _text, _hash in batch]
            texts = [text for _id, text, _hash in batch]

            vectors = self._embedder.embed_passages(texts)

            # Remove first so a re-embed of an existing paper does not
            # create a duplicate id in the index
            self._vector_store.remove(ids)
            self._vector_store.add(ids, vectors)
            self._vector_store.save()

            for paper_id, _text, content_hash in batch:
                self._repository.mark_embedded(paper_id, content_hash, self._embedder.model_name)

            total += len(batch)
            logger.info("Indexed %d/%d", total, len(stale))

        return total
