"""Cross-encoder reranking adapter, implements the Reranker port.

A cross-encoder scores (query, document) pairs jointly through one model
pass, which is more accurate than comparing two independently-computed
embedding vectors - but far too slow to run over an entire corpus. So it's
applied only to the small candidate set retrieval already narrowed down to
(RETRIEVAL_TOP_K, default 20), to pick the best RETRIEVAL_TOP_N for context.
"""
from __future__ import annotations

from fastembed.rerank.cross_encoder import TextCrossEncoder

from app.domain.models import RetrievedChunk


class FastEmbedReranker:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = TextCrossEncoder(model_name=model_name)

    def rerank(self, query: str, candidates: list[RetrievedChunk]) -> list[tuple[RetrievedChunk, float]]:
        if not candidates:
            return []
        documents = [f"{c.title}. {c.abstract}" for c in candidates]
        scores = list(self._model.rerank(query, documents))
        paired = list(zip(candidates, scores))
        paired.sort(key=lambda pair: pair[1], reverse=True)
        return paired
