"""Embedder adapter backed by fastembed (ONNX, local, no API key/torch).

Implements the Embedder port. BGE models are asymmetric: queries and
passages should be embedded slightly differently (fastembed's query_embed/
passage_embed apply the right instruction prefix internally), which matters
for retrieval quality - using plain .embed() for both is a common mistake
that silently hurts recall.
"""
from __future__ import annotations

from fastembed import TextEmbedding


class FastEmbedEmbedder:
    def __init__(self, model_name: str, dim: int):
        self.model_name = model_name
        self.dim = dim
        self._model = TextEmbedding(model_name=model_name)

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return [vec.tolist() for vec in self._model.passage_embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        return next(iter(self._model.query_embed([text]))).tolist()
