"""Turns retrieved candidates into the numbered <papers> block sent to the LLM.

Single responsibility: given candidates (already score-filtered by
RetrievingState), decide which ones actually make it into the prompt and
render them. Reranking is optional (Reranker port, may be None) so this
class works identically with or without it - Liskov-safe substitution.
"""
from __future__ import annotations

from app.domain.models import RetrievedChunk
from app.ports import Reranker


class ContextBuilder:
    def __init__(self, reranker: Reranker | None = None, chars_per_token: float = 4.0):
        self._reranker = reranker
        self._chars_per_token = chars_per_token

    def build(
        self, question: str, candidates: list[RetrievedChunk], top_n: int, max_context_tokens: int
    ) -> tuple[str, list[RetrievedChunk]]:
        ranked = self._rerank(question, candidates) if self._reranker else candidates
        top = ranked[:top_n]

        budget_chars = max_context_tokens * self._chars_per_token
        used_chars = 0
        blocks: list[str] = []
        used: list[RetrievedChunk] = []

        for chunk in top:
            block = self._render_block(len(used) + 1, chunk)
            # Always keep at least one block, even if it alone exceeds the
            # budget - an over-budget answer beats an empty one.
            if used and used_chars + len(block) > budget_chars:
                break
            blocks.append(block)
            used.append(chunk)
            used_chars += len(block)

        return "\n\n".join(blocks), used

    def _rerank(self, question: str, candidates: list[RetrievedChunk]) -> list[RetrievedChunk]:
        pairs = self._reranker.rerank(question, candidates)  # [(chunk, score), ...] best-first
        return [chunk for chunk, _score in pairs]

    def _render_block(self, index: int, chunk: RetrievedChunk) -> str:
        authors = ", ".join(chunk.authors[:3])
        if len(chunk.authors) > 3:
            authors += " et al."
        return f"[{index}] {chunk.title} ({authors}, {chunk.published_at.date()})\n{chunk.abstract}"
