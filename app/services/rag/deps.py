from __future__ import annotations

from dataclasses import dataclass

from app.services.rag.context_builder import ContextBuilder
from app.services.rag.generator import Generator
from app.services.rag.retriever import Retriever
from app.services.rag.validator import AnswerValidator


@dataclass(frozen=True)
class RagDeps:
    retriever: Retriever
    context_builder: ContextBuilder
    generator: Generator
    validator: AnswerValidator

    top_k: int
    top_n: int
    min_score: float
    max_context_tokens: int
    max_attempts: int