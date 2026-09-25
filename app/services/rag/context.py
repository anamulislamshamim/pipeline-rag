"""Mutable state carried through the RAG pipeline's state machine.

Deliberately a plain mutable dataclass, not frozen: each QueryState.handle()
call reads and updates it in place, then hands it to the next state. This
is the "context" in the classic State pattern - the states themselves stay
stateless and reusable (see states.py), all per-query data lives here.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from app.domain.models import Answer, RetrievedChunk


@dataclass
class PipelineContext:
    question: str

    candidates: list[RetrievedChunk] = field(default_factory=list)
    context_text: str = ""
    used_chunks: list[RetrievedChunk] = field(default_factory=list)
    raw_output: str = ""

    attempt: int = 0
    last_error: str | None = None

    answer: Answer | None = None
