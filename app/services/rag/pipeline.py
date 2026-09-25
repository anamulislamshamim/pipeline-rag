from __future__ import annotations

from app.domain.models import Answer
from app.services.rag.context import PipelineContext
from app.services.rag.deps import RagDeps
from app.services.rag.states import RETRIEVING, QueryState


class RAGPipeline:
    """Drives the query lifecycle state machine (see states.py) to an Answer.

    Deliberately has almost no logic of its own - it exists so the states
    don't need to know about each other's plumbing (starting state,
    loop termination), keeping each state's handle() focused purely on its
    own transition rule.
    """

    def __init__(self, deps: RagDeps):
        self._deps = deps

    def ask(self, question: str) -> Answer:
        ctx = PipelineContext(question=question)
        state: QueryState | None = RETRIEVING
        while state is not None:
            state = state.handle(ctx, self._deps)

        assert ctx.answer is not None, "pipeline terminated without setting ctx.answer - bug in a state"
        return ctx.answer
