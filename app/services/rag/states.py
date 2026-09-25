"""RAG query lifecycle as an explicit State machine.

    Retrieving ──(no hits >= min_score)──► NoMatch [terminal]
        │
        └─(has hits)─► BuildingContext ─► Generating ─► Validating ─┬─► done (ctx.answer set)
                            ▲                                       │
                            └──────────(retry, attempts left)───────┤
                                                                     └─► NoMatch [terminal]
                                                                         (attempts exhausted, or
                                                                          model itself says
                                                                          "sufficient": false)

Two structural guarantees this buys, both required by the assignment:
1. The LLM is architecturally unreachable from NoMatch - there is no code
   path from "nothing relevant found" to a generated answer. This is a
   stronger guarantee than "the prompt tells the model not to guess".
2. A malformed or out-of-range-citation response gets exactly one retry
   (see AnswerValidator) before the pipeline gives up honestly, instead of
   ever returning fabricated content.

Each state is a stateless singleton - it takes no constructor arguments and
carries no instance data, so one instance is created below and reused
across every request. `handle()` reads/writes the *context* argument only.
This is what a classic State pattern actor looks like when the "state's own
data" isn't needed - stateless flyweights are perfectly idiomatic here and
keep the pipeline reusable across concurrent requests.
"""
from __future__ import annotations

from typing import Protocol

from app.services.rag.context import PipelineContext
from app.services.rag.deps import RagDeps
from app.services.rag.validator import ValidationError


class QueryState(Protocol):
    def handle(self, ctx: PipelineContext, deps: RagDeps) -> "QueryState | None":
        """Mutate ctx as needed and return the next state, or None if done
        (ctx.answer must be set by the time None is returned)."""
        ...


class _RetrievingState:
    def handle(self, ctx: PipelineContext, deps: RagDeps) -> QueryState | None:
        results = deps.retriever.retrieve(ctx.question, deps.top_k)
        candidates = [c for c in results if c.score >= deps.min_score]

        if not candidates:
            return NO_MATCH

        ctx.candidates = candidates
        return BUILDING_CONTEXT


class _BuildingContextState:
    def handle(self, ctx: PipelineContext, deps: RagDeps) -> QueryState | None:
        context_text, used = deps.context_builder.build(
            ctx.question, ctx.candidates, deps.top_n, deps.max_context_tokens
        )
        ctx.context_text = context_text
        ctx.used_chunks = used
        return GENERATING


class _GeneratingState:
    def handle(self, ctx: PipelineContext, deps: RagDeps) -> QueryState | None:
        ctx.raw_output = deps.generator.generate(ctx.question, ctx.context_text, correction=ctx.last_error)
        ctx.last_error = None
        return VALIDATING


class _ValidatingState:
    def handle(self, ctx: PipelineContext, deps: RagDeps) -> QueryState | None:
        try:
            ctx.answer = deps.validator.validate(ctx.raw_output, ctx.used_chunks)
            return None  # done
        except ValidationError as exc:
            ctx.attempt += 1
            ctx.last_error = str(exc)
            if ctx.attempt < deps.max_attempts:
                return GENERATING
            # Retries exhausted: give up honestly rather than ever
            # returning an unvalidated / possibly-hallucinated answer.
            return NO_MATCH


class _NoMatchState:
    def handle(self, ctx: PipelineContext, deps: RagDeps) -> QueryState | None:
        if ctx.answer is None:  # not already set by a "sufficient: false" path upstream
            ctx.answer = _no_match_answer()
        return None


def _no_match_answer():
    from app.domain.models import Answer

    return Answer(
        answer="I couldn't find papers in this dataset relevant to that question.",
        grounded=False,
        sources=[],
    )


# Stateless singletons, reused across every request.
RETRIEVING = _RetrievingState()
BUILDING_CONTEXT = _BuildingContextState()
GENERATING = _GeneratingState()
VALIDATING = _ValidatingState()
NO_MATCH = _NoMatchState()
