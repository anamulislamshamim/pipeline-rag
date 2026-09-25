"""Composition root for the API process.

This is the only module allowed to import concrete adapters directly -
everywhere else (services, states, the pipeline) depends on app.ports.
FastAPI's Depends() plus lru_cache gives us cheap singletons without a
separate DI framework, which would be over-engineering for this scope.
"""
from __future__ import annotations

import os
from functools import lru_cache

from app.adapters.faiss_store import FaissVectorStore
from app.adapters.fastembed_embedder import FastEmbedEmbedder
from app.adapters.llm_factory import build_llm_client
from app.adapters.reranker_fastembed import FastEmbedReranker
from app.adapters.sql_repository import SqlPaperRepository
from app.config import Settings, get_settings
from app.db.session import make_session_factory
from app.ports import Embedder, LLMClient, PaperRepository, Reranker, VectorStore
from app.services.rag.context_builder import ContextBuilder
from app.services.rag.deps import RagDeps
from app.services.rag.generator import Generator
from app.services.rag.pipeline import RAGPipeline
from app.services.rag.retriever import Retriever
from app.services.rag.validator import AnswerValidator


@lru_cache
def get_repository() -> PaperRepository:
    settings = get_settings()
    return SqlPaperRepository(make_session_factory(settings))


@lru_cache
def get_embedder() -> Embedder:
    settings = get_settings()
    return FastEmbedEmbedder(model_name=settings.embed_model, dim=settings.embed_dim)


@lru_cache
def get_vector_store() -> VectorStore:
    settings = get_settings()
    path = os.path.join(settings.faiss_dir, settings.faiss_index_name)
    return FaissVectorStore(dim=settings.embed_dim, index_path=path)


@lru_cache
def get_reranker() -> Reranker | None:
    settings = get_settings()
    if not settings.rerank_enabled:
        return None
    return FastEmbedReranker(model_name=settings.rerank_model)


@lru_cache
def get_llm_client() -> LLMClient:
    return build_llm_client(get_settings())


@lru_cache
def get_rag_pipeline() -> RAGPipeline:
    settings: Settings = get_settings()

    deps = RagDeps(
        retriever=Retriever(get_embedder(), get_vector_store(), get_repository()),
        context_builder=ContextBuilder(reranker=get_reranker()),
        generator=Generator(get_llm_client()),
        validator=AnswerValidator(),
        top_k=settings.retrieval_top_k,
        top_n=settings.retrieval_top_n,
        min_score=settings.retrieval_min_score,
        max_context_tokens=settings.rag_max_context_tokens,
        max_attempts=settings.rag_max_attempts,
    )
    return RAGPipeline(deps)
