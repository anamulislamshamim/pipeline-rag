"""Domain models: plain dataclasses with zero framework or DB dependencies.

There are what flow between adapters and services. Keeping them free of 
SQLAlchemy/Pydantic ties means the parsing and business logic can be
unit tested without a database or an HTTP client.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass(frozen=True)
class PaperRecord:
    """A single paper as parsed from the arXiv feed, before it touches
    the DB."""

    arxiv_id: str # normalized, nor version suffix, e.g. "2401.01234"
    version: int
    title: str
    abstract: str
    authors: list[str] # Ordered, as they appear on the paper
    primary_category: str
    categories: list[str] # Include primary category
    published_at: datetime
    updated_at: datetime
    url: str
    doi: str | None = None
    journal_ref: str | None = None
    content_hash: str = field(default="") # sha256(title + abstract), set by the cleaner/parser


@dataclass(frozen=True)
class RetrievedChunk:
    """A paper returned by the vector store for a query, with its 
    similarity score."""

    paper_id: int
    arxiv_id: str
    title: str
    abstract: str
    authors: list[str]
    url: str
    published_at: datetime
    score: float


@dataclass(frozen=True)
class Answer:
    """The final, validated response of the RAG pipeline."""

    answer: str
    grounded: bool
    source: list[RetrievedChunk]

