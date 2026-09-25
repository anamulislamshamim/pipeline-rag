from __future__ import annotations

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000)


class SourceOut(BaseModel):
    arxiv_id: str
    title: str
    authors: list[str]
    url: str
    published: str
    score: float


class AskResponse(BaseModel):
    answer: str
    grounded: bool
    sources: list[SourceOut]
