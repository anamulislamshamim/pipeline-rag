from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.deps import get_rag_pipeline
from app.api.schemas import AskRequest, AskResponse, SourceOut
from app.services.rag.pipeline import RAGPipeline

router = APIRouter()


@router.post("/ask", response_model=AskResponse)
def ask(payload: AskRequest, pipeline: RAGPipeline = Depends(get_rag_pipeline)) -> AskResponse:
    answer = pipeline.ask(payload.question)
    return AskResponse(
        answer=answer.answer,
        grounded=answer.grounded,
        sources=[
            SourceOut(
                arxiv_id=source.arxiv_id,
                title=source.title,
                authors=source.authors,
                url=source.url,
                published=source.published_at.date().isoformat(),
                score=round(source.score, 4),
            )
            for source in answer.sources
        ],
    )
