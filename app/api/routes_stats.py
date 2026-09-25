from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.api.deps import get_db_session
from app.services.stats.registry import get_aggregation, list_aggregations

router = APIRouter()


@router.get("/stats")
def list_stats() -> dict:
    """Lists the available aggregation names for GET /stats/{aggregation}."""
    return {"available": list_aggregations()}


@router.get("/stats/{aggregation}")
def run_stats(
    aggregation: str,
    granularity: str = Query("month", pattern="^(month|year)$", description="Used by papers-by-category-over-time"),
    top: int = Query(5, ge=1, le=20, description="Used by papers-by-category-over-time"),
    limit: int = Query(10, ge=1, le=50, description="Used by top-categories / top-authors"),
    session: Session = Depends(get_db_session),
) -> dict:
    try:
        strategy = get_aggregation(aggregation)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    params = {"granularity": granularity, "top": top, "limit": limit}
    try:
        data = strategy.run(session, **params)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return {"aggregation": aggregation, "params": params, "data": data}
