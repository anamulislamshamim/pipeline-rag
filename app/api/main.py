from __future__ import annotations

from fastapi import FastAPI

from app.api.routes_ask import router as ask_router
from app.config import get_settings
from app.logging_config import configure_logging

settings = get_settings()
configure_logging(settings.log_level)

app = FastAPI(title="Pipeline Inc RAG assignment for Shamim", version="1.0.0")
app.include_router(ask_router)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
