"""CLI entrypoints for ingestion and dataset maintenance.

Usage (inside the container, see README for docker compose form):
    python -m app.cli ingest --categories cs.AI,cs.CL,cs.LG --max-papers 3000
    python -m app.cli reindex
    python -m app.cli wipe --yes

Note: deliberately NOT using `from __future__ import annotations` here.
Typer needs real type objects (not the stringified annotations that PEP 563
produces) to build its CLI parser - with postponed evaluation active it
silently mis-detects option types and everything after the first option
gets treated as an unexpected positional argument.
"""
import logging
import os
from datetime import datetime, timezone

import typer

from app.adapters.arxiv_client import ArxivClient
from app.adapters.faiss_store import FaissVectorStore
from app.adapters.fastembed_embedder import FastEmbedEmbedder
from app.adapters.sql_repository import SqlPaperRepository
from app.config import Settings, get_settings
from app.db.models import IngestionRun
from app.db.session import make_session_factory
from app.logging_config import configure_logging
from app.services.indexing import IndexingService
from app.services.ingestion import IngestionService

app = typer.Typer(help="Pipeline Inc RAG assessment - data CLI")
logger = logging.getLogger(__name__)


@app.command()
def ingest(
    categories: str = typer.Option(
        None,
        "--categories",
        help="Comma-separated arXiv categories, e.g. --categories cs.AI,cs.CL,cs.LG "
        "(defaults to ARXIV_CATEGORIES from .env)",
    ),
    max_papers: int = typer.Option(None, "--max-papers", help="Maximum papers to fetch this run"),
    skip_index: bool = typer.Option(
        False,
        "--skip-index",
        help="Skip re-embedding into FAISS after ingesting (faster; run `reindex` later)",
    ),
):
    """Fetch papers from arXiv, upsert them into Postgres, then (re-)embed
    any new or changed papers into the FAISS index so /ask can see them.

    Safe to re-run: existing unchanged papers are skipped, changed papers are
    updated in place, and re-runs will stop early once they catch up with
    already-ingested data (see IngestionService). Only actually-changed
    papers get re-embedded (see IndexingService).
    """
    settings = get_settings()
    configure_logging(settings.log_level)

    cats = [c.strip() for c in categories.split(",") if c.strip()] if categories else settings.arxiv_category_list
    limit = max_papers or settings.arxiv_max_papers

    session_factory = make_session_factory(settings)
    repository = SqlPaperRepository(session_factory)
    source = ArxivClient(
        base_url=settings.arxiv_base_url,
        min_request_interval_seconds=settings.arxiv_min_request_interval_seconds,
    )
    service = IngestionService(source, repository, page_size=settings.arxiv_page_size)

    run_row = _start_run(session_factory)
    try:
        summary = service.run(categories=cats, max_papers=limit)
        _finish_run(session_factory, run_row.id, "success", summary.fetched, summary.inserted, summary.updated)
        typer.echo(f"Done. {summary}")
    except Exception as exc:  # noqa: BLE001 - top-level CLI boundary
        logger.exception("Ingestion failed")
        _finish_run(session_factory, run_row.id, "failed", 0, 0, 0, error=str(exc))
        raise typer.Exit(code=1) from exc

    if skip_index:
        typer.echo("Skipped indexing (--skip-index). Run `reindex` before querying /ask.")
        return

    indexing = _build_indexing_service(settings, repository)
    embedded = indexing.sync()
    typer.echo(f"Indexed {embedded} paper(s) into FAISS.")


@app.command()
def reindex():
    """Re-embed any papers whose content or embedding model changed, without
    re-fetching from arXiv. Useful after --skip-index, after changing
    EMBED_MODEL, or to catch up the FAISS index for any other reason."""
    settings = get_settings()
    configure_logging(settings.log_level)
    session_factory = make_session_factory(settings)
    repository = SqlPaperRepository(session_factory)

    indexing = _build_indexing_service(settings, repository)
    embedded = indexing.sync()
    typer.echo(f"Indexed {embedded} paper(s).")


@app.command()
def wipe(
    yes: bool = typer.Option(False, "--yes", help="Confirm you want to delete all data"),
):
    """Delete all papers, authors, categories, and embedding records, and
    remove the on-disk FAISS index file, so a fresh import can be run from
    scratch."""
    if not yes:
        typer.echo("Refusing to wipe without --yes. This deletes ALL ingested data.")
        raise typer.Exit(code=1)

    settings = get_settings()
    configure_logging(settings.log_level)
    session_factory = make_session_factory(settings)
    SqlPaperRepository(session_factory).wipe()

    index_path = os.path.join(settings.faiss_dir, settings.faiss_index_name)
    if os.path.exists(index_path):
        os.remove(index_path)
        typer.echo(f"Removed FAISS index file: {index_path}")

    typer.echo("Dataset wiped.")


def _build_indexing_service(settings: Settings, repository: SqlPaperRepository) -> IndexingService:
    embedder = FastEmbedEmbedder(model_name=settings.embed_model, dim=settings.embed_dim)
    index_path = os.path.join(settings.faiss_dir, settings.faiss_index_name)
    vector_store = FaissVectorStore(dim=settings.embed_dim, index_path=index_path)
    return IndexingService(repository, embedder, vector_store)


def _start_run(session_factory) -> IngestionRun:
    with session_factory() as session:
        run = IngestionRun(status="running", started_at=datetime.now(timezone.utc))
        session.add(run)
        session.commit()
        session.refresh(run)
        return run


def _finish_run(session_factory, run_id: int, status: str, fetched: int, inserted: int, updated: int, error: str | None = None):
    with session_factory() as session:
        run = session.get(IngestionRun, run_id)
        run.status = status
        run.finished_at = datetime.now(timezone.utc)
        run.fetched = fetched
        run.inserted = inserted
        run.updated = updated
        run.error = error
        session.commit()


if __name__ == "__main__":
    app()
