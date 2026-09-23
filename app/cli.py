"""CLI entrypoints for ingestion and dataset maintenance.

Usage (inside the container, see README for docker compose form):
    python -m app.cli ingest --categories cs.AI,cs.CL,cs.LG --max-papers 3000
    python -m app.cli wipe --yes

Note: deliberately NOT using `from __future__ import annotations` here.
Typer needs real type objects (not the stringified annotations that PEP 563
produces) to build its CLI parser - with postponed evaluation active it
silently mis-detects option types and everything after the first option
gets treated as an unexpected positional argument.
"""
import logging
from datetime import datetime, timezone

import typer

from app.adapters.arxiv_client import ArxivClient
from app.adapters.sql_repository import SqlPaperRepository
from app.config import get_settings
from app.db.models import IngestionRun
from app.db.session import make_session_factory
from app.logging_config import configure_logging
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
    max_papers: int = typer.Option(
        None,
        "--max-papers",
        help="Maximum papers to fetch this run"
    ),
):
    """Fetch papers from arXiv and upsert them into Postgres.

    Safe to re-run: existing unchanged papers are skipped, changed papers are
    updated in place, and re-runs will stop early once they catch up with
    already-ingested data (see IngestionService).
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

    run_now = _start_run(session_factory)
    try:
        summary = service.run(categories=cats, max_papers=limit)
        _finish_run(session_factory, run_now.id, "success", summary.fetched, summary.inserted, summary.updated)
        typer.echo(f"Done, {summary}")
    except Exception as exc:
        logger.exception("Ingestion failed")
        _finish_run(session_factory, run_now.id, "failed", 0, 0, 0, error=str(exc))
        raise typer.Exit(code=1) from exc


@app.command()
def wipe(
    yes: bool = typer.Option(
        False,
        "--yes",
        help="Confirm you want to delete all data",
    ),

):
    """Delete all papers, authors, categories, and embedding records so a
    fresh import can be run from scratch. Does NOT delete the FAISS index
    file on disk - run `reindex --rebuild` afterwards (Day 2) if needed."""
    if not yes:
        typer.echo("Refusing to wipe without --yes. This delete ALL ingested data.")
        raise typer.Exit(code=1)

    settings = get_settings()
    configure_logging(settings.log_level)
    session_factory = make_session_factory(settings)
    SqlPaperRepository(session_factory).wipe()
    typer.echo("Dataset wiped.")


def _start_run(session_factory) -> IngestionRun:
    with session_factory() as session:
        run = IngestionRun(status="running", started_at=datetime.now(timezone.utc))
        session.add(run)
        session.commit()
        session.refresh(run)
        return run


def _finish_run(session_factory, run_id: int, status: str, fetched: int, inserted: int, updated: int, error: str | None = None) -> None:
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