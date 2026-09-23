"""Orchestrates a full ingestion run.

Design notes:
- Depends only on the PaperSource and PaperRepository *ports*, not on the
  concrete arXiv/SQL adapters (Dependency Inversion) -> trivially unit
  tested with in-memory fakes (see tests/test_ingestion_idempotency.py).
- Commits per page (via repository.upsert_papers per batch) so a crash
  partway through loses at most one page of work, not the whole run.
- Because ArxivClient.fetch() yields newest-updated-first, once we hit a
  full page where every paper was already up to date, everything after it
  is guaranteed unchanged too -> we can stop early on incremental re-runs.
"""
from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass

from app.ports import PaperRepository, PaperSource, UpsertResult

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IngestionSummary:
    fetched: int
    inserted: int
    updated: int
    unchanged: int
    stopped_early: bool


class IngestionService:
    def __init__(self, source: PaperSource, repository: PaperRepository, page_size: int = 100):
        self._source = source
        self._repository = repository
        self._page_size = page_size

    def run(self, categories: list[str], max_papers: int) -> IngestionSummary:
        logger.info("Starting ingestion: categories=%s max_papers=%d", categories, max_papers)

        totals = {"fetched": 0, "inserted": 0, "updated": 0, "unchanged": 0}
        stopped_early = False

        record_stream = self._source.fetch(
            categories=categories, max_results=max_papers, page_size=self._page_size
        )

        while True:
            page = list(itertools.islice(record_stream, self._page_size))
            if not page:
                break

            result: UpsertResult = self._repository.upsert_papers(page)
            totals["fetched"] += result.fetched
            totals["inserted"] += result.inserted
            totals["updated"] += result.updated
            totals["unchanged"] += result.unchanged

            logger.info(
                "Page done: +%d inserted, +%d updated, +%d unchanged (running totals: %s)",
                result.inserted, result.updated, result.unchanged, totals,
            )

            # Results are sorted newest-updated-first. If an entire page was
            # already fully up to date, every subsequent page will be too.
            if result.inserted == 0 and result.updated == 0 and result.fetched == self._page_size:
                logger.info("Full page of unchanged papers reached; stopping early (incremental sync).")
                stopped_early = True
                break

        summary = IngestionSummary(stopped_early=stopped_early, **totals)
        logger.info("Ingestion finished: %s", summary)
        return summary
