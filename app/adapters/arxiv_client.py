"""HTTP client for the arXiv export API.

Implements the PaperSource port. Handles pagination via start/max_results,
self-throttles to respect arXiv's "no more than 1 request per 3 seconds"
policy, and retries transient failures with backoff.
"""
from __future__ import annotations

import logging
from collections.abc import Iterator

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.adapters.arxiv_parser import parse_feed, total_results
from app.adapters.rate_limiter import RateLimiter
from app.domain.models import PaperRecord

logger = logging.getLogger(__name__)


class ArxivTransientError(Exception):
    """Raised on a 429/5xx response, so tenacity knows to retry only these."""


class ArxivClient:
    def __init__(
        self,
        base_url: str,
        min_request_interval_seconds: float,
        http_client: httpx.Client | None = None,
    ):
        self._base_url = base_url
        self._rate_limiter = RateLimiter(min_request_interval_seconds)
        self._http = http_client or httpx.Client(timeout=30.0)

    def fetch(self, categories: list[str], max_results: int, page_size: int) -> Iterator[PaperRecord]:
        """Yield PaperRecords across as many pages as needed, sorted by most
        recently updated first (so incremental re-ingestion can stop early
        once it hits already-seen, unchanged papers)."""
        search_query = " OR ".join(f"cat:{c}" for c in categories)
        start = 0
        yielded = 0

        while yielded < max_results:
            batch_size = min(page_size, max_results - yielded)
            xml_bytes = self._get_page(search_query, start, batch_size)

            if start == 0:
                available = total_results(xml_bytes)
                logger.info("arXiv reports %d total results for query", available)

            records = parse_feed(xml_bytes)
            if not records:
                logger.info("No more results at start=%d; stopping", start)
                break

            for record in records:
                yield record
                yielded += 1
                if yielded >= max_results:
                    break

            start += len(records)

    @retry(
        retry=retry_if_exception_type(ArxivTransientError),
        wait=wait_exponential(multiplier=2, min=2, max=60),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    def _get_page(self, search_query: str, start: int, max_results: int) -> bytes:
        self._rate_limiter.wait()
        params = {
            "search_query": search_query,
            "start": start,
            "max_results": max_results,
            "sortBy": "lastUpdatedDate",
            "sortOrder": "descending",
        }
        logger.debug("GET %s params=%s", self._base_url, params)
        response = self._http.get(self._base_url, params=params)

        if response.status_code == 429 or response.status_code >= 500:
            raise ArxivTransientError(f"arXiv returned {response.status_code}")
        response.raise_for_status()
        return response.content
