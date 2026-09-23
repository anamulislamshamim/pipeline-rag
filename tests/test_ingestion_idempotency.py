import unittest
from datetime import datetime, timedelta, timezone

from app.domain.models import PaperRecord
from app.ports import PaperView, UpsertResult
from app.services.ingestion import IngestionService


def make_record(i: int, abstract: str = "abstract", updated_at: datetime | None = None) -> PaperRecord:
    base_time = datetime(2024, 1, 1, tzinfo=timezone.utc)
    return PaperRecord(
        arxiv_id=f"2401.{i:05d}",
        version=1,
        title=f"Paper {i}",
        abstract=abstract,
        authors=["A. Author"],
        primary_category="cs.AI",
        categories=["cs.AI"],
        published_at=base_time,
        updated_at=updated_at or base_time,
        url=f"http://arxiv.org/abs/2401.{i:05d}v1",
        content_hash=f"hash-{abstract}",
    )


class FakePaperSource:
    """Implements the PaperSource port over an in-memory list."""

    def __init__(self, records: list[PaperRecord]):
        self._records = records

    def fetch(self, categories, max_results, page_size):
        yield from self._records[:max_results]


class InMemoryPaperRepository:
    """Implements the PaperRepository port with a plain dict.

    Mirrors the real upsert semantics of SqlPaperRepository closely enough
    to exercise IngestionService's page-by-page logic and early-stop rule.
    """

    def __init__(self):
        self._by_arxiv_id: dict[str, PaperRecord] = {}

    def upsert_papers(self, records) -> UpsertResult:
        records = list(records)
        inserted = updated = unchanged = 0
        for record in records:
            existing = self._by_arxiv_id.get(record.arxiv_id)
            if existing is None:
                self._by_arxiv_id[record.arxiv_id] = record
                inserted += 1
            elif existing.updated_at < record.updated_at or existing.content_hash != record.content_hash:
                self._by_arxiv_id[record.arxiv_id] = record
                updated += 1
            else:
                unchanged += 1
        return UpsertResult(inserted=inserted, updated=updated, unchanged=unchanged, fetched=len(records))

    def get_papers_needing_embedding(self, model):
        return []

    def mark_embedded(self, paper_id, content_hash, model):
        pass

    def get_papers_by_ids(self, paper_ids):
        return {}

    def wipe(self):
        self._by_arxiv_id.clear()


class TestIngestionIdempotency(unittest.TestCase):
    def test_first_run_inserts_everything(self):
        records = [make_record(i) for i in range(5)]
        repo = InMemoryPaperRepository()
        service = IngestionService(FakePaperSource(records), repo, page_size=10)

        summary = service.run(categories=["cs.AI"], max_papers=5)

        self.assertEqual(summary.inserted, 5)
        self.assertEqual(summary.updated, 0)
        self.assertEqual(summary.unchanged, 0)
        self.assertEqual(len(repo._by_arxiv_id), 5)

    def test_rerunning_with_same_data_is_a_noop(self):
        records = [make_record(i) for i in range(5)]
        repo = InMemoryPaperRepository()
        IngestionService(FakePaperSource(records), repo, page_size=10).run(["cs.AI"], 5)

        # Re-run with the exact same records: nothing should change.
        summary2 = IngestionService(FakePaperSource(records), repo, page_size=10).run(["cs.AI"], 5)

        self.assertEqual(summary2.inserted, 0)
        self.assertEqual(summary2.updated, 0)
        self.assertEqual(summary2.unchanged, 5)
        self.assertEqual(len(repo._by_arxiv_id), 5)  # still no duplicates

    def test_changed_abstract_triggers_update_not_duplicate(self):
        records = [make_record(i) for i in range(5)]
        repo = InMemoryPaperRepository()
        IngestionService(FakePaperSource(records), repo, page_size=10).run(["cs.AI"], 5)

        later = datetime(2024, 1, 2, tzinfo=timezone.utc)
        changed = [make_record(0, abstract="a totally new abstract", updated_at=later)] + records[1:]
        summary2 = IngestionService(FakePaperSource(changed), repo, page_size=10).run(["cs.AI"], 5)

        self.assertEqual(summary2.inserted, 0)
        self.assertEqual(summary2.updated, 1)
        self.assertEqual(summary2.unchanged, 4)
        self.assertEqual(len(repo._by_arxiv_id), 5)  # still no duplicates
        self.assertEqual(repo._by_arxiv_id["2401.00000"].abstract, "a totally new abstract")

    def test_stops_early_when_a_full_page_is_unchanged(self):
        # Simulates incremental re-ingestion: newest-updated-first, and once
        # we hit a whole page of already-seen papers we should stop fetching.
        records = [make_record(i) for i in range(20)]
        repo = InMemoryPaperRepository()
        IngestionService(FakePaperSource(records), repo, page_size=10).run(["cs.AI"], 20)

        class CountingSource(FakePaperSource):
            def __init__(self, records):
                super().__init__(records)
                self.pages_served = 0

            def fetch(self, categories, max_results, page_size):
                for i in range(0, min(len(self._records), max_results), page_size):
                    self.pages_served += 1
                    yield from self._records[i : i + page_size]

        source = CountingSource(records)  # identical data -> first page is fully unchanged
        summary = IngestionService(source, repo, page_size=10).run(["cs.AI"], max_papers=20)

        self.assertTrue(summary.stopped_early)
        self.assertEqual(summary.unchanged, 10)
        # Only the first page should have been consumed before stopping.
        self.assertEqual(source.pages_served, 1)


if __name__ == "__main__":
    unittest.main()
