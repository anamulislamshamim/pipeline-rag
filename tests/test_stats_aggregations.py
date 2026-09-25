"""Integration tests for the stats aggregations, against a real Postgres.

Requires a running, migrated DB - run with:
    docker compose run --rm api pytest tests/test_stats_aggregations.py -v
Not runnable in the dependency-light sandbox used for the rest of the suite
(no DB there).

Safety: each test runs inside a single DB transaction that is rolled back
in tearDown. The test deletes existing rows and inserts its own fixture
data, but since nothing is ever committed, your real ingested dataset is
completely unaffected by running this file - even though the test acts as
if the tables were empty.
"""
import unittest
from datetime import datetime, timezone

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from app.config import Settings
from app.services.stats.registry import get_aggregation


class TestStatsAggregations(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.settings = Settings()
        cls.engine = create_engine(cls.settings.database_url, future=True)

    @classmethod
    def tearDownClass(cls):
        cls.engine.dispose()

    def setUp(self):
        self.connection = self.engine.connect()
        self.trans = self.connection.begin()
        self.session = sessionmaker(bind=self.connection, future=True)()
        self._seed()

    def tearDown(self):
        self.session.close()
        self.trans.rollback()  # undoes every change setUp/the test made
        self.connection.close()

    def _seed(self):
        s = self.session
        # Clear within the (never-committed) transaction so results are
        # deterministic regardless of what's really in the DB.
        for table in ("paper_authors", "paper_categories", "paper_embeddings", "papers", "authors", "categories"):
            s.execute(text(f"DELETE FROM {table}"))

        cat_ids = {}
        for code in ("cs.AI", "cs.CL", "hep-th"):
            cat_ids[code] = s.execute(
                text("INSERT INTO categories (code) VALUES (:code) RETURNING id"), {"code": code}
            ).scalar_one()

        author_ids = {}
        for name in ("Alice", "Bob", "Carol"):
            author_ids[name] = s.execute(
                text("INSERT INTO authors (name, name_normalized) VALUES (:n, :nn) RETURNING id"),
                {"n": name, "nn": name.lower()},
            ).scalar_one()

        # (primary_category, published_at, [authors]) - chosen so bucket/
        # category groupings are unambiguous to assert on.
        papers = [
            ("cs.AI", datetime(2024, 1, 15, tzinfo=timezone.utc), ["Alice", "Bob"]),
            ("cs.AI", datetime(2024, 1, 20, tzinfo=timezone.utc), ["Bob", "Carol"]),
            ("cs.AI", datetime(2024, 2, 10, tzinfo=timezone.utc), ["Alice"]),
            ("cs.CL", datetime(2024, 2, 5, tzinfo=timezone.utc), ["Alice", "Bob", "Carol"]),
            ("hep-th", datetime(2023, 12, 1, tzinfo=timezone.utc), ["Carol"]),
        ]

        for i, (cat_code, published_at, authors) in enumerate(papers):
            paper_id = s.execute(
                text(
                    """
                    INSERT INTO papers
                        (arxiv_id, version, title, abstract, primary_category_id,
                         published_at, updated_at, url, content_hash)
                    VALUES
                        (:arxiv_id, 1, :title, 'abstract text', :cat_id,
                         :published_at, :published_at, :url, :hash)
                    RETURNING id
                    """
                ),
                {
                    "arxiv_id": f"9999.{i:05d}",
                    "title": f"Test Paper {i}",
                    "cat_id": cat_ids[cat_code],
                    "published_at": published_at,
                    "url": f"http://arxiv.org/abs/9999.{i:05d}",
                    "hash": f"hash-{i}",
                },
            ).scalar_one()

            s.execute(
                text("INSERT INTO paper_categories (paper_id, category_id) VALUES (:p, :c)"),
                {"p": paper_id, "c": cat_ids[cat_code]},
            )
            for position, author_name in enumerate(authors):
                s.execute(
                    text(
                        "INSERT INTO paper_authors (paper_id, author_id, position) VALUES (:p, :a, :pos)"
                    ),
                    {"p": paper_id, "a": author_ids[author_name], "pos": position},
                )
        s.flush()

    def test_top_categories_counts(self):
        rows = get_aggregation("top-categories").run(self.session, limit=10)
        counts = {r["category"]: r["paper_count"] for r in rows}

        self.assertEqual(counts, {"cs.AI": 3, "cs.CL": 1, "hep-th": 1})

    def test_top_categories_respects_limit(self):
        rows = get_aggregation("top-categories").run(self.session, limit=1)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["category"], "cs.AI")  # the only unambiguous top-1

    def test_top_authors_counts(self):
        rows = get_aggregation("top-authors").run(self.session, limit=10)
        counts = {r["author"]: r["paper_count"] for r in rows}

        self.assertEqual(counts, {"Alice": 3, "Bob": 3, "Carol": 3})

    def test_avg_authors_per_paper(self):
        rows = get_aggregation("avg-authors-per-paper").run(self.session)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["paper_count"], 5)
        # author-links: 2 + 2 + 1 + 3 + 1 = 9 across 5 papers = 1.8
        self.assertAlmostEqual(rows[0]["avg_authors_per_paper"], 1.8)

    def test_papers_by_category_over_time_monthly_buckets(self):
        rows = get_aggregation("papers-by-category-over-time").run(self.session, granularity="month", top=3)

        expected = [
            {"category": "hep-th", "bucket": "2023-12-01", "paper_count": 1},
            {"category": "cs.AI", "bucket": "2024-01-01", "paper_count": 2},
            {"category": "cs.AI", "bucket": "2024-02-01", "paper_count": 1},
            {"category": "cs.CL", "bucket": "2024-02-01", "paper_count": 1},
        ]
        self.assertEqual(rows, expected)

    def test_unknown_aggregation_raises_keyerror(self):
        with self.assertRaises(KeyError):
            get_aggregation("not-a-real-aggregation")

    def test_invalid_granularity_raises_valueerror(self):
        with self.assertRaises(ValueError):
            get_aggregation("papers-by-category-over-time").run(self.session, granularity="fortnight")


if __name__ == "__main__":
    unittest.main()
