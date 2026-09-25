"""Concrete aggregation strategies. Each does exactly one query - kept as
plain SQL (via SQLAlchemy's text()) rather than the ORM, since these are
read-only reporting queries with no domain object to hydrate, and raw SQL
is the most direct, reviewable way to express a GROUP BY/aggregate.

Aggregation is based on each paper's primary category (papers.primary_category_id)
rather than the full paper_categories many-to-many, so a paper is counted
exactly once per chart instead of once per category it's tagged with -
simpler to reason about and matches how the assignment's examples read
("paper count by category").
"""
from __future__ import annotations

from sqlalchemy import text
from sqlalchemy.orm import Session


class PapersByCategoryOverTime:
    name = "papers-by-category-over-time"

    def run(self, session: Session, granularity: str = "month", top: int = 5, **_: object) -> list[dict]:
        if granularity not in ("month", "year"):
            raise ValueError("granularity must be 'month' or 'year'")
        top = max(1, min(int(top), 20))

        rows = session.execute(
            text(
                """
                WITH top_categories AS (
                    SELECT c.code
                    FROM papers p
                    JOIN categories c ON c.id = p.primary_category_id
                    GROUP BY c.code
                    ORDER BY count(*) DESC
                    LIMIT :top
                )
                SELECT
                    c.code AS category,
                    date_trunc(:granularity, p.published_at) AS bucket,
                    count(*) AS paper_count
                FROM papers p
                JOIN categories c ON c.id = p.primary_category_id
                WHERE c.code IN (SELECT code FROM top_categories)
                GROUP BY c.code, bucket
                ORDER BY bucket, c.code
                """
            ),
            {"granularity": granularity, "top": top},
        ).all()

        return [
            {"category": r.category, "bucket": r.bucket.date().isoformat(), "paper_count": r.paper_count}
            for r in rows
        ]


class TopCategories:
    name = "top-categories"

    def run(self, session: Session, limit: int = 10, **_: object) -> list[dict]:
        limit = max(1, min(int(limit), 50))

        rows = session.execute(
            text(
                """
                SELECT c.code AS category, count(*) AS paper_count
                FROM papers p
                JOIN categories c ON c.id = p.primary_category_id
                GROUP BY c.code
                ORDER BY paper_count DESC
                LIMIT :limit
                """
            ),
            {"limit": limit},
        ).all()

        return [{"category": r.category, "paper_count": r.paper_count} for r in rows]


class TopAuthors:
    name = "top-authors"

    def run(self, session: Session, limit: int = 10, **_: object) -> list[dict]:
        limit = max(1, min(int(limit), 50))

        rows = session.execute(
            text(
                """
                SELECT a.name AS author, count(*) AS paper_count
                FROM paper_authors pa
                JOIN authors a ON a.id = pa.author_id
                GROUP BY a.id, a.name
                ORDER BY paper_count DESC
                LIMIT :limit
                """
            ),
            {"limit": limit},
        ).all()

        return [{"author": r.author, "paper_count": r.paper_count} for r in rows]


class AvgAuthorsPerPaper:
    name = "avg-authors-per-paper"

    def run(self, session: Session, **_: object) -> list[dict]:
        row = session.execute(
            text(
                """
                SELECT
                    count(DISTINCT p.id) AS paper_count,
                    count(pa.author_id) AS author_link_count,
                    CASE WHEN count(DISTINCT p.id) = 0 THEN 0
                         ELSE round(count(pa.author_id)::numeric / count(DISTINCT p.id), 2)
                    END AS avg_authors_per_paper
                FROM papers p
                LEFT JOIN paper_authors pa ON pa.paper_id = p.id
                """
            )
        ).one()

        # Returned as a single-item list, not a bare dict, so every
        # aggregation's `data` field has the same list shape for callers.
        return [
            {
                "paper_count": row.paper_count,
                "avg_authors_per_paper": float(row.avg_authors_per_paper),
            }
        ]
