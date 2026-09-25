"""SQLAlchemy implementation of the PaperRepository port.

Re-running ingestion must not duplicate rows, and must only touch a peper's
row (and mart it stale for re-embedding) when its content actually changed.
"""
from __future__ import annotations
import logging
import re
from collections.abc import Iterable
from typing import Sequence

from sqlalchemy import delete, select
from sqlalchemy.orm import Session, sessionmaker

from app.db.models import (
    Author,
    Category,
    IngestionRun,
    Paper,
    PaperAuthor,
    PaperCategory,
    PaperEmbedding,
)
from app.domain.models import PaperRecord
from app.ports import PaperView, UpsertResult

logger = logging.getLogger(__name__)

_WHITESPACE_RE = re.compile(r"\s+")


def normalize_author_name(name: str) -> str:
    return _WHITESPACE_RE.sub(" ", name.strip().lower())


class SqlPaperRepository:
    def __init__(self, session_factory: sessionmaker[Session]):
        self._session_factory = session_factory

    # Ingestion
    def upsert_papers(self, records: Iterable[PaperRecord]) -> UpsertResult:
        records = list(records)
        inserted = updated = unchanged = 0

        with self._session_factory() as session:
            for record in records:
                category_ids = self._get_or_create_categories(session, record.categories)
                primary_category_id = category_ids[record.primary_category]
                author_ids = self._get_or_create_authors(session, record.authors)

                existing = session.scalar(select(Paper).where(Paper.arxiv_id == record.arxiv_id))

                if existing is None:
                    paper = Paper(
                        arxiv_id=record.arxiv_id,
                        version=record.version,
                        title=record.title,
                        abstract=record.abstract,
                        primary_category_id=primary_category_id,
                        published_at=record.published_at,
                        updated_at=record.updated_at,
                        doi=record.doi,
                        journal_ref=record.journal_ref,
                        url=record.url,
                        content_hash=record.content_hash,
                    )
                    session.add(paper)
                    session.flush()
                    self._replace_links(session, paper.id, category_ids.values(), author_ids)
                    inserted += 1

                elif existing.updated_at < record.updated_at or existing.content_hash != record.content_hash:
                    existing.version = record.version
                    existing.title = record.title
                    existing.abstract = record.abstract
                    existing.primary_category_id = primary_category_id
                    existing.updated_at = record.updated_at
                    existing.doi = record.doi
                    existing.journal_ref = record.journal_ref
                    existing.url = record.url
                    existing.content_hash = record.content_hash
                    self._replace_links(session, existing.id, category_ids.values(), author_ids)
                    updated += 1

                else:
                    unchanged += 1

            session.commit()

        logger.info(
            "Upsert complete: %d fetched, %d inserted, %d updated, %d unchanged",
            len(records), inserted, updated, unchanged,
        )
        return UpsertResult(inserted=inserted, updated=updated, unchanged=unchanged, fetched=len(records))


    def _get_or_create_categories(self, session: Session, codes: Sequence[str]) -> dict[str, int]:
        result: dict[str, int] = {}

        for code in codes:
            category = session.scalar(select(Category).where(Category.code == code))
            if category is None:
                category = Category(code=code)
                session.add(category)
                session.flush()
            result[code] = category.id

        return result

    def _get_or_create_authors(self, session: Session, names: Sequence[str]) -> list[int]:
        ids: list[int] = []
        for name in names:
            normalized = normalize_author_name(name)
            author = session.scalar(select(Author).where(Author.name_normalized == normalized))

            if author is None:
                author = Author(name=name, name_normalized=normalized)
                session.add(author)
                session.flush()
            ids.append(author.id)
        return ids

    def _replace_links(
        self,
        session: Session,
        paper_id: int,
        category_ids: Iterable[int],
        author_ids: Sequence[int],
    ) -> None:
        session.execute(delete(PaperCategory).where(PaperCategory.paper_id == paper_id))
        session.execute(delete(PaperAuthor).where(PaperAuthor.paper_id == paper_id))

        for cid in category_ids:
            session.add(PaperCategory(paper_id=paper_id, category_id=cid))

        for position, aid in enumerate(author_ids):
            session.add(PaperAuthor(paper_id=paper_id, author_id=aid, position=position))

    # Embedding Tracking (Used by IndexingService)
    def get_papers_needing_embedding(self, model: str) -> list[tuple[int, str, str]]:
        with self._session_factory() as session:
            rows = session.execute(
                select(Paper.id, Paper.abstract, Paper.title, Paper.content_hash, PaperEmbedding.content_hash, PaperEmbedding.model)
                .outerjoin(PaperEmbedding, PaperEmbedding.paper_id == Paper.id)
            ).all()

        stale: list[tuple[int, str, str]] = []
        for paper_id, abstract, title, current_hash, embedded_hash, embedded_model in rows:
            if embedded_hash is None or embedded_hash != current_hash or embedded_model != model:
                text_for_embedding = f"{title}. {abstract}"
                stale.append((paper_id, text_for_embedding, current_hash))
        return stale

    def mark_embedded(self, paper_id: int, content_hash: str, model: str) -> None:
        with self._session_factory() as session:
            existing = session.get(PaperEmbedding, paper_id)
            if existing is None:
                session.add(PaperEmbedding(paper_id=paper_id, content_hash=content_hash, model=model))
            else:
                existing.content_hash = content_hash
                existing.model = model
            session.commit()

    def get_papers_by_ids(self, paper_ids: list[int]) -> dict[int, PaperView]:
        if not paper_ids:
            return {}

        with self._session_factory() as session:
            papers = session.scalars(select(Paper).where(Paper.id.in_(paper_ids))).all() 
            result = {}
            for p in papers:
                authors = (
                    session.execute(
                        select(Author.name)
                        .join(PaperAuthor, PaperAuthor.author_id == Author.id)
                        .where(PaperAuthor.paper_id == p.id)
                        .order_by(PaperAuthor.position)
                    )
                    .scalars()
                    .all()
                )
                result[p.id] = PaperView(
                    id=p.id, arxiv_id=p.arxiv_id, title=p.title, abstract=p.abstract,
                    authors=list(authors), url=p.url, published_at=p.published_at,
                )
            return result

    # Maintainance
    def wipe(self) -> None:
        with self._session_factory() as session:
            session.execute(delete(PaperEmbedding))
            session.execute(delete(PaperAuthor))
            session.execute(delete(PaperCategory))
            session.execute(delete(Paper))
            session.execute(delete(Author))
            session.execute(delete(Category))
            session.execute(delete(IngestionRun))
            session.commit()
        logger.warning("Dataset wiped: papers, authors, categories, embeddings, ingestion_runs")
