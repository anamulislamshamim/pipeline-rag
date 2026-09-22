from __future__ import annotations
from datetime import datetime

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    SmallInteger,
    String,
    Text,
    UniqueConstraint,
    func
)
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.db.base import Base


class Category(Base):
    __tablename__ = "categories"

    id: Mapped[int] = mapped_column(primary_key=True)
    code: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)


class Author(Base):
    __tablename__ = "authors"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    name_normalized: Mapped[str] = mapped_column(Text, unique=True, nullable=False)


class Paper(Base):
    __tablename__ = "papers"

    id: Mapped[int] = mapped_column(primary_key=True)
    arxiv_id: Mapped[str] = mapped_column(String(32), unique=True, nullable=False, index=True)
    version: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    abstract: Mapped[str] = mapped_column(Text, nullable=False)

    primary_category_id: Mapped[int] = mapped_column(ForeignKey("categories.id"), nullable=False)
    primary_category: Mapped["Category"] = relationship(foreign_keys=[primary_category_id])

    published_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    doi: Mapped[str | None] = mapped_column(Text, nullable=True)
    journal_ref: Mapped[str] = mapped_column(Text, nullable=True)
    url: Mapped[str] = mapped_column(Text, nullable=False)

    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), onupdate=func.now())
    synced_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now()
    )

    categories: Mapped[list["Category"]] = relationship(
        secondary="paper_categories", viewonly=True
    )
    embedding: Mapped["PaperEmbedding | None"] = relationship(
        back_populates="paper", uselist=False, cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("ix_papers_published", "published_at"),
        Index("ix_papers_primary_cat", "primary_category_id"),
    )


# Intermediary Table: Many-to-Many relationship
class PaperCategory(Base):
    __tablename__ = "paper_categories"

    paper_id: Mapped[int] = mapped_column(ForeignKey("papers.id", ondelete="CASCADE"), primary_key=True)
    category_id: Mapped[int] = mapped_column(ForeignKey("categories.id"), primary_key=True)

    __table_args__ = (
        Index(
            "ix_pc_category",
            "category_id",
            "paper_id",
        ),
    )


# Intermediary Table: Many-to-Many relationship
class PaperAuthor(Base):
    __tablename__ = "paper_authors"

    paper_id: Mapped[int] = mapped_column(ForeignKey("papers.id", ondelete="CASCADE"), primary_key=True)
    author_id: Mapped[int] = mapped_column(ForeignKey("authors.id"), primary_key=True)
    position: Mapped[int] = mapped_column(SmallInteger, nullable=False)

    __table_args__ = (
        UniqueConstraint("paper_id", "position", name="uq_paper_author_position"),
        Index("ix_pa_author", "author_id"),
    )


# One-to-one relationship
class PaperEmbedding(Base):
    """
    Tracks which papers are currently reflected in the FAISS index and 
    what content hash / model, so the indexer can compute a re-embed diff.
    """

    __tablename__ = "paper_embeddings"

    paper_id: Mapped[int] = mapped_column(
        ForeignKey("papers.id", ondelete="CASCADE"), primary_key=True
    )
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    model: Mapped[str] = mapped_column(String(128), nullable=False)
    embedded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    paper: Mapped["Paper"] = relationship(back_populates="embedding")


class IngestionRun(Base):
    __tablename__ = "ingestion_runs"

    id: Mapped[int] = mapped_column(primary_key=True)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    finished_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=True)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="running")
    fetched: Mapped[int] = mapped_column(default=0)
    inserted: Mapped[int] = mapped_column(default=0)
    updated: Mapped[int] = mapped_column(default=0)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)

    __table_args__ = (
        CheckConstraint("status in ('running', 'success', 'failed')", name="ck_ingestion_status"),
    )
