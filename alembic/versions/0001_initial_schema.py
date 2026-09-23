"""initial schema

Revision ID: 0001
Revises:
Create Date: 2026-09-22
"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "categories",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("code", sa.String(length=64), nullable=False, unique=True),
    )

    op.create_table(
        "authors",
        sa.Column("id", sa.BigInteger(), primary_key=True),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("name_normalized", sa.Text(), nullable=False, unique=True),
    )

    op.create_table(
        "papers",
        sa.Column("id", sa.BigInteger(), primary_key=True),
        sa.Column("arxiv_id", sa.String(length=32), nullable=False, unique=True),
        sa.Column("version", sa.SmallInteger(), nullable=False),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("abstract", sa.Text(), nullable=False),
        sa.Column("primary_category_id", sa.Integer(), sa.ForeignKey("categories.id"), nullable=False),
        sa.Column("published_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("doi", sa.Text(), nullable=True),
        sa.Column("journal_ref", sa.Text(), nullable=True),
        sa.Column("url", sa.Text(), nullable=False),
        sa.Column("content_hash", sa.String(length=64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("synced_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_papers_arxiv_id", "papers", ["arxiv_id"])
    op.create_index("ix_papers_published", "papers", ["published_at"])
    op.create_index("ix_papers_primary_cat", "papers", ["primary_category_id"])

    op.create_table(
        "paper_categories",
        sa.Column("paper_id", sa.BigInteger(), sa.ForeignKey("papers.id", ondelete="CASCADE"), primary_key=True),
        sa.Column("category_id", sa.Integer(), sa.ForeignKey("categories.id"), primary_key=True),
    )
    op.create_index("ix_pc_category", "paper_categories", ["category_id", "paper_id"])

    op.create_table(
        "paper_authors",
        sa.Column("paper_id", sa.BigInteger(), sa.ForeignKey("papers.id", ondelete="CASCADE"), primary_key=True),
        sa.Column("author_id", sa.BigInteger(), sa.ForeignKey("authors.id"), primary_key=True),
        sa.Column("position", sa.SmallInteger(), nullable=False),
        sa.UniqueConstraint("paper_id", "position", name="uq_paper_author_position"),
    )
    op.create_index("ix_pa_author", "paper_authors", ["author_id"])

    op.create_table(
        "paper_embeddings",
        sa.Column("paper_id", sa.BigInteger(), sa.ForeignKey("papers.id", ondelete="CASCADE"), primary_key=True),
        sa.Column("content_hash", sa.String(length=64), nullable=False),
        sa.Column("model", sa.String(length=128), nullable=False),
        sa.Column("embedded_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    op.create_table(
        "ingestion_runs",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("started_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("status", sa.String(length=16), nullable=False, server_default="running"),
        sa.Column("fetched", sa.Integer(), server_default="0"),
        sa.Column("inserted", sa.Integer(), server_default="0"),
        sa.Column("updated", sa.Integer(), server_default="0"),
        sa.Column("error", sa.Text(), nullable=True),
        sa.CheckConstraint("status in ('running','success','failed')", name="ck_ingestion_status"),
    )


def downgrade() -> None:
    op.drop_table("ingestion_runs")
    op.drop_table("paper_embeddings")
    op.drop_index("ix_pa_author", table_name="paper_authors")
    op.drop_table("paper_authors")
    op.drop_index("ix_pc_category", table_name="paper_categories")
    op.drop_table("paper_categories")
    op.drop_index("ix_papers_primary_cat", table_name="papers")
    op.drop_index("ix_papers_published", table_name="papers")
    op.drop_index("ix_papers_arxiv_id", table_name="papers")
    op.drop_table("papers")
    op.drop_table("authors")
    op.drop_table("categories")
