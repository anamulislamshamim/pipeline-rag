from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.config import Settings


def make_engine(settings: Settings):
    return create_engine(
        settings.database_url, 
        pool_pre_ping=True, # Check the connection is alive before use it
        future=True, # Use sqlalchemy 2.0 style class        
    )


def make_session_factory(settings: Settings) -> sessionmaker[Session]:
    engine = make_engine(settings)
    return sessionmaker(
        bind=engine, 
        autoflush=False, 
        expire_on_commit=False
    )


@contextmanager
def session_scope(session_factory: sessionmaker[Session]) -> Iterator[Session]:
    """Provides a transactional scope: commits on success, rolls back on error."""
    session = session_factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
