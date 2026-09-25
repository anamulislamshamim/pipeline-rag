"""Strategy interface for chart aggregations.

Each aggregation is a small, independently-instantiable class implementing
this shape, registered by name in registry.py. Adding a new chart means
writing one new class + adding one registry line - no route or dispatch
code changes (Open/Closed), the same pattern as app/adapters/llm_factory.py
for LLM providers.
"""
from __future__ import annotations

from typing import Protocol

from sqlalchemy.orm import Session


class Aggregation(Protocol):
    name: str

    def run(self, session: Session, **params: object) -> list[dict]:
        """Run the aggregation and return rows as plain JSON-serializable dicts.

        Implementations should accept and ignore any params they don't use
        (**_ catch-all) so the route can pass a uniform param set to every
        aggregation without knowing which ones are relevant to which.
        """
        ...
