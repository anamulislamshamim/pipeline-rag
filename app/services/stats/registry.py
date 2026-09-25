"""Registry: name -> Aggregation instance.

To add a new chart: write a class in aggregations.py implementing
Aggregation, then add one line here. No changes needed in
app/api/routes_stats.py - this is the Open/Closed principle in practice.
"""
from __future__ import annotations

from app.services.stats.aggregations import (
    AvgAuthorsPerPaper,
    PapersByCategoryOverTime,
    TopAuthors,
    TopCategories,
)
from app.services.stats.base import Aggregation

_REGISTRY: dict[str, Aggregation] = {
    "papers-by-category-over-time": PapersByCategoryOverTime(),
    "top-categories": TopCategories(),
    "top-authors": TopAuthors(),
    "avg-authors-per-paper": AvgAuthorsPerPaper(),
}


def get_aggregation(name: str) -> Aggregation:
    try:
        return _REGISTRY[name]
    except KeyError as exc:
        raise KeyError(f"Unknown aggregation '{name}'. Available: {sorted(_REGISTRY)}") from exc


def list_aggregations() -> list[str]:
    return sorted(_REGISTRY)
