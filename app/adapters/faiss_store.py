"""FAISS-backed VectorStore adapter.

IndexIDMap2 wraps a flat inner-product index so FAISS ids equal our
`papers.id` directly - no separate id-mapping table needed. Vectors are
L2-normalized before insertion/search so inner product equals cosine
similarity. This is exact (brute-force) search: fine up to roughly a
million vectors at this dimensionality; the VectorStore port is what makes
swapping to an approximate index (HNSW/IVF) later a contained change.

Two things matter for correctness here, both required by the assignment:
- save() writes to a temp file then renames, so a crash mid-write never
  leaves a corrupt index on disk.
- reload_if_stale() lets a long-lived process (the API) notice that a
  separate process (the ingestion CLI) wrote a newer index file, without
  restarting.
"""
from __future__ import annotations

import os
import threading

import faiss
import numpy as np


class FaissVectorStore:
    def __init__(self, dim: int, index_path: str):
        self._dim = dim
        self._path = index_path
        self._lock = threading.Lock()
        self._mtime: float | None = None
        self._index = self._load_or_create()

    def _load_or_create(self):
        if os.path.exists(self._path):
            index = faiss.read_index(self._path)
            self._mtime = os.path.getmtime(self._path)
            return index
        base = faiss.IndexFlatIP(self._dim)
        return faiss.IndexIDMap2(base)

    def add(self, ids: list[int], vectors: list[list[float]]) -> None:
        if not ids:
            return
        arr = _normalize(np.array(vectors, dtype="float32"))
        id_arr = np.array(ids, dtype="int64")
        with self._lock:
            self._index.add_with_ids(arr, id_arr)

    def remove(self, ids: list[int]) -> None:
        if not ids:
            return
        id_arr = np.array(ids, dtype="int64")
        with self._lock:
            self._index.remove_ids(id_arr)

    def search(self, vector: list[float], top_k: int) -> list[tuple[int, float]]:
        with self._lock:
            if self._index.ntotal == 0:
                return []
            query = _normalize(np.array([vector], dtype="float32"))
            scores, ids = self._index.search(query, min(top_k, self._index.ntotal))

        results: list[tuple[int, float]] = []
        for score, idx in zip(scores[0], ids[0]):
            if idx == -1:  # FAISS pads with -1 when fewer than top_k results exist
                continue
            results.append((int(idx), float(score)))
        return results

    def save(self) -> None:
        os.makedirs(os.path.dirname(self._path) or ".", exist_ok=True)
        tmp_path = self._path + ".tmp"
        with self._lock:
            faiss.write_index(self._index, tmp_path)
        os.replace(tmp_path, self._path)  # atomic on POSIX
        self._mtime = os.path.getmtime(self._path)

    def reload_if_stale(self) -> None:
        if not os.path.exists(self._path):
            return
        mtime = os.path.getmtime(self._path)
        if self._mtime is None or mtime > self._mtime:
            with self._lock:
                self._index = faiss.read_index(self._path)
                self._mtime = mtime


def _normalize(arr: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0  # avoid divide-by-zero on an all-zero vector
    return arr / norms
