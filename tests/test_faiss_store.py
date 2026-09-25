"""Tests for FaissVectorStore. Requires faiss-cpu/numpy - run with:
    docker compose run --rm api pytest tests/test_faiss_store.py -v
(Not runnable outside the container; these packages aren't part of the
lightweight Day-1 dependency set used for pure-logic tests.)
"""
import os
import tempfile
import unittest

from app.adapters.faiss_store import FaissVectorStore


class TestFaissVectorStore(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.index_path = os.path.join(self.tmpdir.name, "test.index")

    def tearDown(self):
        self.tmpdir.cleanup()

    def _vec(self, *values):
        # Pad/truncate to dim=4 for simplicity across tests.
        v = list(values) + [0.0] * 4
        return v[:4]

    def test_add_and_search_returns_best_match_first(self):
        store = FaissVectorStore(dim=4, index_path=self.index_path)
        store.add(
            ids=[1, 2, 3],
            vectors=[self._vec(1, 0, 0, 0), self._vec(0, 1, 0, 0), self._vec(0.9, 0.1, 0, 0)],
        )

        results = store.search(self._vec(1, 0, 0, 0), top_k=3)

        self.assertEqual(len(results), 3)
        self.assertEqual(results[0][0], 1)  # exact match should rank first
        self.assertGreater(results[0][1], results[1][1])

    def test_search_on_empty_index_returns_empty_list(self):
        store = FaissVectorStore(dim=4, index_path=self.index_path)
        self.assertEqual(store.search(self._vec(1, 0, 0, 0), top_k=5), [])

    def test_remove_then_search_excludes_removed_id(self):
        store = FaissVectorStore(dim=4, index_path=self.index_path)
        store.add(ids=[1, 2], vectors=[self._vec(1, 0, 0, 0), self._vec(0, 1, 0, 0)])

        store.remove([1])
        results = store.search(self._vec(1, 0, 0, 0), top_k=5)

        self.assertNotIn(1, [pid for pid, _ in results])
        self.assertIn(2, [pid for pid, _ in results])

    def test_save_then_reload_from_disk_preserves_data(self):
        store = FaissVectorStore(dim=4, index_path=self.index_path)
        store.add(ids=[1, 2], vectors=[self._vec(1, 0, 0, 0), self._vec(0, 1, 0, 0)])
        store.save()

        self.assertTrue(os.path.exists(self.index_path))

        reloaded = FaissVectorStore(dim=4, index_path=self.index_path)
        results = reloaded.search(self._vec(1, 0, 0, 0), top_k=2)
        self.assertEqual({pid for pid, _ in results}, {1, 2})

    def test_reload_if_stale_picks_up_changes_from_another_writer(self):
        writer = FaissVectorStore(dim=4, index_path=self.index_path)
        writer.add(ids=[1], vectors=[self._vec(1, 0, 0, 0)])
        writer.save()

        reader = FaissVectorStore(dim=4, index_path=self.index_path)
        self.assertEqual(len(reader.search(self._vec(1, 0, 0, 0), top_k=5)), 1)

        # A second process adds more data and saves...
        writer.add(ids=[2], vectors=[self._vec(0, 1, 0, 0)])
        writer.save()

        # ...the reader shouldn't see it until it explicitly reloads.
        stale_results = reader.search(self._vec(0, 1, 0, 0), top_k=5)
        self.assertEqual({pid for pid, _ in stale_results}, {1})

        reader.reload_if_stale()
        fresh_results = reader.search(self._vec(0, 1, 0, 0), top_k=5)
        self.assertEqual({pid for pid, _ in fresh_results}, {1, 2})

    def test_re_adding_same_id_does_not_duplicate(self):
        store = FaissVectorStore(dim=4, index_path=self.index_path)
        store.add(ids=[1], vectors=[self._vec(1, 0, 0, 0)])
        # Simulate IndexingService's re-embed pattern: remove then re-add.
        store.remove([1])
        store.add(ids=[1], vectors=[self._vec(0, 1, 0, 0)])

        results = store.search(self._vec(0, 1, 0, 0), top_k=10)
        matches = [pid for pid, _ in results if pid == 1]
        self.assertEqual(len(matches), 1)


if __name__ == "__main__":
    unittest.main()
