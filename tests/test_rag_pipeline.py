"""Tests the RAG state machine end-to-end with fakes only at the port
boundary (Embedder, VectorStore, PaperRepository, LLMClient). Everything
else - Retriever, ContextBuilder, Generator, AnswerValidator, the states,
RAGPipeline itself - is real production code. No DB, no FAISS, no network,
no LLM required to run this file.
"""
import unittest
from datetime import datetime, timezone

from app.domain.models import Answer
from app.ports import PaperView
from app.services.rag.context_builder import ContextBuilder
from app.services.rag.deps import RagDeps
from app.services.rag.generator import Generator
from app.services.rag.pipeline import RAGPipeline
from app.services.rag.retriever import Retriever
from app.services.rag.validator import AnswerValidator


class FakeEmbedder:
    dim = 3
    model_name = "fake-embedder"

    def embed_passages(self, texts):
        return [[0.1, 0.2, 0.3] for _ in texts]

    def embed_query(self, text):
        return [0.1, 0.2, 0.3]


class FakeVectorStore:
    """search() ignores the query vector and returns a preset list of
    (paper_id, score) hits - the tests control relevance by setting this."""

    def __init__(self, hits):
        self.hits = hits

    def add(self, ids, vectors):
        pass

    def remove(self, ids):
        pass

    def search(self, vector, top_k):
        return self.hits[:top_k]

    def save(self):
        pass

    def reload_if_stale(self):
        pass


class FakeRepository:
    def __init__(self, papers: dict[int, PaperView]):
        self._papers = papers

    def get_papers_by_ids(self, paper_ids):
        return {pid: self._papers[pid] for pid in paper_ids if pid in self._papers}

    # Unused by the RAG pipeline, present only to satisfy the port shape.
    def upsert_papers(self, records):
        raise NotImplementedError

    def get_papers_needing_embedding(self, model):
        raise NotImplementedError

    def mark_embedded(self, paper_id, content_hash, model):
        raise NotImplementedError

    def wipe(self):
        raise NotImplementedError


class FakeLLMClient:
    """Returns each string in `responses` in order, one per call. Records
    how many times it was called so tests can assert the LLM was (or
    crucially, was NOT) invoked."""

    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self.call_count = 0

    def complete_json(self, system, user):
        self.call_count += 1
        return self._responses.pop(0)


def make_paper_view(paper_id: int) -> PaperView:
    return PaperView(
        id=paper_id,
        arxiv_id=f"2401.{paper_id:05d}",
        title=f"Paper {paper_id}",
        abstract=f"This is the abstract of paper {paper_id}, about retrieval-augmented generation.",
        authors=["A. Author"],
        url=f"http://arxiv.org/abs/2401.{paper_id:05d}",
        published_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
    )


def build_pipeline(hits, papers_by_id, llm_responses, min_score=0.35, max_attempts=2) -> tuple[RAGPipeline, FakeLLMClient]:
    embedder = FakeEmbedder()
    vector_store = FakeVectorStore(hits)
    repository = FakeRepository(papers_by_id)
    llm = FakeLLMClient(llm_responses)

    deps = RagDeps(
        retriever=Retriever(embedder, vector_store, repository),
        context_builder=ContextBuilder(reranker=None),
        generator=Generator(llm),
        validator=AnswerValidator(),
        top_k=10,
        top_n=5,
        min_score=min_score,
        max_context_tokens=3000,
        max_attempts=max_attempts,
    )
    return RAGPipeline(deps), llm


class TestRagPipelineNoHallucination(unittest.TestCase):
    def test_no_hits_at_all_never_calls_llm(self):
        pipeline, llm = build_pipeline(hits=[], papers_by_id={}, llm_responses=[])

        answer = pipeline.ask("What is retrieval-augmented generation?")

        self.assertFalse(answer.grounded)
        self.assertEqual(answer.sources, [])
        self.assertEqual(llm.call_count, 0)

    def test_hits_below_score_threshold_never_calls_llm(self):
        pipeline, llm = build_pipeline(
            hits=[(1, 0.1), (2, 0.2)],  # both below min_score=0.35
            papers_by_id={1: make_paper_view(1), 2: make_paper_view(2)},
            llm_responses=[],
            min_score=0.35,
        )

        answer = pipeline.ask("An unrelated question")

        self.assertFalse(answer.grounded)
        self.assertEqual(llm.call_count, 0)

    def test_happy_path_returns_grounded_answer_with_correct_source(self):
        pipeline, llm = build_pipeline(
            hits=[(1, 0.9)],
            papers_by_id={1: make_paper_view(1)},
            llm_responses=['{"answer": "RAG grounds LLM answers in retrieved text [1].", "citations": [1], "sufficient": true}'],
        )

        answer = pipeline.ask("What is RAG?")

        self.assertTrue(answer.grounded)
        self.assertEqual(llm.call_count, 1)
        self.assertEqual(len(answer.sources), 1)
        self.assertEqual(answer.sources[0].arxiv_id, "2401.00001")

    def test_malformed_json_is_retried_once_then_succeeds(self):
        pipeline, llm = build_pipeline(
            hits=[(1, 0.9)],
            papers_by_id={1: make_paper_view(1)},
            llm_responses=[
                "this is not json at all",
                '{"answer": "Fixed on retry [1].", "citations": [1], "sufficient": true}',
            ],
            max_attempts=2,
        )

        answer = pipeline.ask("What is RAG?")

        self.assertTrue(answer.grounded)
        self.assertEqual(llm.call_count, 2)  # proves the retry actually happened

    def test_out_of_range_citation_triggers_retry_not_a_fabricated_source(self):
        pipeline, llm = build_pipeline(
            hits=[(1, 0.9)],  # only ONE chunk will be in context
            papers_by_id={1: make_paper_view(1)},
            llm_responses=[
                '{"answer": "See [1] and [2].", "citations": [1, 2], "sufficient": true}',  # [2] doesn't exist
                '{"answer": "See [1].", "citations": [1], "sufficient": true}',
            ],
            max_attempts=2,
        )

        answer = pipeline.ask("What is RAG?")

        self.assertTrue(answer.grounded)
        self.assertEqual(llm.call_count, 2)
        self.assertEqual(len(answer.sources), 1)

    def test_exhausted_retries_gives_up_honestly_instead_of_hallucinating(self):
        pipeline, llm = build_pipeline(
            hits=[(1, 0.9)],
            papers_by_id={1: make_paper_view(1)},
            llm_responses=["not json", "still not json", "still not json"],
            max_attempts=2,
        )

        answer = pipeline.ask("What is RAG?")

        self.assertFalse(answer.grounded)
        self.assertEqual(answer.sources, [])
        self.assertEqual(llm.call_count, 2)  # stopped at max_attempts, never called a 3rd time

    def test_model_saying_insufficient_is_honored_without_retrying(self):
        pipeline, llm = build_pipeline(
            hits=[(1, 0.9)],
            papers_by_id={1: make_paper_view(1)},
            llm_responses=['{"answer": "The papers do not cover this.", "citations": [], "sufficient": false}'],
        )

        answer = pipeline.ask("What is RAG?")

        self.assertFalse(answer.grounded)
        self.assertEqual(answer.sources, [])
        self.assertEqual(llm.call_count, 1)  # honest admission, not an error - no retry needed

    def test_stale_faiss_id_missing_from_db_is_skipped_not_fatal(self):
        # Simulates a FAISS entry pointing at a paper deleted from Postgres
        # without a matching reindex.
        pipeline, llm = build_pipeline(
            hits=[(1, 0.9), (999, 0.8)],  # 999 has no matching PaperView
            papers_by_id={1: make_paper_view(1)},
            llm_responses=['{"answer": "See [1].", "citations": [1], "sufficient": true}'],
        )

        answer = pipeline.ask("What is RAG?")

        self.assertTrue(answer.grounded)
        self.assertEqual(len(answer.sources), 1)


if __name__ == "__main__":
    unittest.main()
