import unittest
from datetime import datetime, timezone

from app.domain.models import RetrievedChunk
from app.services.rag.validator import AnswerValidator, ValidationError


def make_chunk(paper_id: int) -> RetrievedChunk:
    return RetrievedChunk(
        paper_id=paper_id,
        arxiv_id=f"2401.{paper_id:05d}",
        title=f"Paper {paper_id}",
        abstract="abstract",
        authors=["A. Author"],
        url="http://arxiv.org/abs/x",
        published_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        score=0.9,
    )


class TestAnswerValidator(unittest.TestCase):
    def setUp(self):
        self.validator = AnswerValidator()
        self.chunks = [make_chunk(1), make_chunk(2)]

    def test_valid_response_with_citation(self):
        raw = '{"answer": "X does Y [1].", "citations": [1], "sufficient": true}'
        answer = self.validator.validate(raw, self.chunks)
        self.assertTrue(answer.grounded)
        self.assertEqual(len(answer.sources), 1)
        self.assertEqual(answer.sources[0].paper_id, 1)

    def test_strips_markdown_code_fences(self):
        raw = '```json\n{"answer": "X [1].", "citations": [1], "sufficient": true}\n```'
        answer = self.validator.validate(raw, self.chunks)
        self.assertTrue(answer.grounded)

    def test_multiple_citations_deduplicated_and_ordered(self):
        raw = '{"answer": "X [2] and [1] and [1] again.", "citations": [2, 1, 1], "sufficient": true}'
        answer = self.validator.validate(raw, self.chunks)
        self.assertEqual([s.paper_id for s in answer.sources], [1, 2])

    def test_sufficient_false_is_not_an_error(self):
        raw = '{"answer": "Not covered by these papers.", "citations": [], "sufficient": false}'
        answer = self.validator.validate(raw, self.chunks)
        self.assertFalse(answer.grounded)
        self.assertEqual(answer.sources, [])

    def test_missing_answer_key_raises(self):
        raw = '{"citations": [1], "sufficient": true}'
        with self.assertRaises(ValidationError):
            self.validator.validate(raw, self.chunks)

    def test_invalid_json_raises(self):
        with self.assertRaises(ValidationError):
            self.validator.validate("not json at all", self.chunks)

    def test_citation_out_of_range_raises(self):
        raw = '{"answer": "X [5].", "citations": [5], "sufficient": true}'
        with self.assertRaises(ValidationError):
            self.validator.validate(raw, self.chunks)

    def test_citation_zero_raises(self):
        raw = '{"answer": "X [0].", "citations": [0], "sufficient": true}'
        with self.assertRaises(ValidationError):
            self.validator.validate(raw, self.chunks)

    def test_citations_not_a_list_raises(self):
        raw = '{"answer": "X.", "citations": "1", "sufficient": true}'
        with self.assertRaises(ValidationError):
            self.validator.validate(raw, self.chunks)

    def test_sufficient_true_but_no_citations_raises(self):
        raw = '{"answer": "X.", "citations": [], "sufficient": true}'
        with self.assertRaises(ValidationError):
            self.validator.validate(raw, self.chunks)


if __name__ == "__main__":
    unittest.main()
