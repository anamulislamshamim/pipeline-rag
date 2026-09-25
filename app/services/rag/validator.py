"""Validates and parses the LLM's raw JSON response into a domain Answer.

This is the second of two anti-hallucination gates (the first is the
retrieval score threshold in RetrievingState, which stops the LLM from
being called at all when nothing relevant was found):

- Malformed JSON -> ValidationError -> the pipeline retries generation once.
- A citation index outside the retrieved set -> ValidationError -> retry.
  This specifically prevents the model from inventing a source.
- The model's own "sufficient": false is NOT an error - it's the model
  honestly saying the papers don't answer the question, so it's returned
  as an ungrounded Answer immediately, no retry needed.
- On success, `sources` includes only the chunks actually cited, not every
  chunk that was in context.
"""
from __future__ import annotations

import json
import re

from app.domain.models import Answer, RetrievedChunk

_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


class ValidationError(Exception):
    """Raised when the LLM's response can't be trusted as-is; triggers a retry."""


class AnswerValidator:
    def validate(self, raw: str, used_chunks: list[RetrievedChunk]) -> Answer:
        data = self._parse_json(raw)

        if "answer" not in data:
            raise ValidationError("response is missing the 'answer' field")
        answer_text = str(data["answer"]).strip()
        if not answer_text:
            raise ValidationError("'answer' field is empty")

        sufficient = bool(data.get("sufficient", True))
        citations = self._parse_citations(data.get("citations", []), len(used_chunks))

        if not sufficient:
            # The model itself says the context doesn't answer the question.
            # Trust that rather than treating it as malformed output.
            return Answer(answer=answer_text, grounded=False, sources=[])

        if not citations:
            raise ValidationError("sufficient=true but no citations were given")

        sources = [used_chunks[i - 1] for i in sorted(set(citations))]
        return Answer(answer=answer_text, grounded=True, sources=sources)

    def _parse_json(self, raw: str) -> dict:
        cleaned = _CODE_FENCE_RE.sub("", raw.strip())
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise ValidationError(f"response is not valid JSON: {exc}") from exc
        if not isinstance(data, dict):
            raise ValidationError("response JSON must be an object")
        return data

    def _parse_citations(self, raw_citations: object, num_chunks: int) -> list[int]:
        if not isinstance(raw_citations, list):
            raise ValidationError("'citations' must be a list of integers")
        citations: list[int] = []
        for item in raw_citations:
            try:
                index = int(item)
            except (TypeError, ValueError) as exc:
                raise ValidationError(f"citation '{item}' is not an integer") from exc
            if not (1 <= index <= num_chunks):
                raise ValidationError(
                    f"citation [{index}] is out of range - only [1..{num_chunks}] were provided"
                )
            citations.append(index)
        return citations
