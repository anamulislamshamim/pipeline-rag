from __future__ import annotations

from app.ports import LLMClient
from app.services.rag.prompt import SYSTEM_PROMPT, build_user_prompt


class Generator:
    def __init__(self, llm_client: LLMClient):
        self._llm_client = llm_client

    def generate(self, question: str, context_text: str, correction: str | None = None) -> str:
        user_prompt = build_user_prompt(question, context_text, correction)
        return self._llm_client.complete_json(system=SYSTEM_PROMPT, user=user_prompt)
