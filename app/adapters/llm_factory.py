"""Factory for LLMClient implementations.

This is the one place that knows about concrete LLM adapters. Everything
else in the RAG pipeline depends on the LLMClient Protocol from app.ports.

To add a new provider (OpenAI, a different local model server, etc.):
  1. Write app/adapters/llm_<provider>.py implementing LLMClient.
  2. Add one line to _REGISTRY below.
No other file changes - this is the Open/Closed principle in practice, and
directly why swapping LLM_PROVIDER in .env (ollama <-> gemini) requires
zero code changes.
"""
from __future__ import annotations

from app.adapters.llm_gemini import GeminiLLMClient
from app.adapters.llm_ollama import OllamaLLMClient
from app.config import Settings
from app.ports import LLMClient

_REGISTRY = {
    "ollama": lambda s: OllamaLLMClient(base_url=s.ollama_base_url, model=s.ollama_model),
    "gemini": lambda s: GeminiLLMClient(api_key=s.gemini_api_key, model=s.gemini_model),
}


def build_llm_client(settings: Settings) -> LLMClient:
    provider = settings.llm_provider.lower()
    try:
        factory = _REGISTRY[provider]
    except KeyError as exc:
        raise ValueError(
            f"Unknown LLM_PROVIDER '{settings.llm_provider}'. Supported: {sorted(_REGISTRY)}"
        ) from exc
    return factory(settings)
