"""Ollama LLM adapter, implements the LLMClient port.

Talks to a locally-running Ollama server's /api/chat endpoint. Uses
Ollama's `format: "json"` option to constrain output to valid JSON where
the model supports it, and temperature 0 for deterministic, less
hallucination-prone answers.

Retries only wrap transient connection/5xx failures. A malformed or
out-of-range-citation JSON response is NOT retried here - that's handled
one layer up, by the RAG state machine (ValidatingState), because only the
pipeline knows the retrieved context needed to re-prompt correctly.
"""
from __future__ import annotations

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential


class OllamaError(Exception):
    """Raised on connection failures or 5xx responses, so tenacity retries only these."""


class OllamaLLMClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        http_client: httpx.Client | None = None,
        timeout: float = 180.0,  # local CPU inference can be slow; generous default
    ):
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._http = http_client or httpx.Client(timeout=timeout)

    @retry(
        retry=retry_if_exception_type(OllamaError),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    def complete_json(self, system: str, user: str) -> str:
        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "format": "json",
            "stream": False,
            "options": {"temperature": 0},
        }
        try:
            response = self._http.post(f"{self._base_url}/api/chat", json=payload)
        except httpx.TransportError as exc:
            raise OllamaError(
                f"Could not reach Ollama at {self._base_url} - is it running on the host "
                f"and is OLLAMA_BASE_URL correct? ({exc})"
            ) from exc

        if response.status_code >= 500:
            raise OllamaError(f"Ollama returned {response.status_code}: {response.text[:300]}")
        response.raise_for_status()  # 4xx (e.g. model not pulled) is not transient - don't retry

        data = response.json()
        try:
            return data["message"]["content"]
        except (KeyError, TypeError) as exc:
            raise OllamaError(f"Unexpected Ollama response shape: {data}") from exc
