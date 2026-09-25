"""Google Gemini LLM adapter, implements the LLMClient port.

Plain REST via httpx rather than the google-genai SDK, to keep this
consistent with the Ollama adapter and avoid an extra dependency for a
single endpoint. Uses `x-goog-api-key` header auth and
generationConfig.responseMimeType="application/json" to constrain output.

Note: Google's Gemini model lineup changes over time. If GEMINI_MODEL 404s,
list what's available to your key with:
  curl -H "x-goog-api-key: $GEMINI_API_KEY" \\
       https://generativelanguage.googleapis.com/v1beta/models
and set GEMINI_MODEL in .env to one that supports generateContent.
"""
from __future__ import annotations

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential


class GeminiError(Exception):
    """Raised on connection failures or 429/5xx responses, so tenacity retries only these."""


class GeminiLLMClient:
    def __init__(
        self,
        api_key: str,
        model: str,
        http_client: httpx.Client | None = None,
        timeout: float = 60.0,
    ):
        if not api_key:
            raise ValueError("GEMINI_API_KEY is required when LLM_PROVIDER=gemini")
        self._api_key = api_key
        self._model = model
        self._base_url = "https://generativelanguage.googleapis.com/v1beta"
        self._http = http_client or httpx.Client(timeout=timeout)

    @retry(
        retry=retry_if_exception_type(GeminiError),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    def complete_json(self, system: str, user: str) -> str:
        url = f"{self._base_url}/models/{self._model}:generateContent"
        payload = {
            "system_instruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {"temperature": 0, "responseMimeType": "application/json"},
        }
        headers = {"x-goog-api-key": self._api_key, "Content-Type": "application/json"}

        try:
            response = self._http.post(url, json=payload, headers=headers)
        except httpx.TransportError as exc:
            raise GeminiError(f"Could not reach Gemini API: {exc}") from exc

        if response.status_code == 429 or response.status_code >= 500:
            raise GeminiError(f"Gemini returned {response.status_code}: {response.text[:300]}")
        response.raise_for_status()  # 4xx (bad key/model) is not transient - don't retry

        data = response.json()
        try:
            return data["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError, TypeError) as exc:
            raise GeminiError(f"Unexpected Gemini response shape: {data}") from exc
