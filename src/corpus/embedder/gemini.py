"""Gemini embeddings via google-genai SDK.

Default model: `gemini-embedding-001` — supports asymmetric retrieval via
`task_type` and Matryoshka-style dimension trimming (768 / 1536 / 3072
all valid). Free tier available through AI Studio (~1500 requests/day);
paid tier via Vertex AI for production-scale workloads.

Install: `pip install corpus-rag[gemini]` or `uv add google-genai`.

Auth: set `GEMINI_API_KEY` or `GOOGLE_API_KEY` in env. AI Studio key from
https://aistudio.google.com/apikey is the simplest path for personal use.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Sequence
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Gemini limits as of 2026 (verify on ai.google.dev/docs):
#   - 100 inputs per batch_embed_contents call
#   - 2048 tokens per text (longer texts truncated server-side when truncation enabled)
# Conservative char budget per batch: 250K chars (~62K tokens estimate).
MAX_INPUTS_PER_BATCH = 100
MAX_CHARS_PER_BATCH = 250_000


@dataclass(frozen=True)
class _EmbedResult:
    embeddings: list[list[float]]


class GeminiEmbedder:
    def __init__(
        self,
        model: str = "gemini-embedding-001",
        api_key: str | None = None,
        dim: int | None = None,
    ):
        key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not key:
            raise RuntimeError(
                "GEMINI_API_KEY (or GOOGLE_API_KEY) missing. Set it in .env or "
                "pass api_key= to GeminiEmbedder. Get a key at "
                "https://aistudio.google.com/apikey."
            )
        try:
            from google import genai
        except ImportError as e:
            raise ImportError(
                "Gemini provider requires the [gemini] extra. "
                "Install with `pip install corpus-rag[gemini]` or `uv add google-genai`."
            ) from e

        self._client = genai.Client(api_key=key)
        self._model = model
        self._dim = dim  # None = use model's native dim
        self.total_tokens_used = 0  # Gemini API doesn't return per-call usage; we don't pretend

    def embed_documents(self, texts: Sequence[str]) -> list[list[float] | None]:
        return self._embed_with_task_type(list(texts), task_type="RETRIEVAL_DOCUMENT")

    def embed_query(self, text: str) -> list[float]:
        if not text or not text.strip():
            raise ValueError("Cannot embed empty query")
        result = self._embed_batch([text], task_type="RETRIEVAL_QUERY")
        return result.embeddings[0]

    def _embed_with_task_type(
        self, texts: list[str], task_type: str
    ) -> list[list[float] | None]:
        live_indices: list[int] = []
        live_texts: list[str] = []
        for i, t in enumerate(texts):
            if t and t.strip():
                live_indices.append(i)
                live_texts.append(t)

        if not live_texts:
            return [None] * len(texts)

        results: list[list[float] | None] = [None] * len(texts)
        for batch_indices, batch_texts in self._pack_batches(live_indices, live_texts):
            result = self._embed_batch(batch_texts, task_type=task_type)
            for idx, vec in zip(batch_indices, result.embeddings, strict=True):
                results[idx] = vec
        return results

    def _pack_batches(
        self, indices: list[int], texts: list[str]
    ) -> list[tuple[list[int], list[str]]]:
        batches: list[tuple[list[int], list[str]]] = []
        cur_idx: list[int] = []
        cur_txt: list[str] = []
        cur_chars = 0
        for i, t in zip(indices, texts, strict=True):
            t_chars = len(t)
            if cur_txt and (
                len(cur_txt) >= MAX_INPUTS_PER_BATCH
                or cur_chars + t_chars > MAX_CHARS_PER_BATCH
            ):
                batches.append((cur_idx, cur_txt))
                cur_idx, cur_txt, cur_chars = [], [], 0
            cur_idx.append(i)
            cur_txt.append(t)
            cur_chars += t_chars
        if cur_txt:
            batches.append((cur_idx, cur_txt))
        return batches

    def _embed_batch(self, texts: list[str], task_type: str) -> _EmbedResult:
        from google.genai import types as gtypes

        config = gtypes.EmbedContentConfig(task_type=task_type)
        if self._dim is not None:
            config.output_dimensionality = self._dim

        response = self._client.models.embed_content(
            model=self._model,
            contents=texts,
            config=config,
        )
        # Response shape: response.embeddings = [ContentEmbedding(values=[...]), ...]
        return _EmbedResult(embeddings=[e.values for e in response.embeddings])
