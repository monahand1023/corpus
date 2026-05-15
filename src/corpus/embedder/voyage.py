"""Voyage embeddings — batched, retried, asymmetric document/query input types.

The Voyage SDK's `max_retries` handles transient failures via tenacity. We
add: empty-string filtering (Voyage 400s), greedy batch packing under their
per-request limits, and recursive split on `InvalidRequestError` (oversize
batch) until it fits.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Sequence
from dataclasses import dataclass

import voyageai
import voyageai.error as ve

logger = logging.getLogger(__name__)

# Voyage limits as of 2026 (verify on docs.voyageai.com — they've changed mid-0.x):
#   - up to 128 inputs per /embed request
#   - up to ~120K tokens per /embed request
# Conservative 400K-char batch budget (~100K tokens estimate) leaves headroom.
MAX_INPUTS_PER_BATCH = 128
MAX_CHARS_PER_BATCH = 400_000


@dataclass(frozen=True)
class EmbedResult:
    embeddings: list[list[float]]
    total_tokens: int


class VoyageEmbedder:
    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        max_retries: int = 5,
        timeout: float = 60.0,
    ):
        key = api_key or os.environ.get("VOYAGE_API_KEY")
        if not key:
            raise RuntimeError(
                "VOYAGE_API_KEY missing. Set it in .env or pass api_key= to VoyageEmbedder."
            )
        self._client = voyageai.Client(api_key=key, max_retries=max_retries, timeout=timeout)
        self._model = model
        self.total_tokens_used = 0

    def embed_documents(self, texts: Sequence[str]) -> list[list[float] | None]:
        """`input_type='document'` is mandatory for voyage-3-large asymmetric
        retrieval quality. Returns parallel list; None for empty-string inputs."""
        return self._embed_with_input_type(list(texts), input_type="document")

    def embed_query(self, text: str) -> list[float]:
        """`input_type='query'` is mandatory; the model treats query and
        document embeddings asymmetrically."""
        if not text or not text.strip():
            raise ValueError("Cannot embed empty query")
        result = self._embed_batch([text], input_type="query")
        return result.embeddings[0]

    def _embed_with_input_type(
        self, texts: list[str], input_type: str
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
            embeddings = self._embed_with_split(batch_texts, input_type=input_type)
            for idx, vec in zip(batch_indices, embeddings, strict=True):
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
                len(cur_txt) >= MAX_INPUTS_PER_BATCH or cur_chars + t_chars > MAX_CHARS_PER_BATCH
            ):
                batches.append((cur_idx, cur_txt))
                cur_idx, cur_txt, cur_chars = [], [], 0
            cur_idx.append(i)
            cur_txt.append(t)
            cur_chars += t_chars
        if cur_txt:
            batches.append((cur_idx, cur_txt))
        return batches

    def _embed_with_split(self, texts: list[str], input_type: str) -> list[list[float]]:
        try:
            result = self._embed_batch(texts, input_type=input_type)
            return result.embeddings
        except (ve.InvalidRequestError, ve.MalformedRequestError) as e:
            if len(texts) <= 1:
                logger.error("Voyage rejected single-item embed: %s", e)
                raise
            mid = len(texts) // 2
            logger.warning("Voyage rejected batch of %d, splitting and retrying", len(texts))
            left = self._embed_with_split(texts[:mid], input_type=input_type)
            right = self._embed_with_split(texts[mid:], input_type=input_type)
            return left + right

    def _embed_batch(self, texts: list[str], input_type: str) -> EmbedResult:
        response = self._client.embed(
            texts=texts,
            model=self._model,
            input_type=input_type,
            truncation=True,
        )
        self.total_tokens_used += response.total_tokens
        return EmbedResult(embeddings=response.embeddings, total_tokens=response.total_tokens)

    def count_tokens(self, texts: Sequence[str]) -> int:
        return self._client.count_tokens(texts=list(texts), model=self._model)
