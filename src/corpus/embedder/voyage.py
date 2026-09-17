"""Voyage embeddings — batched, retried, asymmetric document/query input types.

The Voyage SDK's `max_retries` handles transient failures via tenacity. We
add: empty-string filtering (Voyage 400s), greedy batch packing under their
per-request limits, and recursive split on `InvalidRequestError` (oversize
batch) until it fits.
"""

from __future__ import annotations

import logging
import os
import time
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

import voyageai
import voyageai.error as ve

logger = logging.getLogger(__name__)

# Voyage limits as of 2026 (verify on docs.voyageai.com — they've changed mid-0.x):
#   - up to 128 inputs per /embed request
#   - a HARD 120,000 tokens per /embed request. Exceeding it returns
#     InvalidRequestError: "The max allowed tokens per submitted batch is 120000."
#
# Pack by REAL tokens, not by characters. Token density is not a constant: on a
# mixed personal/work corpus it ranged from 3.66 chars/token (plain prose) to
# 1.51 (dense or structured text) — a 2.4x spread. A char budget sized for prose
# silently becomes a 2x overrun on dense content, and every oversized batch is
# rejected and recursively halved, which is far slower than embedding itself.
# (A previous 400,000-char budget, annotated "~100K tokens", is 264K tokens at
# 1.51 chars/token — more than double the cap.)
MAX_INPUTS_PER_BATCH = 128
MAX_TOKENS_PER_BATCH = 100_000

# Fallback when the local tokenizer is unavailable. Deliberately pessimistic:
# below the densest ratio observed (1.51) so it can only over-split, never
# overflow. Over-splitting costs a few extra requests; overflowing costs a
# rejection plus a recursive-halving cascade.
FALLBACK_CHARS_PER_TOKEN = 1.3

# Voyage projects have a per-minute token limit that varies by model and plan
# (3,000,000 TPM was observed for voyage-3-large; confirm yours for other
# models). Firing requests blindly on a large ingest blows past it and
# degrades into a wall of 429 retries that can stall the run. We proactively pace
# under a target below the hard limit, sleeping when the rolling 60s window is full.
# Override via CORPUS_TPM_TARGET (raise it if your project's limit is higher).
TPM_TARGET = int(os.environ.get("CORPUS_TPM_TARGET") or 2_600_000)
_TPM_WINDOW_SECONDS = 60.0


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
        self.counts_tokens = True
        # None until first use; False if the tokenizer is unavailable here.
        self._tokenizer_ok: bool | None = None
        # Rolling window of (monotonic_ts, tokens) for proactive TPM throttling.
        self._token_window: deque[tuple[float, int]] = deque()

    def _throttle(self, est_tokens: int) -> None:
        """Sleep if sending `est_tokens` now would exceed the rolling 60s TPM target.

        Keeps ingestion smoothly under Voyage's per-minute token limit instead of
        firing blindly and thrashing on 429 rate-limit retries."""
        now = time.monotonic()
        while self._token_window and now - self._token_window[0][0] > _TPM_WINDOW_SECONDS:
            self._token_window.popleft()
        used = sum(t for _, t in self._token_window)
        if used + est_tokens > TPM_TARGET and self._token_window:
            sleep_for = _TPM_WINDOW_SECONDS - (now - self._token_window[0][0]) + 0.5
            if sleep_for > 0:
                logger.info(
                    "TPM throttle: %d tokens in window + %d est > %d target; sleeping %.1fs",
                    used, est_tokens, TPM_TARGET, sleep_for,
                )
                time.sleep(sleep_for)

    def embed_documents(self, texts: Sequence[str]) -> list[list[float] | None]:
        """`input_type='document'` is mandatory for Voyage's asymmetric
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

    def token_counts(self, texts: list[str]) -> list[int]:
        """Per-text token counts from Voyage's LOCAL tokenizer (no API cost).

        Falls back to a pessimistic char estimate if the tokenizer cannot be
        reached (offline, older SDK), so packing degrades rather than breaks.
        """
        if self._tokenizer_ok is not False:
            try:
                counts = [len(e) for e in self._client.tokenize(texts, model=self._model)]
                # A tokenizer that returns the wrong arity is not usable. Fall
                # back rather than propagate a length mismatch into packing.
                if len(counts) != len(texts):
                    raise ValueError(
                        f"tokenizer returned {len(counts)} counts for {len(texts)} texts"
                    )
                self._tokenizer_ok = True
                return counts
            except Exception as e:
                if self._tokenizer_ok is None:
                    logger.warning(
                        "Voyage tokenizer unavailable (%s); packing batches with a "
                        "conservative char estimate instead.", e,
                    )
                self._tokenizer_ok = False
        return [max(1, int(len(t) / FALLBACK_CHARS_PER_TOKEN)) for t in texts]

    def _pack_batches(
        self, indices: list[int], texts: list[str]
    ) -> list[tuple[list[int], list[str]]]:
        """Group inputs under BOTH per-request caps: 128 inputs and 120K tokens.

        Packing on input count alone is not enough — 128 dense inputs can exceed
        the token cap on their own.
        """
        counts = self.token_counts(texts)
        batches: list[tuple[list[int], list[str]]] = []
        cur_idx: list[int] = []
        cur_txt: list[str] = []
        cur_tokens = 0
        for i, t, n_tok in zip(indices, texts, counts, strict=True):
            if cur_txt and (
                len(cur_txt) >= MAX_INPUTS_PER_BATCH
                or cur_tokens + n_tok > MAX_TOKENS_PER_BATCH
            ):
                batches.append((cur_idx, cur_txt))
                cur_idx, cur_txt, cur_tokens = [], [], 0
            cur_idx.append(i)
            cur_txt.append(t)
            cur_tokens += n_tok
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
        # Estimate tokens from chars (~3 chars/token, conservative for mixed-language
        # content) and pace under the TPM target before firing the request.
        est_tokens = sum(self.token_counts(texts))
        self._throttle(est_tokens)
        response = self._client.embed(
            texts=texts,
            model=self._model,
            input_type=input_type,
            truncation=True,
        )
        # TRUNCATION, MADE VISIBLE. `truncation=True` above means a text past
        # the model's context is embedded from its opening tokens and comes
        # back looking exactly like a complete one -- the chunk is then
        # indexed by its first part only, with no error anywhere.
        #
        # The signal is free: if the server billed materially fewer tokens
        # than Voyage's LOCAL tokenizer counted for the same texts, the
        # difference is text that was discarded. No context-limit constant to
        # hardcode, keep current, or get wrong for a model someone swaps in.
        #
        # Only when the local tokenizer is trusted -- it falls back to a char
        # estimate when unavailable, and comparing an estimate against a real
        # count would warn on every batch.
        if self._tokenizer_ok:
            sent = sum(self.token_counts(texts))
            # 2%: the local tokenizer and the server disagree slightly on
            # ordinary text, and a threshold at zero would fire constantly.
            if sent and response.total_tokens < sent * 0.98:
                logger.warning(
                    "Voyage billed %d tokens for %d text(s) but the local "
                    "tokenizer counted %d -- roughly %d tokens were TRUNCATED "
                    "away. Those chunks are indexed by their opening only. "
                    "Split them before the next ingest.",
                    response.total_tokens,
                    len(texts),
                    sent,
                    sent - response.total_tokens,
                )

        self.total_tokens_used += response.total_tokens
        self._token_window.append((time.monotonic(), response.total_tokens))
        # SDK types embeddings as list[list[float]] | list[list[int]]; the int
        # variant only appears for quantized output dtypes, which we never request.
        embeddings = cast("list[list[float]]", response.embeddings)
        return EmbedResult(embeddings=embeddings, total_tokens=response.total_tokens)

    def count_tokens(self, texts: Sequence[str]) -> int:
        return self._client.count_tokens(texts=list(texts), model=self._model)
