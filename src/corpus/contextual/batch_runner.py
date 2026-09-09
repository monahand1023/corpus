"""Submit, poll, and apply Anthropic Message Batches for Contextual Retrieval.

Flow per source type:
  1. RESUME: if the state file holds in-flight batches, await + apply them FIRST
     (using each batch's PERSISTED custom_id→chunk_ids mapping), so an interrupted
     run never re-pays for results already sitting on Anthropic's servers.
  2. Recompute chunks still missing context; build per-doc-window requests.
  3. Submit in sub-batches under the request-count AND 256 MB payload caps;
     persist each batch's id + mapping before awaiting it.
  4. Poll (with retry) until each batch ends, then apply its results.

Correctness key: the apply maps results to chunks via the mapping CAPTURED AT
BUILD TIME (build_window_mapping), NOT by re-deriving windows from the live DB.
Re-derivation breaks the moment any chunk in the corpus changes context state
(e.g. a partially-applied source), which silently maps contexts to wrong chunks.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from corpus.contextual.batch_builder import (
    DEFAULT_WINDOW_SIZE,
    build_batch_requests,
    build_window_mapping,
    parse_batch_result,
)
from corpus.contextual.batch_util import (  # noqa: F401 — re-exported for back-compat
    MAX_BATCH_BYTES,
    MAX_REQUESTS_PER_BATCH,
    pack_batches,
    retry_with_backoff,
)
from corpus.contextual.contextualizer import (
    DEFAULT_MIN_TOKENS,
    is_useful_context,
    should_contextualize,
)
from corpus.db.sqlite import ChunkStore, StoredChunk

logger = logging.getLogger(__name__)

POLL_INTERVAL_S = 30


@dataclass
class BatchResult:
    requests_submitted: int = 0
    chunks_contextualized: int = 0
    errors: int = 0
    # Chunks with no context that were deliberately left that way, so a run's
    # cost is explainable without re-deriving the policy.
    skipped_ineligible: int = 0


class BatchContextualizer:
    def __init__(
        self,
        store: ChunkStore,
        embedder: Any,
        client: Any | None = None,
        state_path: Path | None = None,
        window_size: int = DEFAULT_WINDOW_SIZE,
        min_tokens: int = DEFAULT_MIN_TOKENS,
    ):
        self._store = store
        self._embedder = embedder
        self._window = window_size
        self._min_tokens = min_tokens
        self._state_path = state_path
        if client is not None:
            self._client = client
        else:
            import os
            key = os.environ.get("ANTHROPIC_API_KEY")
            if not key:
                raise RuntimeError("ANTHROPIC_API_KEY missing.")
            from anthropic import Anthropic
            self._client = Anthropic(api_key=key)

    # ------------------------------------------------------------------ run

    def run(
        self,
        source_type: str,
        doc_body_fn: Callable[[str], str],
        dry_run: bool = False,
    ) -> BatchResult:
        result = BatchResult()

        # 1. Resume any in-flight batches recorded in state BEFORE building anything.
        state = self._load_state(source_type)
        if state and not dry_run:
            by_id = {c.id: c for c in self._store.chunks_missing_context(source_type)}
            for entry in state["batches"]:
                if entry.get("applied"):
                    continue
                logger.info("resuming in-flight batch %s", entry["batch_id"])
                self._await_batch(entry["batch_id"])
                result.chunks_contextualized += self._apply_by_mapping(
                    entry["batch_id"], entry["mapping"], by_id
                )
                entry["applied"] = True
                self._write_state(state)

        # 2. Build requests for whatever is still missing context.
        chunks = self._store.chunks_missing_context(source_type)
        # Spend policy, applied before anything is built or priced. A chunk
        # shorter than `min_tokens` is made WORSE by contextualization — the
        # blurb rivals the content in length, so the embedding ends up
        # describing the blurb. See `contextualizer.should_contextualize`.
        eligible = [
            c
            for c in chunks
            if should_contextualize(
                (c.metadata or {}).get("token_count"), self._min_tokens
            )
        ]
        result.skipped_ineligible = len(chunks) - len(eligible)
        chunks = eligible
        if not chunks:
            logger.info("no eligible chunks missing context for %s", source_type)
            return result
        keys = {c.source_key for c in chunks}
        doc_bodies = {k: doc_body_fn(k) for k in keys}
        requests = build_batch_requests(chunks, doc_bodies, window_size=self._window)
        full_mapping = build_window_mapping(chunks, window_size=self._window)
        result.requests_submitted = len(requests)
        if dry_run:
            logger.info("[dry-run] would submit %d requests for %d chunks across %d docs",
                        len(requests), len(chunks), len(keys))
            return result

        # 3. Submit + persist mapping + await + apply, sub-batch by sub-batch.
        if state is None:
            state = {"source_type": source_type, "batches": []}
        by_id = {c.id: c for c in chunks}
        for sub in pack_batches(requests):
            batch = self._client.messages.batches.create(requests=sub)
            sub_mapping = {r["custom_id"]: full_mapping[r["custom_id"]] for r in sub}
            # Persist id+mapping BEFORE awaiting, so an interruption here is recoverable.
            state["batches"].append({"batch_id": batch.id, "applied": False, "mapping": sub_mapping})
            self._write_state(state)
            logger.info("submitted batch %s (%d requests)", batch.id, len(sub))
            self._await_batch(batch.id)
            result.chunks_contextualized += self._apply_by_mapping(batch.id, sub_mapping, by_id)
            state["batches"][-1]["applied"] = True
            self._write_state(state)

        return result

    # ----------------------------------------------------- polling / results

    def _await_batch(self, batch_id: str) -> None:
        while True:
            b = retry_with_backoff(lambda: self._client.messages.batches.retrieve(batch_id),
                                   f"retrieve({batch_id})")
            if getattr(b, "processing_status", None) == "ended":
                return
            time.sleep(POLL_INTERVAL_S)

    def _apply_by_mapping(self, batch_id: str, mapping: dict[str, list[str]],
                          chunk_by_id: dict[str, StoredChunk]) -> int:
        """Apply a batch's results using the persisted custom_id→chunk_ids mapping.
        Only writes chunks that are still present in chunk_by_id (i.e. still missing
        context), so re-applying an already-applied batch is a harmless no-op."""
        applied = 0
        errors = 0
        skipped_useless = 0
        results = retry_with_backoff(lambda: list(self._client.messages.batches.results(batch_id)),
                              f"results({batch_id})")
        for row in results:
            try:
                res = row.result
                if getattr(res, "type", None) != "succeeded":
                    continue
                ids = mapping.get(row.custom_id)
                if not ids:
                    continue
                tool_input = None
                for block in res.message.content:
                    if getattr(block, "type", None) == "tool_use":
                        tool_input = block.input
                        break
                if not tool_input:
                    continue
                to_embed = []
                order = []
                for cid, ctx in parse_batch_result(tool_input, ids).items():
                    c = chunk_by_id.get(cid)
                    if c is None:
                        # Either already applied (a harmless re-apply) or the
                        # chunk no longer exists — its document was re-ingested
                        # between submit and apply, so its id changed. The
                        # second case wastes what was paid for this result and
                        # is worth seeing, but it is indistinguishable from the
                        # first without tracking ids across the gap, and both
                        # are correct to skip.
                        continue
                    if not is_useful_context(ctx):
                        # ACCEPTED COST: leaving this NULL means a later run
                        # re-submits and re-pays for the same chunk. Measured
                        # at 0.037% of one archive — a few hundred chunks, a
                        # few cents — and a chunk that is empty or redacted
                        # today may have content the next time its document is
                        # ingested, so retrying is not purely waste. Marking
                        # it done would need either a schema column or a
                        # sentinel that retrieval must then learn to ignore.
                        # Not worth either for the amount involved.
                        # The model correctly declining to situate an empty or
                        # fully-redacted chunk. Storing that answer would
                        # prepend a meaningless token to the embedding and
                        # mark the chunk done, so it would never be retried.
                        # Leaving it NULL costs nothing and keeps it eligible.
                        skipped_useless += 1
                        continue
                    to_embed.append(f"{ctx}\n\n{c.content}")
                    order.append((cid, ctx))
                if not to_embed:
                    continue
                embeddings = self._embedder.embed_documents(to_embed)
                for (cid, ctx), emb in zip(order, embeddings, strict=True):
                    if emb is not None and self._store.set_context(cid, ctx, emb):
                        applied += 1
            except Exception:
                errors += 1
                logger.exception("failed to apply result %s (skipping)",
                                 getattr(row, "custom_id", "?"))
        if errors:
            logger.warning("apply %s: %d result rows skipped due to errors", batch_id, errors)
        if skipped_useless:
            logger.info(
                "batch %s: %d context(s) were placeholders and were left unset",
                batch_id,
                skipped_useless,
            )
        return applied

    # -------------------------------------------------------------- state io

    def _load_state(self, source_type: str) -> dict[str, Any] | None:
        if self._state_path is None or not self._state_path.exists():
            return None
        try:
            state = json.loads(self._state_path.read_text())
        except (ValueError, OSError):
            return None
        if not isinstance(state, dict):
            return None
        # Only the new mapping-bearing format is resumable; legacy {batch_id} state
        # (no per-window mapping) can't be safely re-applied, so we ignore it.
        if state.get("source_type") != source_type or "batches" not in state:
            return None
        return state

    def _write_state(self, state: dict[str, Any]) -> None:
        if self._state_path is not None:
            self._state_path.write_text(json.dumps(state))
