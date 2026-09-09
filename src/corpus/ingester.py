"""Ingestion orchestrator: connector → chunker → embed → upsert.

Idempotent. Content-hash dedup skips unchanged chunks. Orphan deletion
removes chunks for source docs that disappeared since the last run.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from corpus.config import CorpusConfig, PruningConfig
from corpus.connectors.registry import build_pipeline
from corpus.db.sqlite import (
    DEFAULT_MAX_ORPHAN_RATIO,
    DEFAULT_MIN_CHUNKS_FOR_GUARD,
    ChunkStore,
    OrphanPruneRefused,
)
from corpus.embedder.base import Embedder
from corpus.embedder.factory import make_embedder
from corpus.types import Chunk

logger = logging.getLogger(__name__)

INGEST_BATCH = 256


def _reported_failures(connector: object) -> int:
    """Read a connector's optional per-file failure count.

    `failed_files` is an OPTIONAL connector capability, deliberately not part of
    the `Connector` protocol: connectors living outside this repository (which
    register themselves into CONNECTOR_REGISTRY at runtime) predate it and must
    keep working untouched. An absent attribute therefore means "does not report
    failures" rather than "had zero failures" — such a connector is not
    protected by the pruning gate, which is the price of not breaking it.

    Only a non-negative int counts as a report. Test doubles and mocks expose
    arbitrary attributes, so a bare truthiness or comparison check would raise
    or silently mis-gate.
    """
    raw = getattr(connector, "failed_files", 0)
    if isinstance(raw, bool) or not isinstance(raw, int) or raw < 0:
        return 0
    return raw


def _pruning_settings(config: object) -> tuple[float, int]:
    """Resolve `delete_orphans`'s guard thresholds from `config.pruning`.

    Falls back to the engine defaults unless `config.pruning` is a real
    `PruningConfig` -- tests routinely construct `Ingester` with a
    `MagicMock` config (see `_reported_failures` for the same defensive
    posture toward connectors), and a MagicMock's auto-vivified
    `.pruning.max_orphan_ratio` is another MagicMock, not a float. Falling
    back to the same defaults `delete_orphans` itself bakes in is exactly
    the right behavior anyway: a config double that never mentions pruning
    should behave like a corpus.toml that never mentions `[pruning]`.
    """
    pruning = getattr(config, "pruning", None)
    if isinstance(pruning, PruningConfig):
        return pruning.max_orphan_ratio, pruning.min_chunks_for_guard
    return DEFAULT_MAX_ORPHAN_RATIO, DEFAULT_MIN_CHUNKS_FOR_GUARD


@dataclass
class IngestResult:
    source_name: str
    documents: int
    chunks_seen: int
    chunks_upserted: int
    chunks_skipped: int
    orphans_deleted: int
    tokens_used: int
    elapsed_seconds: float
    files_failed: int = 0
    pruning_performed: bool = True
    # True when delete_orphans refused to prune because the orphan count
    # exceeded the blast-radius guard (OrphanPruneRefused) -- distinct from
    # pruning_performed=False due to failed_files, which is a DIFFERENT
    # reason pruning didn't run. prune_refused_detail carries the readable
    # numbers for the CLI/caller to surface; it's the exception's own
    # message, so the reasoning is written in exactly one place.
    prune_refused: bool = False
    prune_refused_detail: str | None = None


class Ingester:
    def __init__(
        self,
        config: CorpusConfig,
        store: ChunkStore | None = None,
        embedder: Embedder | None = None,
    ):
        self._config = config
        self._store = store or ChunkStore(
            config.db_path, embedding_dim=config.embedder.dim
        )
        self._embedder = embedder or make_embedder(
            provider=config.embedder.provider,
            model=config.embedder.model,
            dim=config.embedder.dim,
        )
        self._owned_store = store is None

    def ingest(self, source_name: str, prune_anyway: bool = False) -> IngestResult:
        source_cfg = self._config.source_by_name(source_name)
        if source_cfg is None:
            raise ValueError(
                f"No source named '{source_name}' in corpus.toml. "
                f"Configured sources: {[s.name for s in self._config.sources]}"
            )
        connector, chunker = build_pipeline(source_cfg)

        start = time.monotonic()
        tokens_before = self._embedder.total_tokens_used
        documents = 0
        chunks_seen = 0
        chunks_upserted = 0
        chunks_skipped = 0
        seen_ids: set[str] = set()

        buffer: list[Chunk] = []
        for doc in connector.load():
            documents += 1
            for ch in chunker.chunk(doc):
                seen_ids.add(ch.id)
                chunks_seen += 1
                buffer.append(ch)
            if len(buffer) >= INGEST_BATCH:
                u, s = self._flush(buffer)
                chunks_upserted += u
                chunks_skipped += s
                buffer = []
            if documents % 200 == 0:
                logger.info(
                    "  ingested %d docs / %d chunks (%d upserted, %d unchanged)",
                    documents,
                    chunks_seen,
                    chunks_upserted,
                    chunks_skipped,
                )
        if buffer:
            u, s = self._flush(buffer)
            chunks_upserted += u
            chunks_skipped += s

        # ENUMERATION COMPLETENESS: reaching this line means connector.load()
        # ran to exhaustion, so seen_ids is the complete set for this source and
        # anything absent is genuinely gone. A connector that cannot enumerate
        # its source completely MUST raise (FileNotFoundError for an
        # unreachable root or volume) rather than yield a short list — a partial
        # enumeration here would delete every chunk it failed to yield.
        #
        # PER-FILE COMPLETENESS: a connector may also skip INDIVIDUAL files it
        # could not read, without raising. Those files yield no document, so
        # their chunk ids are absent from seen_ids and delete_orphans would
        # delete them — losing indexed content because a file was momentarily
        # locked. A connector reports such skips via an optional `failed_files`
        # counter; when it is non-zero we skip pruning entirely rather than
        # guess which absences are real. This also covers a file that was read
        # only PARTIALLY (e.g. a PDF page that failed to extract): the shorter
        # body produces fewer chunks, so the tail chunk ids vanish from seen_ids
        # and would be pruned even though the document itself was yielded.
        #
        # `failed_files` is deliberately NOT the only rail: it protects against
        # a connector that KNOWS it failed on specific files. It says nothing
        # about a connector that silently under-enumerates while reporting
        # zero failures -- e.g. one that skips files it believes are unchanged
        # and, on a later run, matches nothing at all. That bug produces
        # `failed_files == 0` and a `seen_ids` that is quietly wrong, which is
        # exactly what `delete_orphans`'s blast-radius guard below exists to
        # catch: it refuses when the fraction of a source's existing chunks
        # about to be pruned is implausibly large for a routine re-ingest,
        # independent of what the connector reported.
        failed_files = _reported_failures(connector)
        prune = failed_files == 0 or prune_anyway

        orphans = 0
        prune_refused = False
        prune_refused_detail: str | None = None
        if prune:
            max_orphan_ratio, min_chunks_for_guard = _pruning_settings(self._config)
            try:
                orphans = self._store.delete_orphans(
                    source_name,
                    seen_ids,
                    max_orphan_ratio=max_orphan_ratio,
                    min_chunks_for_guard=min_chunks_for_guard,
                    force=prune_anyway,
                )
            except OrphanPruneRefused as e:
                prune = False
                prune_refused = True
                prune_refused_detail = str(e)
        else:
            logger.warning(
                "  %s: %d file(s) could not be fully read; pruning skipped "
                "(re-run this source with --prune-anyway to prune regardless)",
                source_name,
                failed_files,
            )

        return IngestResult(
            source_name=source_name,
            documents=documents,
            chunks_seen=chunks_seen,
            chunks_upserted=chunks_upserted,
            chunks_skipped=chunks_skipped,
            orphans_deleted=orphans,
            tokens_used=self._embedder.total_tokens_used - tokens_before,
            elapsed_seconds=time.monotonic() - start,
            files_failed=failed_files,
            pruning_performed=prune,
            prune_refused=prune_refused,
            prune_refused_detail=prune_refused_detail,
        )

    def _flush(self, chunks: list[Chunk]) -> tuple[int, int]:
        known = self._store.get_known_hashes([c.id for c in chunks])
        to_embed = [c for c in chunks if known.get(c.id) != c.content_hash]
        already = len(chunks) - len(to_embed)
        if not to_embed:
            return 0, already

        texts = [c.content for c in to_embed]
        embeddings = self._embedder.embed_documents(texts)

        pairs: list[tuple[Chunk, list[float]]] = []
        for chunk, emb in zip(to_embed, embeddings, strict=True):
            if emb is None:
                continue
            pairs.append((chunk, emb))
        result = self._store.upsert_batch(pairs)
        return result.upserted, result.skipped + already

    def close(self) -> None:
        if self._owned_store:
            self._store.close()
