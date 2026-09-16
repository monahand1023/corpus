"""Ingestion orchestrator: connector → chunker → embed → upsert.

Idempotent. Content-hash dedup skips unchanged chunks. Orphan deletion
removes chunks for source docs that disappeared since the last run.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

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


def _reported_int_attr(obj: object, name: str) -> int:
    """Shared coercion for the optional `failed_files` / `skipped_files`
    connector counters: only a non-negative int counts as a genuine report.
    Test doubles and mocks expose arbitrary attributes, so a bare truthiness
    or comparison check would raise or silently mis-gate."""
    raw = getattr(obj, name, 0)
    if isinstance(raw, bool) or not isinstance(raw, int) or raw < 0:
        return 0
    return raw


def _reported_failures(connector: object) -> int:
    """Read a connector's optional per-file failure count.

    `failed_files` is an OPTIONAL connector capability, deliberately not part of
    the `Connector` protocol: connectors living outside this repository (which
    register themselves into CONNECTOR_REGISTRY at runtime) predate it and must
    keep working untouched. An absent attribute therefore means "does not report
    failures" rather than "had zero failures" — such a connector is not
    protected by the pruning gate, which is the price of not breaking it.

    `failed_files` means MIGHT SUCCEED NEXT TIME: a file that was momentarily
    locked, a transient read error. It suppresses pruning for the whole source
    (see the PER-FILE COMPLETENESS comment in `ingest()`). Contrast with the
    separate `skipped_files` counter (`_reported_skips` below), which means a
    file the connector has decided it will NEVER be able to read — that does
    NOT suppress pruning, because the file's absence from the index is not a
    surprise to investigate; it's the connector's own settled judgment.
    """
    return _reported_int_attr(connector, "failed_files")


def _reported_skips(connector: object) -> int:
    """Read a connector's optional `skipped_files` counter: files the
    connector has permanently given up on, as opposed to `failed_files`'
    "might work next time" (see `_reported_failures`).

    Concrete case this exists for: a source directory containing thousands of
    files in a format the connector deliberately does not support (e.g. a raw
    image format alongside the JPEGs it does handle). Every one of those files
    is unreadable on every run, forever — counting them as `failed_files`
    would suppress pruning for that source permanently, since the gate never
    sees the zero it's waiting for. `skipped_files` is exactly as optional and
    defensively read as `failed_files`: an absent attribute means "does not
    report skips," not "skipped nothing," and it never suppresses pruning by
    itself. It IS reported on `IngestResult` and logged, so a large skip count
    stays visible instead of disappearing silently.
    """
    return _reported_int_attr(connector, "skipped_files")


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


def _resolved_path_or_none(source_cfg: Any) -> str | None:
    """A source's resolved path, or None if it has none to give.

    A source config is not required to expose `resolved_path` — a
    third-party or synthetic config need not — and a path that cannot be
    resolved is not worth failing an otherwise-complete ingest over.
    """
    try:
        return str(source_cfg.resolved_path())
    except (OSError, AttributeError):
        return None


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
    # False when the embedder reports no usage, so `tokens_used` is an
    # absence rather than a zero. See Embedder.counts_tokens.
    tokens_counted: bool = True
    files_failed: int = 0
    pruning_performed: bool = True
    files_skipped: int = 0
    # True when delete_orphans refused to prune because the orphan count
    # exceeded the blast-radius guard (OrphanPruneRefused) -- distinct from
    # pruning_performed=False due to failed_files, which is a DIFFERENT
    # reason pruning didn't run. prune_refused_detail carries the readable
    # numbers for the CLI/caller to surface; it's the exception's own
    # message, so the reasoning is written in exactly one place.
    prune_refused: bool = False
    prune_refused_detail: str | None = None
    # Set when this run yielded materially fewer documents than the previous
    # one for the same source. Advisory, never blocking -- the destructive
    # case is already covered by the orphan-prune guard. This catches the
    # NON-destructive case that guard cannot see: a connector that under-
    # yields while pruning happens to be suppressed deletes nothing, reports
    # no failures, and looks like a completely normal run.
    yield_drop_detail: str | None = None
    # Set when this source name was last ingested from a DIFFERENT path.
    # Either a folder moved (harmless) or two different folders normalized to
    # the same source name (destructive: chunk ids collide and the second
    # ingest overwrites the first, with nothing pruned for any guard to see).
    path_change_detail: str | None = None
    # Names the DOCUMENTS a warning is about -- set only when one of the
    # warnings above already fired, because identifying them costs a second
    # scan of the source's chunks and there is no reason to pay it on a run
    # that looks healthy.
    vanished_detail: str | None = None
    # Set when this run permanently skipped more inputs than the last one AND
    # pruning deleted chunks. A skip does not suppress pruning — it means
    # "never readable" — so files reclassified as unreadable have their
    # indexed content deleted as if they had been removed from disk.
    skip_rise_detail: str | None = None


class Ingester:
    def __init__(
        self,
        config: CorpusConfig,
        store: ChunkStore | None = None,
        embedder: Embedder | None = None,
    ):
        self._config = config
        self._store = store or ChunkStore(
            config.db_path,
            embedding_dim=config.embedder.dim,
            cache_size_mb=config.performance.cache_size_mb,
            mmap_size_mb=config.performance.mmap_size_mb,
            temp_store_memory=config.performance.temp_store_memory,
        )
        self._embedder = embedder or make_embedder(
            provider=config.embedder.provider,
            model=config.embedder.model,
            dim=config.embedder.dim,
        )
        self._owned_store = store is None

    def _check_skip_rise(self, source_name: str, skipped: int, orphans: int) -> str | None:
        """Warn when MORE inputs became permanently unreadable and chunks died.

        `skipped_files` means "this connector will never read these", which is
        why — unlike `failed_files` — it does not suppress pruning. That is
        right for a format the connector has never supported. It is wrong for
        a file it read successfully last week.

        A rise in the skip count is exactly that second case: a parser
        regression, a permission change, a dependency that stopped loading, or
        a new "unsupported" rule shipped in the engine itself. Those files then
        yield no document, their chunks become orphans, and pruning deletes
        content that is still sitting on disk and was readable a run ago.

        Only reported when chunks were actually deleted, because a rise with
        nothing pruned has cost nothing yet. This does not block the prune —
        the blast-radius guard already covers the catastrophic version, and
        this is for the handful-of-files case that slips under it.
        """
        if not orphans or not skipped:
            return None
        previous = self._store.last_skipped(source_name)
        if previous is None or skipped <= previous:
            return None
        return (
            f"permanently skipped {skipped:,} input(s), up from {previous:,} "
            f"last run, and pruned {orphans:,} chunk(s). Files the connector "
            f"used to read may have been reclassified as unreadable and had "
            f"their indexed content deleted while still present on disk. "
            f"Check what changed: a parser, a permission, an optional "
            f"dependency, or a new skip rule in the engine."
        )

    def _check_source_path(self, source_name: str, source_cfg: Any) -> str | None:
        """Warn when a source name is reused for a different path.

        Chunk ids derive from (source_type, source_key, kind, index). Two
        folders that normalize to the same source name and contain a file of
        the same name therefore produce identical chunk ids, and the second
        ingest OVERWRITES the first's content. Nothing is pruned, so neither
        the blast-radius guard nor the yield-drop check ever fires — both runs
        report a document each and look entirely healthy.

        `corpus.util.autodetect.normalize_source_name` keeps distinct folder
        names apart where it can, but it cannot help when two folders in
        different places genuinely share a name (`~/a/Notes` and `~/b/Notes`),
        which is the common case. This is the guard for that.

        A warning rather than a refusal: moving a folder and re-ingesting it
        is legitimate and produces the identical signal, and refusing it would
        train the operator to reach for an override reflexively.
        """
        current = _resolved_path_or_none(source_cfg)
        if current is None:
            return None
        previous = self._store.last_source_path(source_name)
        if previous is None or previous == current:
            return None
        detail = (
            f"last ingested from {previous!r}, now reading {current!r}. If the "
            f"folder moved, nothing to do. If these are two DIFFERENT folders "
            f"that share a source name, this run has overwritten the other's "
            f"chunks — give one of them a distinct [[sources]] name."
        )
        logger.warning("  %s: %s", source_name, detail)
        return detail

    def _check_yield_drop(self, source_name: str, documents: int, chunks: int) -> str | None:
        """Warn when a source yields far fewer documents than it did last time.

        The orphan-prune guard already refuses a destructive sweep that looks
        too large. It cannot see this case: when pruning is suppressed (a
        connector reported unreadable files) a collapse in yield deletes
        nothing, reports no failure, and produces output indistinguishable
        from a healthy run. The index quietly stops matching reality until
        someone happens to notice a search returning less than it used to.

        Advisory only. A genuine bulk deletion -- a folder emptied on
        purpose -- is a normal thing to do, and this must not stand in its
        way. It exists so the operator is TOLD, not stopped.

        Reuses `pruning.max_orphan_ratio` and `min_chunks_for_guard` rather
        than introducing a second pair of knobs: "how big a drop is
        suspicious" is the same judgment in both places, and two settings
        that must agree are a way to get them disagreeing.
        """
        previous = self._store.last_yield(source_name)
        if previous is None:
            return None
        previous_documents, previous_chunks = previous
        max_drop_ratio, min_for_guard = _pruning_settings(self._config)

        # Both quantities, because they fail differently. Documents catch a
        # connector that stops finding files. CHUNKS catch one that still
        # yields every document but extracts far less from each — a parser
        # regression, an encoding fallback that returns empty, a container
        # format that silently stopped opening. That second case leaves the
        # document count identical and would otherwise pass unremarked while
        # overwriting good content with thin content.
        worst: tuple[float, str, int, int] | None = None
        for label, now, before in (
            ("documents", documents, previous_documents),
            ("chunks", chunks, previous_chunks),
        ):
            if before < min_for_guard or now >= before:
                continue
            ratio = (before - now) / before
            if ratio <= max_drop_ratio:
                continue
            if worst is None or ratio > worst[0]:
                worst = (ratio, label, now, before)
        if worst is None:
            return None
        ratio, label, now, before = worst
        detail = (
            f"yielded {now:,} {label}, down from {before:,} last run "
            f"({ratio:.0%} fewer). If that is expected (content genuinely "
            f"removed), re-run with --prune-anyway to accept it as the new "
            f"baseline. If not, the connector may be under-enumerating or "
            f"under-extracting -- check for an unreachable path, a parser "
            f"that started failing, or a skip that silently swallows files."
        )
        logger.warning("  %s: %s", source_name, detail)
        return detail

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
        # independent of what the connector reported. `skipped_files` (read
        # separately, see `_reported_skips`) is a THIRD, distinct signal: files
        # the connector has permanently given up on. It does not suppress
        # pruning at all -- their absence from the index is expected, not
        # evidence of a bug -- but it IS reported and logged so a large skip
        # count stays visible.
        failed_files = _reported_failures(connector)
        skipped_files = _reported_skips(connector)
        if skipped_files:
            logger.info(
                "  %s: %d file(s) permanently unreadable by design (unsupported "
                "format, etc.) -- not counted toward the pruning gate",
                source_name,
                skipped_files,
            )
        path_change_detail = self._check_source_path(source_name, source_cfg)
        yield_drop_detail = self._check_yield_drop(source_name, documents, chunks_seen)
        prune = failed_files == 0 or prune_anyway

        orphans = 0
        vanished_sample: list[str] = []
        vanished_total = 0
        prune_refused = False
        prune_refused_detail: str | None = None
        if prune:
            max_orphan_ratio, min_chunks_for_guard = _pruning_settings(self._config)
            try:
                sweep = self._store.sweep_orphans(
                    source_name,
                    seen_ids,
                    max_orphan_ratio=max_orphan_ratio,
                    min_chunks_for_guard=min_chunks_for_guard,
                    force=prune_anyway,
                )
                orphans = sweep.deleted
                vanished_sample = list(sweep.vanished)
                vanished_total = sweep.vanished_total
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

        # Name the documents that stopped being found -- ALWAYS, not only when
        # a threshold tripped.
        #
        # This is free: the accounting rides along on the scan `sweep_orphans`
        # already performs (measured on a very large source, adding
        # `source_key` to that SELECT cost nothing). Being free is what makes
        # it unconditional, and unconditional is what closes the two holes the
        # threshold guards structurally cannot see:
        #
        #   * a drop UNDER the yield-drop ratio -- 19% of a source vanishing
        #     said nothing at all
        #   * substitution -- N documents replaced by N others, every count
        #     identical while the content silently turns over
        #
        # The one path with no sweep to ride on is a suppressed prune (a
        # connector reported unreadable files), so that case pays for its own
        # scan. It has already gone wrong by then, which is exactly when the
        # extra ~900 ms is worth spending.
        if vanished_total == 0 and (prune_refused or not prune):
            vanished_sample, vanished_total = self._store.vanished_documents(
                source_name, seen_ids
            )

        vanished_detail: str | None = None
        if vanished_total:
            shown = ", ".join(vanished_sample)
            more = (
                f" (+{vanished_total - len(vanished_sample):,} more)"
                if vanished_total > len(vanished_sample)
                else ""
            )
            vanished_detail = (
                f"{vanished_total:,} document(s) indexed previously were not found "
                f"this run: {shown}{more}"
            )
            logger.warning("  %s: %s", source_name, vanished_detail)

        skip_rise_detail = self._check_skip_rise(source_name, skipped_files, orphans)

        # THE BASELINE ONLY ADVANCES ON A TRUSTED RUN.
        #
        # Recording unconditionally turns a single bad run into a ratchet: the
        # collapsed yield becomes the new normal, so the next equally bad run
        # compares favourably and says nothing. Repeated losses just under the
        # guard's ratio then walk a source down to nothing, each step looking
        # healthy, until it falls under `min_chunks_for_guard` and one run can
        # delete the remainder.
        #
        # So a run that showed ANY anomaly — unreadable files, a refused
        # prune, a yield drop, a path change — leaves the previous baseline
        # in place and keeps comparing against it. `--prune-anyway` is the
        # acknowledgement gesture: it means the operator looked, so it
        # advances the baseline even when the run was anomalous.
        anomalous = bool(
            failed_files
            or prune_refused
            or yield_drop_detail
            or path_change_detail
            or skip_rise_detail
        )
        if prune_anyway or not anomalous:
            self._store.record_yield(
                source_name,
                documents,
                chunks_seen,
                source_path=_resolved_path_or_none(source_cfg),
                skipped=skipped_files,
            )

        return IngestResult(
            source_name=source_name,
            documents=documents,
            chunks_seen=chunks_seen,
            chunks_upserted=chunks_upserted,
            chunks_skipped=chunks_skipped,
            orphans_deleted=orphans,
            tokens_used=self._embedder.total_tokens_used - tokens_before,
            tokens_counted=getattr(self._embedder, "counts_tokens", True),
            elapsed_seconds=time.monotonic() - start,
            files_failed=failed_files,
            pruning_performed=prune,
            yield_drop_detail=yield_drop_detail,
            vanished_detail=vanished_detail,
            path_change_detail=path_change_detail,
            skip_rise_detail=skip_rise_detail,
            files_skipped=skipped_files,
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
