"""Unit tests for corpus.ingester.Ingester.

Tests use injected store/embedder to avoid touching real connectors or
embedding APIs. build_pipeline is patched at its import site in ingester.py.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from corpus.config import CorpusConfig
from corpus.connectors.markdown import MarkdownChunker
from corpus.connectors.registry import CONNECTOR_REGISTRY
from corpus.db.sqlite import ChunkStore
from corpus.ingester import Ingester, IngestResult
from corpus.types import Chunk, ChunkKind, ChunkMetadata, SourceDocument
from corpus.util.hash import chunk_id, sha256

DIM = 1024


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_chunk(source_type: str, source_key: str, idx: int, content: str) -> Chunk:
    cid = chunk_id(source_type, source_key, ChunkKind.HEADER, idx)
    return Chunk(
        id=cid,
        content=content,
        content_hash=sha256(content),
        metadata=ChunkMetadata(
            source_type=source_type,
            source_key=source_key,
            chunk_kind=ChunkKind.HEADER,
            chunk_index=idx,
            title=f"{source_type}:{source_key}",
        ),
    )


def fake_embedder(token_start: int = 0) -> MagicMock:
    """Return a mock Embedder whose embed_documents returns unit vectors."""
    emb = MagicMock()
    emb.total_tokens_used = token_start
    emb.embed_documents = lambda texts: [[0.0] * DIM for _ in texts]
    return emb


def make_config(source_name: str = "notes", return_source: object = True) -> MagicMock:
    cfg = MagicMock()
    cfg.source_by_name.side_effect = (
        lambda name: (object() if return_source else None)
        if name == source_name
        else None
    )
    cfg.sources = []
    return cfg


def make_ingester(
    tmp_path: Path,
    source_name: str = "notes",
    embedder: MagicMock | None = None,
    return_source: bool = True,
) -> tuple[Ingester, ChunkStore, MagicMock]:
    """Build an Ingester backed by a real on-disk ChunkStore and a mock embedder."""
    store = ChunkStore(tmp_path / "test.db", embedding_dim=DIM)
    emb = embedder or fake_embedder()
    config = make_config(source_name=source_name, return_source=return_source)
    ingester = Ingester(config=config, store=store, embedder=emb)
    return ingester, store, config


def make_ingester_with_config(
    tmp_path: Path,
    sources: list[dict[str, object]],
    embedder: MagicMock | None = None,
) -> tuple[Ingester, ChunkStore]:
    """Build an Ingester backed by a REAL CorpusConfig (not the MagicMock config
    of make_ingester() above), so build_pipeline() actually resolves a connector
    through CONNECTOR_REGISTRY by the source's configured `type`. Needed for tests
    that register a fake connector in CONNECTOR_REGISTRY rather than patching
    corpus.ingester.build_pipeline directly."""
    config = CorpusConfig.model_validate({"db_path": tmp_path / "config_test.db", "sources": sources})
    store = ChunkStore(tmp_path / "test.db", embedding_dim=DIM)
    emb = embedder or fake_embedder()
    ingester = Ingester(config=config, store=store, embedder=emb)
    return ingester, store


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_store(tmp_path: Path) -> ChunkStore:
    s = ChunkStore(tmp_path / "ingester_test.db", embedding_dim=DIM)
    yield s
    s.close()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_ingest_unknown_source_raises(tmp_path: Path) -> None:
    """ingest() with an unknown source name must raise ValueError immediately."""
    ingester, store, _ = make_ingester(tmp_path, return_source=False)
    try:
        with pytest.raises(ValueError, match="nonexistent"):
            ingester.ingest("nonexistent")
    finally:
        store.close()


def test_ingest_empty_source_returns_zero_counts(tmp_path: Path) -> None:
    """When the connector yields no documents all counts are zero."""
    ingester, store, _ = make_ingester(tmp_path)
    mock_connector = MagicMock()
    mock_connector.load.return_value = iter([])
    mock_chunker = MagicMock()

    try:
        with patch("corpus.ingester.build_pipeline", return_value=(mock_connector, mock_chunker)):
            result = ingester.ingest("notes")

        assert result.documents == 0
        assert result.chunks_seen == 0
        assert result.chunks_upserted == 0
        assert result.chunks_skipped == 0
        assert result.orphans_deleted == 0
    finally:
        store.close()


def test_ingest_counts_documents_and_chunks(tmp_path: Path) -> None:
    """Two documents with two chunks each → documents=2, chunks_seen=4."""
    ingester, store, _ = make_ingester(tmp_path)

    docs = [MagicMock(), MagicMock()]
    chunks_per_doc = [
        [make_chunk("notes", "doc-a", 0, "alpha"), make_chunk("notes", "doc-a", 1, "beta")],
        [make_chunk("notes", "doc-b", 0, "gamma"), make_chunk("notes", "doc-b", 1, "delta")],
    ]

    mock_connector = MagicMock()
    mock_connector.load.return_value = iter(docs)
    mock_chunker = MagicMock()
    mock_chunker.chunk.side_effect = chunks_per_doc

    try:
        with patch("corpus.ingester.build_pipeline", return_value=(mock_connector, mock_chunker)):
            result = ingester.ingest("notes")

        assert result.documents == 2
        assert result.chunks_seen == 4
    finally:
        store.close()


def test_ingest_all_new_chunks_get_upserted(tmp_path: Path) -> None:
    """With a fresh store, every chunk should be upserted and none skipped."""
    ingester, store, _ = make_ingester(tmp_path)

    chunks = [
        make_chunk("notes", "doc-x", 0, "content one"),
        make_chunk("notes", "doc-x", 1, "content two"),
        make_chunk("notes", "doc-x", 2, "content three"),
    ]
    doc = MagicMock()

    mock_connector = MagicMock()
    mock_connector.load.return_value = iter([doc])
    mock_chunker = MagicMock()
    mock_chunker.chunk.return_value = chunks

    try:
        with patch("corpus.ingester.build_pipeline", return_value=(mock_connector, mock_chunker)):
            result = ingester.ingest("notes")

        assert result.chunks_upserted == 3
        assert result.chunks_skipped == 0
    finally:
        store.close()


def test_ingest_dedup_skips_unchanged_content(tmp_path: Path) -> None:
    """Running ingest twice with the same chunks: second run skips all (dedup)."""
    ingester, store, _ = make_ingester(tmp_path)

    chunks = [
        make_chunk("notes", "doc-y", 0, "same content a"),
        make_chunk("notes", "doc-y", 1, "same content b"),
    ]
    doc = MagicMock()

    mock_connector = MagicMock()
    mock_connector.load.return_value = iter([doc])
    mock_chunker = MagicMock()
    mock_chunker.chunk.return_value = chunks

    try:
        with patch("corpus.ingester.build_pipeline", return_value=(mock_connector, mock_chunker)):
            first = ingester.ingest("notes")

        assert first.chunks_upserted == 2
        assert first.chunks_skipped == 0

        # Reset side-effects for second run
        mock_connector.load.return_value = iter([doc])
        mock_chunker.chunk.return_value = chunks

        with patch("corpus.ingester.build_pipeline", return_value=(mock_connector, mock_chunker)):
            second = ingester.ingest("notes")

        assert second.chunks_upserted == 0
        assert second.chunks_skipped == 2
    finally:
        store.close()


def test_ingest_orphan_deletion(tmp_path: Path) -> None:
    """Chunks from a doc that disappears on the second run are deleted as orphans."""
    ingester, store, _ = make_ingester(tmp_path)

    chunks_a = [make_chunk("notes", "doc-a", 0, "doc-a content")]
    chunks_b = [make_chunk("notes", "doc-b", 0, "doc-b content")]

    doc_a = MagicMock()
    doc_b = MagicMock()

    mock_connector = MagicMock()
    mock_chunker = MagicMock()

    try:
        # First run: both doc-a and doc-b
        mock_connector.load.return_value = iter([doc_a, doc_b])
        mock_chunker.chunk.side_effect = [chunks_a, chunks_b]

        with patch("corpus.ingester.build_pipeline", return_value=(mock_connector, mock_chunker)):
            first = ingester.ingest("notes")

        assert first.chunks_upserted == 2
        assert first.orphans_deleted == 0

        # Second run: only doc-a; doc-b's chunk becomes an orphan
        mock_connector.load.return_value = iter([doc_a])
        mock_chunker.chunk.side_effect = [chunks_a]

        with patch("corpus.ingester.build_pipeline", return_value=(mock_connector, mock_chunker)):
            second = ingester.ingest("notes")

        assert second.orphans_deleted == 1
    finally:
        store.close()


def test_flush_skips_none_embeddings(tmp_path: Path) -> None:
    """If the embedder returns None for some positions, those chunks are not upserted."""
    store = ChunkStore(tmp_path / "none_emb.db", embedding_dim=DIM)

    emb = MagicMock()
    emb.total_tokens_used = 0
    # Return [vector, None, vector] — the middle chunk should be skipped
    emb.embed_documents = lambda texts: [
        [0.0] * DIM if i != 1 else None
        for i in range(len(texts))
    ]

    config = make_config()
    ingester = Ingester(config=config, store=store, embedder=emb)

    chunks = [
        make_chunk("notes", "doc-z", 0, "first"),
        make_chunk("notes", "doc-z", 1, "second"),
        make_chunk("notes", "doc-z", 2, "third"),
    ]
    doc = MagicMock()

    mock_connector = MagicMock()
    mock_connector.load.return_value = iter([doc])
    mock_chunker = MagicMock()
    mock_chunker.chunk.return_value = chunks

    try:
        with patch("corpus.ingester.build_pipeline", return_value=(mock_connector, mock_chunker)):
            result = ingester.ingest("notes")

        # Only 2 chunks have real embeddings; the None one is dropped
        assert result.chunks_upserted == 2
    finally:
        store.close()


def test_ingest_result_fields_populated(tmp_path: Path) -> None:
    """IngestResult carries the correct source_name and a positive elapsed time."""
    ingester, store, _ = make_ingester(tmp_path)

    doc = MagicMock()
    chunk = make_chunk("notes", "doc-t", 0, "timing test content")

    mock_connector = MagicMock()
    mock_connector.load.return_value = iter([doc])
    mock_chunker = MagicMock()
    mock_chunker.chunk.return_value = [chunk]

    try:
        with patch("corpus.ingester.build_pipeline", return_value=(mock_connector, mock_chunker)):
            result = ingester.ingest("notes")

        assert result.source_name == "notes"
        assert result.elapsed_seconds > 0
        assert isinstance(result, IngestResult)
    finally:
        store.close()


def test_incomplete_enumeration_does_not_delete_orphans(tmp_path: Path) -> None:
    """An unmounted volume must never be read as 'every document was deleted'.

    Uses a REAL CorpusConfig + CONNECTOR_REGISTRY entry (rather than patching
    corpus.ingester.build_pipeline like the tests above) so this exercises the
    actual connector-resolution path a real unmounted-volume connector would
    go through.
    """
    calls = {"n": 0}

    class FlakyConnector:
        source_type = "flaky"

        def load(self):
            calls["n"] += 1
            if calls["n"] == 1:
                yield SourceDocument(
                    source_type="flaky", source_key="a.txt", title="a",
                    raw={"body": "hello world", "path": "a.txt"},
                )
            else:
                raise FileNotFoundError("volume not mounted")

    CONNECTOR_REGISTRY["flaky"] = lambda cfg: (
        FlakyConnector(),
        MarkdownChunker(source_type="flaky"),
    )
    try:
        ingester, store = make_ingester_with_config(
            tmp_path, sources=[{"name": "flaky", "type": "flaky", "path": str(tmp_path)}]
        )
        try:
            first = ingester.ingest("flaky")
            assert first.chunks_upserted > 0

            def chunk_count() -> int:
                row = store._conn.execute(
                    "SELECT COUNT(*) AS c FROM chunks WHERE source_type = 'flaky'"
                ).fetchone()
                return int(row["c"])

            before = chunk_count()
            assert before > 0

            with pytest.raises(FileNotFoundError):
                ingester.ingest("flaky")

            # The chunks from run 1 must survive the failed run, NOT be swept as orphans.
            assert chunk_count() == before
        finally:
            store.close()
    finally:
        CONNECTOR_REGISTRY.pop("flaky", None)


# ---------------------------------------------------------------------------
# Per-file failure counting gates the prune
#
# delete_orphans() removes every stored chunk whose id is absent from seen_ids,
# scoped by source_type alone. So a file the connector could not read yields no
# document, and its existing chunks are deleted. These tests pin that a reported
# read failure suppresses pruning entirely.
# ---------------------------------------------------------------------------

class _CountingConnector:
    """Connector that yields some docs and reports N unreadable files."""

    def __init__(self, source_type: str, docs: list[str], failures: int = 0) -> None:
        self.source_type = source_type
        self._docs = docs
        self._failures = failures
        self.failed_files = 0

    def load(self):
        self.failed_files = self._failures
        for key in self._docs:
            yield SourceDocument(
                source_type=self.source_type,
                source_key=key,
                title=key,
                raw={"body": f"body of {key}", "path": key},
            )


def _register(name: str, connector: object):
    CONNECTOR_REGISTRY[name] = lambda cfg: (connector, MarkdownChunker(source_type=name))


def _chunk_count(store: ChunkStore, source_type: str) -> int:
    row = store._conn.execute(
        "SELECT COUNT(*) AS c FROM chunks WHERE source_type = ?", (source_type,)
    ).fetchone()
    return int(row["c"])


def test_unreadable_file_preserves_existing_chunks_and_reports_count(tmp_path: Path) -> None:
    """The core regression: a read failure must not let delete_orphans prune."""
    try:
        conn = _CountingConnector("flaky", ["a.txt", "b.txt"])
        _register("flaky", conn)
        ing, store = make_ingester_with_config(
            tmp_path, [{"name": "flaky", "type": "flaky", "path": str(tmp_path)}]
        )
        first = ing.ingest("flaky")
        before = _chunk_count(store, "flaky")
        assert before > 0
        assert first.files_failed == 0
        assert first.pruning_performed is True

        # b.txt is now unreadable: connector yields only a.txt and reports 1 failure.
        conn._docs = ["a.txt"]
        conn._failures = 1
        second = ing.ingest("flaky")

        assert second.files_failed == 1
        assert second.pruning_performed is False
        assert second.orphans_deleted == 0
        assert _chunk_count(store, "flaky") == before, "b.txt's chunks were pruned"
    finally:
        CONNECTOR_REGISTRY.pop("flaky", None)


def test_prune_anyway_forces_the_prune(tmp_path: Path) -> None:
    try:
        conn = _CountingConnector("flaky2", ["a.txt", "b.txt"])
        _register("flaky2", conn)
        ing, store = make_ingester_with_config(
            tmp_path, [{"name": "flaky2", "type": "flaky2", "path": str(tmp_path)}]
        )
        ing.ingest("flaky2")
        before = _chunk_count(store, "flaky2")

        conn._docs = ["a.txt"]
        conn._failures = 1
        result = ing.ingest("flaky2", prune_anyway=True)

        assert result.files_failed == 1
        assert result.pruning_performed is True
        assert result.orphans_deleted > 0
        assert _chunk_count(store, "flaky2") < before
    finally:
        CONNECTOR_REGISTRY.pop("flaky2", None)


def test_zero_failures_still_prunes_genuine_removals(tmp_path: Path) -> None:
    try:
        conn = _CountingConnector("clean", ["a.txt", "b.txt"])
        _register("clean", conn)
        ing, store = make_ingester_with_config(
            tmp_path, [{"name": "clean", "type": "clean", "path": str(tmp_path)}]
        )
        ing.ingest("clean")
        before = _chunk_count(store, "clean")

        conn._docs = ["a.txt"]  # b.txt genuinely deleted, no read failure
        result = ing.ingest("clean")

        assert result.files_failed == 0
        assert result.pruning_performed is True
        assert result.orphans_deleted > 0
        assert _chunk_count(store, "clean") < before
    finally:
        CONNECTOR_REGISTRY.pop("clean", None)


def test_connector_without_failed_files_attribute_still_prunes(tmp_path: Path) -> None:
    """Out-of-tree connectors (e.g. a mail consumer's EmailConnector) never set the
    attribute. They must keep today's behavior exactly."""

    class _LegacyConnector:
        source_type = "legacy"

        def __init__(self) -> None:
            self.docs = ["a.txt", "b.txt"]

        def load(self):
            for key in self.docs:
                yield SourceDocument(
                    source_type="legacy", source_key=key, title=key,
                    raw={"body": f"body of {key}", "path": key},
                )

    try:
        conn = _LegacyConnector()
        _register("legacy", conn)
        ing, store = make_ingester_with_config(
            tmp_path, [{"name": "legacy", "type": "legacy", "path": str(tmp_path)}]
        )
        ing.ingest("legacy")
        before = _chunk_count(store, "legacy")
        conn.docs = ["a.txt"]
        result = ing.ingest("legacy")
        assert result.files_failed == 0
        assert result.pruning_performed is True
        assert _chunk_count(store, "legacy") < before
    finally:
        CONNECTOR_REGISTRY.pop("legacy", None)


def test_non_integer_failed_files_is_ignored(tmp_path: Path) -> None:
    """A MagicMock connector's `failed_files` is another Mock, not an int.
    Only a non-negative int counts as a genuine report."""
    try:
        conn = _CountingConnector("mocky", ["a.txt"])
        conn.failed_files = MagicMock()  # not an int
        _register("mocky", conn)
        ing, _store = make_ingester_with_config(
            tmp_path, [{"name": "mocky", "type": "mocky", "path": str(tmp_path)}]
        )
        conn.load = lambda: iter([
            SourceDocument(source_type="mocky", source_key="a.txt", title="a",
                           raw={"body": "body", "path": "a.txt"})
        ])
        result = ing.ingest("mocky")
        assert result.files_failed == 0
        assert result.pruning_performed is True
    finally:
        CONNECTOR_REGISTRY.pop("mocky", None)


def test_raise_partway_through_iteration_still_does_not_prune(tmp_path: Path) -> None:
    """Invariant: an exception mid-iteration aborts before delete_orphans."""

    class _RaisingConnector:
        source_type = "raiser"

        def __init__(self) -> None:
            self.explode = False

        def load(self):
            yield SourceDocument(source_type="raiser", source_key="a.txt", title="a",
                                 raw={"body": "body of a", "path": "a.txt"})
            if self.explode:
                raise FileNotFoundError("volume vanished mid-scan")
            yield SourceDocument(source_type="raiser", source_key="b.txt", title="b",
                                 raw={"body": "body of b", "path": "b.txt"})

    try:
        conn = _RaisingConnector()
        _register("raiser", conn)
        ing, store = make_ingester_with_config(
            tmp_path, [{"name": "raiser", "type": "raiser", "path": str(tmp_path)}]
        )
        ing.ingest("raiser")
        before = _chunk_count(store, "raiser")
        conn.explode = True
        with pytest.raises(FileNotFoundError):
            ing.ingest("raiser")
        assert _chunk_count(store, "raiser") == before
    finally:
        CONNECTOR_REGISTRY.pop("raiser", None)


# ---------------------------------------------------------------------------
# Orphan-pruning blast-radius guard (delete_orphans' OrphanPruneRefused)
#
# `failed_files` only protects against a connector that KNOWS it failed. The
# regression this guard is for is a connector that reports failed_files == 0
# while silently yielding far fewer documents than it should (e.g. skipping
# files it believes are unchanged) -- these tests exercise that path through
# the full Ingester, one level above the ChunkStore-level guard tests in
# test_db.py.
# ---------------------------------------------------------------------------

def test_orphan_guard_refuses_when_connector_silently_under_yields(tmp_path: Path) -> None:
    """The core regression: failed_files == 0 does not mean seen_ids is
    trustworthy. A connector that silently yields almost nothing must not be
    allowed to prune the rest of the source away."""
    try:
        docs = [f"doc-{i}.txt" for i in range(60)]
        conn = _CountingConnector("bulk", docs)
        _register("bulk", conn)
        ing, store = make_ingester_with_config(
            tmp_path, [{"name": "bulk", "type": "bulk", "path": str(tmp_path)}]
        )
        first = ing.ingest("bulk")
        before = _chunk_count(store, "bulk")
        assert before > 50, "fixture must be above the guard's default floor"
        assert first.prune_refused is False

        # The under-yield bug: the connector reports NO failures but only
        # yields one document (e.g. it silently skipped the rest).
        conn._docs = ["doc-0.txt"]
        result = ing.ingest("bulk")

        assert result.files_failed == 0  # the exact case failed_files can't see
        assert result.prune_refused is True
        assert result.pruning_performed is False
        assert result.orphans_deleted == 0
        assert result.prune_refused_detail is not None
        assert "bulk" in result.prune_refused_detail
        assert _chunk_count(store, "bulk") == before, "chunks were pruned despite the guard"
    finally:
        CONNECTOR_REGISTRY.pop("bulk", None)


def test_prune_anyway_overrides_the_orphan_guard_too(tmp_path: Path) -> None:
    """--prune-anyway (prune_anyway=True) is the one escape hatch, for a
    deliberate bulk deletion the operator has actually reviewed."""
    try:
        docs = [f"doc-{i}.txt" for i in range(60)]
        conn = _CountingConnector("bulk2", docs)
        _register("bulk2", conn)
        ing, store = make_ingester_with_config(
            tmp_path, [{"name": "bulk2", "type": "bulk2", "path": str(tmp_path)}]
        )
        ing.ingest("bulk2")
        before = _chunk_count(store, "bulk2")
        assert before > 50

        conn._docs = ["doc-0.txt"]
        result = ing.ingest("bulk2", prune_anyway=True)

        assert result.prune_refused is False
        assert result.pruning_performed is True
        assert result.orphans_deleted > 0
        assert _chunk_count(store, "bulk2") < before
    finally:
        CONNECTOR_REGISTRY.pop("bulk2", None)


def test_orphan_guard_floor_allows_small_source_to_drop_to_zero(tmp_path: Path) -> None:
    """A source with only 3 chunks that legitimately drops to 0 is not
    blocked by the ratio guard -- the floor exists exactly for this case."""
    try:
        conn = _CountingConnector("tiny", ["a.txt", "b.txt", "c.txt"])
        _register("tiny", conn)
        ing, store = make_ingester_with_config(
            tmp_path, [{"name": "tiny", "type": "tiny", "path": str(tmp_path)}]
        )
        ing.ingest("tiny")
        assert _chunk_count(store, "tiny") == 3

        conn._docs = []
        result = ing.ingest("tiny")

        assert result.prune_refused is False
        assert result.pruning_performed is True
        assert result.orphans_deleted == 3
        assert _chunk_count(store, "tiny") == 0
    finally:
        CONNECTOR_REGISTRY.pop("tiny", None)


def test_orphan_guard_uses_configured_pruning_thresholds(tmp_path: Path) -> None:
    """A real CorpusConfig's [pruning] section actually reaches
    delete_orphans -- not just the engine's own hardcoded defaults."""
    try:
        docs = [f"doc-{i}.txt" for i in range(10)]
        conn = _CountingConnector("configured", docs)
        _register("configured", conn)
        config = CorpusConfig.model_validate({
            "db_path": tmp_path / "config_test.db",
            "sources": [{"name": "configured", "type": "configured", "path": str(tmp_path)}],
            # Floor of 5 makes the guard apply to a source this small, where
            # the engine default (50) would not.
            "pruning": {"max_orphan_ratio": 0.2, "min_chunks_for_guard": 5},
        })
        store = ChunkStore(tmp_path / "test.db", embedding_dim=DIM)
        ing = Ingester(config=config, store=store, embedder=fake_embedder())
        try:
            ing.ingest("configured")
            assert _chunk_count(store, "configured") == 10

            conn._docs = ["doc-0.txt"]  # drops 9/10 = 90%
            result = ing.ingest("configured")

            assert result.prune_refused is True
        finally:
            store.close()
    finally:
        CONNECTOR_REGISTRY.pop("configured", None)


# ---------------------------------------------------------------------------
# skipped_files: a connector's optional "permanently unreadable, don't
# suppress pruning for it" counter, distinct from failed_files' "might
# succeed next time, suppress pruning".
# ---------------------------------------------------------------------------

class _SkippingConnector:
    """Connector that yields some docs and reports N permanently-unreadable
    files via `skipped_files` (not `failed_files`)."""

    def __init__(self, source_type: str, docs: list[str], skips: int = 0) -> None:
        self.source_type = source_type
        self._docs = docs
        self._skips = skips
        self.failed_files = 0
        self.skipped_files = 0

    def load(self):
        self.failed_files = 0
        self.skipped_files = self._skips
        for key in self._docs:
            yield SourceDocument(
                source_type=self.source_type,
                source_key=key,
                title=key,
                raw={"body": f"body of {key}", "path": key},
            )


def test_skipped_files_does_not_suppress_pruning(tmp_path: Path) -> None:
    """Unlike failed_files, a nonzero skipped_files must NOT block pruning --
    those files are permanently unreadable by the connector's own design, so
    their absence from the index is not evidence of anything."""
    try:
        conn = _SkippingConnector("raw_photos", ["a.jpg", "b.jpg"])
        _register("raw_photos", conn)
        ing, store = make_ingester_with_config(
            tmp_path, [{"name": "raw_photos", "type": "raw_photos", "path": str(tmp_path)}]
        )
        ing.ingest("raw_photos")
        before = _chunk_count(store, "raw_photos")

        # b.jpg is now reported as permanently unsupported (e.g. an
        # unsupported raw image format) -- not a transient failure.
        conn._docs = ["a.jpg"]
        conn._skips = 1
        result = ing.ingest("raw_photos")

        assert result.files_failed == 0
        assert result.files_skipped == 1
        assert result.pruning_performed is True  # NOT suppressed, unlike failed_files
        assert result.orphans_deleted > 0
        assert _chunk_count(store, "raw_photos") < before
    finally:
        CONNECTOR_REGISTRY.pop("raw_photos", None)


def test_skipped_files_is_reported_even_when_zero(tmp_path: Path) -> None:
    try:
        conn = _SkippingConnector("plain", ["a.txt"])
        _register("plain", conn)
        ing, _store = make_ingester_with_config(
            tmp_path, [{"name": "plain", "type": "plain", "path": str(tmp_path)}]
        )
        result = ing.ingest("plain")
        assert result.files_skipped == 0
    finally:
        CONNECTOR_REGISTRY.pop("plain", None)


def test_connector_without_skipped_files_attribute_behaves_exactly_as_before(
    tmp_path: Path,
) -> None:
    """Full backwards compatibility: a connector exposing neither
    failed_files nor skipped_files must keep today's behavior untouched."""

    class _LegacyConnector:
        source_type = "legacy2"

        def __init__(self) -> None:
            self.docs = ["a.txt", "b.txt"]

        def load(self):
            for key in self.docs:
                yield SourceDocument(
                    source_type="legacy2", source_key=key, title=key,
                    raw={"body": f"body of {key}", "path": key},
                )

    try:
        conn = _LegacyConnector()
        _register("legacy2", conn)
        ing, store = make_ingester_with_config(
            tmp_path, [{"name": "legacy2", "type": "legacy2", "path": str(tmp_path)}]
        )
        ing.ingest("legacy2")
        before = _chunk_count(store, "legacy2")
        conn.docs = ["a.txt"]
        result = ing.ingest("legacy2")
        assert result.files_skipped == 0
        assert result.pruning_performed is True
        assert _chunk_count(store, "legacy2") < before
    finally:
        CONNECTOR_REGISTRY.pop("legacy2", None)
