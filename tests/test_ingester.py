"""Unit tests for corpus.ingester.Ingester.

Tests use injected store/embedder to avoid touching real connectors or
embedding APIs. build_pipeline is patched at its import site in ingester.py.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from corpus.db.sqlite import ChunkStore
from corpus.ingester import Ingester, IngestResult
from corpus.types import Chunk, ChunkKind, ChunkMetadata
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
