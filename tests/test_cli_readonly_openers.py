"""Commands that only READ must open the store read-only.

Two failures follow from opening it read-write, and both are silent:

  - A MISSING database is CREATED. Point `corpus-eval` at a config whose
    `db_path` has a typo and it builds an empty store, then reports recall
    0.000 — a number that describes a system nobody is running, presented as
    a measurement.
  - A STALE index is REBUILT on open (see `ChunkStore._migrate_fts`, which
    migrates a small store from its constructor). A benchmark then rewrites
    the database it is timing, and inspecting a backup destroys the state
    being inspected.

`corpus-query` and the MCP server already open read-only and say why. These
four did not.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from corpus.config import CorpusConfig, EmbedderConfig, SourceConfig
from corpus.db.sqlite import ChunkStore
from corpus.types import Chunk, ChunkKind, ChunkMetadata
from corpus.util.hash import chunk_id, sha256

DIM = 64


def _config(db: Path, **kw) -> CorpusConfig:
    return CorpusConfig(
        db_path=db, embedder=EmbedderConfig(provider="hash", dim=DIM), **kw
    )


def _stale_store(path: Path) -> None:
    """A real store whose FTS stamp is missing, i.e. due for a rebuild."""
    store = ChunkStore(path, embedding_dim=DIM)
    store.upsert_batch([
        (
            Chunk(
                id=chunk_id("notes", "k", ChunkKind.BODY, 0),
                content="東京で会議をしました",
                content_hash=sha256("k"),
                metadata=ChunkMetadata(
                    source_type="notes",
                    source_key="k",
                    chunk_kind=ChunkKind.BODY,
                    chunk_index=0,
                    title="k",
                ),
            ),
            [math.sin((j + 1) * 0.001) for j in range(DIM)],
        )
    ])
    store._conn.execute("DELETE FROM schema_meta WHERE key = 'fts_version'")
    store._conn.commit()
    store.close()


def _stamped(path: Path) -> str | None:
    store = ChunkStore(path, embedding_dim=DIM, read_only=True)
    try:
        return store.fts_version()
    finally:
        store.close()


# --- corpus-list -------------------------------------------------------------


def test_list_does_not_rebuild_the_index_it_is_counting(tmp_path, monkeypatch):
    import corpus.cli.list_sources as mod

    db = tmp_path / "stale.db"
    _stale_store(db)
    cfg = _config(db, sources=[SourceConfig(name="notes", type="markdown", path=".")])

    monkeypatch.setattr(mod, "load_config_or_exit", lambda _p: cfg)
    monkeypatch.setattr("sys.argv", ["corpus-list"])
    assert mod.main() == 0
    assert _stamped(db) is None, "corpus-list rewrote the database"


# --- corpus-benchmark --------------------------------------------------------


def test_benchmark_does_not_rewrite_the_database_it_is_timing(tmp_path):
    from corpus.cli.benchmark import _run_benchmark

    db = tmp_path / "stale.db"
    _stale_store(db)

    _run_benchmark(
        provider="hash", model="", dim=DIM, config=_config(db),
        queries=["meeting"], runs_per_query=1, top_k=3,
    )
    assert _stamped(db) is None, "the benchmark migrated the index mid-measurement"


def test_benchmark_on_a_missing_database_says_so_instead_of_creating_one(tmp_path):
    from corpus.cli.benchmark import _run_benchmark

    db = tmp_path / "absent.db"
    with pytest.raises(SystemExit):
        _run_benchmark(
            provider="hash", model="", dim=DIM, config=_config(db),
            queries=["meeting"], runs_per_query=1, top_k=3,
        )
    assert not db.exists(), "an empty database was created to benchmark"


# --- corpus-eval / corpus-judge ---------------------------------------------


def test_judge_on_a_missing_database_says_so_instead_of_creating_one(tmp_path, monkeypatch):
    import corpus.cli.judge as mod

    db = tmp_path / "absent.db"
    monkeypatch.setattr(mod, "load_config_or_exit", lambda _p: _config(db))
    with pytest.raises(SystemExit):
        mod._retriever_from_config(None)
    assert not db.exists(), "an empty database was created to judge against"
