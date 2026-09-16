"""corpus-migrate-fts rewrites a live index, and had no tests.

The command exists because the rebuild is one long transaction holding
SQLite's only writer lock, so it must be something an operator CHOSE. Two
properties carry that whole design and neither was pinned:

  - a `--dry-run` must not be able to cause the very migration it reports on
    (it opens the store read-only for exactly this reason), and
  - a rebuild that did not land must be reported as a failure, because the
    caller's next action is to start writing to an index they believe is
    current.

`ChunkStore._migrate_fts` itself is covered in `tests/test_db.py`; what is
tested here is the command wrapped around it.
"""

from __future__ import annotations

import math
from pathlib import Path

from corpus.config import CorpusConfig, EmbedderConfig
from corpus.db.sqlite import FTS_VERSION, ChunkStore
from corpus.types import Chunk, ChunkKind, ChunkMetadata
from corpus.util.hash import chunk_id, sha256

DIM = 64


def _config(db: Path) -> CorpusConfig:
    return CorpusConfig(db_path=db, embedder=EmbedderConfig(provider="hash", dim=DIM))


def _stale_store(path: Path, *, chunks: int = 3) -> None:
    """A store whose index is stamped with a superseded normalization."""
    store = ChunkStore(path, embedding_dim=DIM)
    store.upsert_batch([
        (
            Chunk(
                id=chunk_id("notes", f"k{i}", ChunkKind.BODY, i),
                content=f"東京で会議をしました {i}",
                content_hash=sha256(f"c{i}"),
                metadata=ChunkMetadata(
                    source_type="notes",
                    source_key=f"k{i}",
                    chunk_kind=ChunkKind.BODY,
                    chunk_index=i,
                    title=f"k{i}",
                ),
            ),
            [math.sin((i + 1) * (j + 1) * 0.001) for j in range(DIM)],
        )
        for i in range(chunks)
    ])
    store._conn.execute("DELETE FROM schema_meta WHERE key = 'fts_version'")
    store._conn.commit()
    store.close()


def _stamp(path: Path) -> str | None:
    store = ChunkStore(path, embedding_dim=DIM, read_only=True)
    try:
        return store.fts_version()
    finally:
        store.close()


def _run(monkeypatch, argv, *, cfg):
    import corpus.cli.migrate_fts as mod

    monkeypatch.setattr(mod, "load_config_or_exit", lambda _p: cfg)
    monkeypatch.setattr("sys.argv", argv)
    return mod.main()


def test_a_missing_database_is_an_error_not_a_traceback(tmp_path, monkeypatch, capsys):
    """Running this in a project that has not indexed anything yet is an
    ordinary mistake. Every other command that opens the store by path says
    so in one line; this one let FileNotFoundError out of the CLI."""
    cfg = _config(tmp_path / "absent.db")

    code = _run(monkeypatch, ["corpus-migrate-fts", "--dry-run"], cfg=cfg)

    assert code == 1
    err = capsys.readouterr().err
    assert "absent.db" in err
    assert "not found" in err.lower()


def test_dry_run_does_not_perform_the_migration_it_reports_on(tmp_path, monkeypatch):
    """The reason the probe is opened read-only. Opening a stale store
    read-write is itself what triggers a rebuild on a small store, so a
    dry run that opened it writable would migrate the database it was asked
    only to look at."""
    db = tmp_path / "stale.db"
    _stale_store(db)
    assert _stamp(db) is None

    assert _run(monkeypatch, ["corpus-migrate-fts", "--dry-run"], cfg=_config(db)) == 0
    assert _stamp(db) is None, "a dry run rebuilt the index"


def test_a_current_index_is_left_alone(tmp_path, monkeypatch, capsys):
    db = tmp_path / "current.db"
    _stale_store(db)
    ChunkStore(db, embedding_dim=DIM).close()  # small store migrates on open
    assert _stamp(db) == FTS_VERSION

    assert _run(monkeypatch, ["corpus-migrate-fts"], cfg=_config(db)) == 0
    assert "Nothing to do" in capsys.readouterr().out


def test_a_rebuild_restores_search_for_text_the_old_index_missed(tmp_path, monkeypatch):
    db = tmp_path / "stale.db"
    _stale_store(db)

    assert _run(monkeypatch, ["corpus-migrate-fts"], cfg=_config(db)) == 0
    assert _stamp(db) == FTS_VERSION

    store = ChunkStore(db, embedding_dim=DIM, read_only=True)
    try:
        assert store.fts_search("東京", top_k=5), "CJK still unmatched after rebuild"
    finally:
        store.close()


def test_a_rebuild_that_did_not_land_is_reported_as_a_failure(tmp_path, monkeypatch, capsys):
    """Exit 0 here means 'the index is current', and the operator's next act
    is to let writers back on. A rebuild that silently failed must not look
    like one that worked."""
    import corpus.cli.migrate_fts as mod

    db = tmp_path / "stale.db"
    _stale_store(db)
    # The store will stamp the version it actually knows, never this one.
    monkeypatch.setattr(mod, "FTS_VERSION", "an-unreachable-version")

    code = _run(monkeypatch, ["corpus-migrate-fts"], cfg=_config(db))

    assert code == 1
    assert "ERROR" in capsys.readouterr().out
