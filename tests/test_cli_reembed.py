"""`corpus-reembed` spends real money, so what it prints before it does
matters as much as what it does.

The archives this exists for hold 1.4M chunks. Re-embedding them is a real
bill at any provider, and the only warning is this command's own output.
"""

from __future__ import annotations

import math
from pathlib import Path

from corpus.config import CorpusConfig, EmbedderConfig
from corpus.db.sqlite import ChunkStore
from corpus.types import Chunk, ChunkKind, ChunkMetadata
from corpus.util.hash import chunk_id, sha256

DIM = 8


def _config(db: Path, dim: int = DIM) -> CorpusConfig:
    return CorpusConfig(db_path=db, embedder=EmbedderConfig(provider="hash", dim=dim))


def _store(tmp_path: Path, n: int = 3) -> Path:
    db = tmp_path / "c.db"
    store = ChunkStore(db, embedding_dim=DIM)
    store.upsert_batch([
        (
            Chunk(
                id=chunk_id("notes", f"k{i}", ChunkKind.BODY, i),
                content=f"Body {i} with distinct wording.",
                content_hash=sha256(f"c{i}"),
                metadata=ChunkMetadata(
                    source_type="notes", source_key=f"k{i}",
                    chunk_kind=ChunkKind.BODY, chunk_index=i, title=f"k{i}",
                ),
            ),
            [math.sin((i + 1) * (j + 1) * 0.1) for j in range(DIM)],
        )
        for i in range(n)
    ])
    store.close()
    return db


def _run(monkeypatch, argv, *, cfg):
    import corpus.cli.reembed as mod

    monkeypatch.setattr(mod, "load_config_or_exit", lambda _p: cfg)
    monkeypatch.setattr("sys.argv", argv)
    return mod.main()


def test_dry_run_prices_it_and_spends_nothing(tmp_path, monkeypatch, capsys):
    db = _store(tmp_path, n=3)
    before = db.read_bytes()
    code = _run(monkeypatch, ["corpus-reembed", "--dry-run"], cfg=_config(db))
    out = capsys.readouterr().out
    assert code == 0
    assert "chunks to re-embed" in out
    assert "estimated tokens" in out
    assert db.read_bytes() == before, "a dry run wrote to the database"


def test_it_refuses_to_run_without_confirmation(tmp_path, monkeypatch, capsys):
    """An irreversible, billed operation must not start because someone
    pressed enter."""
    db = _store(tmp_path, n=3)
    monkeypatch.setattr("builtins.input", lambda _p="": "n")
    code = _run(monkeypatch, ["corpus-reembed"], cfg=_config(db))
    assert code == 1
    assert "estimated tokens" in capsys.readouterr().out


def test_a_missing_database_is_an_error_not_an_empty_run(tmp_path, monkeypatch):
    code = _run(
        monkeypatch, ["corpus-reembed", "--dry-run"], cfg=_config(tmp_path / "absent.db")
    )
    assert code == 1


def test_the_full_run_swaps_and_reports(tmp_path, monkeypatch, capsys):
    db = _store(tmp_path, n=3)
    code = _run(monkeypatch, ["corpus-reembed", "--yes"], cfg=_config(db))
    out = capsys.readouterr().out
    assert code == 0, out
    assert "3" in out
    store = ChunkStore(db, embedding_dim=DIM, read_only=True)
    try:
        assert store.stats()["total"] == 3
    finally:
        store.close()


def test_stage_only_leaves_the_archive_untouched(tmp_path, monkeypatch, capsys):
    """Splitting the paid pass from the swap is the point: you can stage
    overnight and swap when you are watching."""
    from corpus.reembed import staged_count

    db = _store(tmp_path, n=3)
    code = _run(monkeypatch, ["corpus-reembed", "--yes", "--stage-only"], cfg=_config(db))
    assert code == 0
    assert staged_count(db) == 3
    store = ChunkStore(db, embedding_dim=DIM, read_only=True)
    try:
        assert store.vector_search([0.1] * DIM, top_k=1), "vectors were disturbed"
    finally:
        store.close()
