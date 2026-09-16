"""Re-embed an archive in place, without destroying what was paid for.

THE GAP. There is no supported way to change embedding provider, model or
dim on an existing archive. A re-ingest re-embeds NOTHING -- `upsert` returns
early when the content hash is unchanged -- and the dim guard's own error
says "Re-ingest from scratch with the new dim", which means `corpus-reset
--all`. That destroys everything stored only in the database:

    one archive:  248,143 contextualized chunks
    another:       11,841 contextualized chunks + 6,138 summaries

All of it Anthropic spend, with no on-disk backup. The safe operation is
narrower than a re-ingest: only the VECTORS change.

WHY A STAGING TABLE. The expensive part is the API calls, and the operation
runs over up to 1.4M chunks. Embeddings accumulate in a staging table so an
interruption never has to pay for the same chunk twice, and the swap into
`chunks_vec` -- which must be dropped and recreated, because vec0 bakes the
dimension into its schema -- is one transaction at the end. A half-swapped
vector table would be an archive that returns wrong neighbours and says
nothing.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from corpus.db.sqlite import ChunkStore
from corpus.reembed import (
    ReembedIncomplete,
    reembed_plan,
    stage_embeddings,
    staged_count,
    swap_in_staged,
)
from corpus.types import Chunk, ChunkKind, ChunkMetadata
from corpus.util.hash import chunk_id, sha256

OLD_DIM = 8
NEW_DIM = 16


def _vec(seed: int, dim: int) -> list[float]:
    return [math.sin((seed + 1) * (i + 1) * 0.1) for i in range(dim)]


def _store(tmp_path: Path, n: int = 5, dim: int = OLD_DIM) -> Path:
    db = tmp_path / "c.db"
    store = ChunkStore(db, embedding_dim=dim)
    store.upsert_batch([
        (
            Chunk(
                id=chunk_id("notes", f"k{i}", ChunkKind.BODY, i),
                content=f"Body number {i} with its own wording.",
                content_hash=sha256(f"c{i}"),
                metadata=ChunkMetadata(
                    source_type="notes", source_key=f"k{i}",
                    chunk_kind=ChunkKind.BODY, chunk_index=i, title=f"k{i}",
                ),
            ),
            _vec(i, dim),
        )
        for i in range(n)
    ])
    # Paid work that must survive: a context blurb and a summary.
    store.set_context(
        chunk_id("notes", "k0", ChunkKind.BODY, 0),
        "From the k0 note.",
        _vec(99, dim),
    )
    store.upsert_summary(
        source_type="notes", source_key="k0", summary="A summary worth money.",
        doc_hash="h", model="m", token_count=10,
    )
    store.close()
    return db


# --- planning ----------------------------------------------------------------


def test_the_plan_reports_what_will_be_embedded_before_any_spend(tmp_path):
    db = _store(tmp_path, n=5)
    plan = reembed_plan(db, new_dim=NEW_DIM)
    assert plan.chunks == 5
    assert plan.current_dim == OLD_DIM
    assert plan.new_dim == NEW_DIM
    assert plan.estimated_tokens > 0
    assert "5" in plan.describe()


def test_the_plan_does_not_modify_anything(tmp_path):
    db = _store(tmp_path, n=3)
    before = db.read_bytes()
    reembed_plan(db, new_dim=NEW_DIM)
    assert db.read_bytes() == before, "planning wrote to the database"


# --- staging -----------------------------------------------------------------


def test_staging_accumulates_without_touching_the_live_vectors(tmp_path):
    db = _store(tmp_path, n=4)
    stage_embeddings(db, new_dim=NEW_DIM, embed=lambda texts: [_vec(9, NEW_DIM)] * len(texts))
    assert staged_count(db) == 4
    store = ChunkStore(db, embedding_dim=OLD_DIM, read_only=True)
    try:
        assert store.stats()["total"] == 4
    finally:
        store.close()


def test_an_interrupted_run_does_not_pay_for_the_same_chunk_twice(tmp_path):
    db = _store(tmp_path, n=6)
    calls: list[int] = []

    def flaky(texts):
        calls.append(len(texts))
        if sum(calls) > 3:
            raise RuntimeError("network died")
        return [_vec(1, NEW_DIM)] * len(texts)

    with pytest.raises(RuntimeError):
        stage_embeddings(db, new_dim=NEW_DIM, embed=flaky, batch_size=3)
    done = staged_count(db)
    assert done == 3

    # Resume: only the remaining chunks are embedded.
    seen: list[int] = []

    def ok(texts):
        seen.append(len(texts))
        return [_vec(2, NEW_DIM)] * len(texts)

    stage_embeddings(db, new_dim=NEW_DIM, embed=ok, batch_size=3)
    assert sum(seen) == 3, f"re-paid for already-staged chunks: {seen}"
    assert staged_count(db) == 6


def test_staging_for_a_different_dim_starts_over(tmp_path):
    """Changing your mind about the target dim must not mix vector widths."""
    db = _store(tmp_path, n=3)
    stage_embeddings(db, new_dim=NEW_DIM, embed=lambda t: [_vec(1, NEW_DIM)] * len(t))
    assert staged_count(db) == 3
    stage_embeddings(db, new_dim=32, embed=lambda t: [_vec(1, 32)] * len(t))
    assert staged_count(db) == 3
    assert staged_count(db, new_dim=NEW_DIM) == 0


# --- the swap ----------------------------------------------------------------


def test_the_swap_refuses_while_any_chunk_is_unstaged(tmp_path):
    """A half-swapped vector table is an archive that returns wrong
    neighbours and says nothing about it."""
    db = _store(tmp_path, n=5)
    stage_embeddings(
        db, new_dim=NEW_DIM, embed=lambda t: [_vec(1, NEW_DIM)] * len(t), limit=2
    )
    with pytest.raises(ReembedIncomplete):
        swap_in_staged(db, new_dim=NEW_DIM)


def test_the_swap_changes_the_dim_and_keeps_everything_else(tmp_path):
    db = _store(tmp_path, n=4)
    stage_embeddings(db, new_dim=NEW_DIM, embed=lambda t: [_vec(3, NEW_DIM)] * len(t))
    swap_in_staged(db, new_dim=NEW_DIM)

    store = ChunkStore(db, embedding_dim=NEW_DIM)
    try:
        assert store.stats()["total"] == 4, "chunks were lost"
        ctx = store._conn.execute(
            "SELECT context FROM chunks WHERE context IS NOT NULL AND context != ''"
        ).fetchall()
        assert len(ctx) == 1, "a paid context blurb was destroyed"
        assert store.get_summary("notes", "k0") is not None, "a paid summary was destroyed"
        hits = store.vector_search(_vec(3, NEW_DIM), top_k=3)
        assert hits, "the new vectors are not searchable"
    finally:
        store.close()


def test_the_old_dim_is_rejected_after_the_swap(tmp_path):
    """The stamp must move with the vectors, or the next open silently
    mismatches."""
    from corpus.db.sqlite import EmbeddingDimMismatch

    db = _store(tmp_path, n=2)
    stage_embeddings(db, new_dim=NEW_DIM, embed=lambda t: [_vec(1, NEW_DIM)] * len(t))
    swap_in_staged(db, new_dim=NEW_DIM)
    with pytest.raises(EmbeddingDimMismatch):
        ChunkStore(db, embedding_dim=OLD_DIM)


def test_the_staging_table_is_cleared_after_a_successful_swap(tmp_path):
    db = _store(tmp_path, n=3)
    stage_embeddings(db, new_dim=NEW_DIM, embed=lambda t: [_vec(1, NEW_DIM)] * len(t))
    swap_in_staged(db, new_dim=NEW_DIM)
    assert staged_count(db, new_dim=NEW_DIM) == 0


def test_fts_rows_survive_the_swap(tmp_path):
    """Only the vectors change. Losing BM25 would halve hybrid retrieval and
    the archive would still look fine."""
    db = _store(tmp_path, n=3)
    stage_embeddings(db, new_dim=NEW_DIM, embed=lambda t: [_vec(1, NEW_DIM)] * len(t))
    swap_in_staged(db, new_dim=NEW_DIM)
    store = ChunkStore(db, embedding_dim=NEW_DIM, read_only=True)
    try:
        assert store.fts_search("wording", top_k=5), "FTS rows were lost"
    finally:
        store.close()
