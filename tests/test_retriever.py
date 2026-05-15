from __future__ import annotations

import math
import re
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from corpus.db.sqlite import ChunkStore
from corpus.retriever import Retriever
from corpus.types import Chunk, ChunkKind, ChunkMetadata
from corpus.util.hash import chunk_id, sha256

DIM = 1024


def fake_embedding(seed: int) -> list[float]:
    return [math.sin((seed + 1) * (i + 1) * 0.001) for i in range(DIM)]


def make_chunk(source_type: str, source_key: str, idx: int, content: str) -> Chunk:
    return Chunk(
        id=chunk_id(source_type, source_key, ChunkKind.HEADER, idx),
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


@pytest.fixture
def retriever(tmp_path: Path) -> Retriever:
    store = ChunkStore(tmp_path / "ret.db", embedding_dim=DIM)
    items = []
    for i in range(6):
        items.append((make_chunk("notes", f"doc-{i}", 0, f"note {i}"), fake_embedding(i)))
    for i in range(3):
        items.append((make_chunk("notes", "doc-0", i + 1, f"doc-0 body {i}"), fake_embedding(i + 10)))
    for i in range(3):
        items.append((make_chunk("papers", f"p-{i}", 0, f"paper {i}"), fake_embedding(i + 20)))
    store.upsert_batch(items)

    embedder = MagicMock()
    embedder.embed_query = MagicMock(return_value=fake_embedding(0))
    r = Retriever(store=store, embedder=embedder)
    yield r
    store.close()


def test_dedupe_by_source_key(retriever: Retriever) -> None:
    """The doc-0 group has 4 chunks; dedupe should collapse to 1."""
    result = retriever.query("anything", top_k=10, max_per_source_type=None, hybrid=False)
    keys = [c.source_key for c in result.chunks]
    assert keys.count("doc-0") == 1


def test_max_per_source_type_cap(retriever: Retriever) -> None:
    result = retriever.query("anything", top_k=10, max_per_source_type=2, hybrid=False)
    counts: dict[str, int] = {}
    for c in result.chunks:
        counts[c.source_type] = counts.get(c.source_type, 0) + 1
    for n in counts.values():
        assert n <= 2


def test_diversity_makes_room_for_papers(retriever: Retriever) -> None:
    """With cap=2, top-6 should include at least one paper not just notes."""
    result = retriever.query("anything", top_k=6, max_per_source_type=2, hybrid=False)
    sources = {c.source_type for c in result.chunks}
    assert "papers" in sources


def test_reference_pattern_drives_expand_context(tmp_path: Path) -> None:
    """expand_context references mode uses regex from config, not hardcoded."""
    store = ChunkStore(tmp_path / "ref.db", embedding_dim=DIM)
    items = [
        # Seed mentions TKT-9
        (make_chunk("tickets", "TKT-1", 0, "depends on TKT-9 to ship"), fake_embedding(1)),
        (make_chunk("tickets", "TKT-9", 0, "the dependency ticket"), fake_embedding(2)),
        (make_chunk("tickets", "TKT-5", 0, "unrelated"), fake_embedding(3)),
    ]
    store.upsert_batch(items)
    embedder = MagicMock()
    pattern = re.compile(r"\bTKT-\d+\b")
    r = Retriever(
        store=store,
        embedder=embedder,
        reference_patterns=[(pattern, "tickets")],
    )

    seed_id = chunk_id("tickets", "TKT-1", ChunkKind.HEADER, 0)
    related = r.expand_context(seed_id, include=["references"])
    keys = {c.source_key for c in related}
    assert "TKT-9" in keys
    assert "TKT-1" not in keys  # self-cite excluded
    assert "TKT-5" not in keys  # not mentioned
    store.close()


def test_no_reference_patterns_means_no_cross_doc_references(tmp_path: Path) -> None:
    """With references=[], expand_context returns siblings + parent only."""
    store = ChunkStore(tmp_path / "noref.db", embedding_dim=DIM)
    items = [
        (make_chunk("notes", "A", 0, "mentions DOC-9"), fake_embedding(1)),
        (make_chunk("notes", "DOC-9", 0, "would-be reference"), fake_embedding(2)),
    ]
    store.upsert_batch(items)
    embedder = MagicMock()
    r = Retriever(store=store, embedder=embedder, reference_patterns=[])

    seed_id = chunk_id("notes", "A", ChunkKind.HEADER, 0)
    related = r.expand_context(seed_id, include=["references"])
    assert related == []  # no patterns configured
    store.close()


def test_auto_fts_weight_boosts_on_reference_match(tmp_path: Path) -> None:
    """When a query contains a configured reference pattern, fts_weight jumps
    to 1.0 (BM25-heavy). Prose queries stay at 0.25."""
    store = ChunkStore(tmp_path / "fw.db", embedding_dim=DIM)
    embedder = MagicMock()
    pattern = re.compile(r"\bTKT-\d+\b")
    r = Retriever(
        store=store,
        embedder=embedder,
        reference_patterns=[(pattern, "tickets")],
    )
    # Query mentioning a reference pattern → BM25-heavy
    assert r._auto_fts_weight("what happened with TKT-42") == 1.0
    # Plain prose query → semantic-dominant
    assert r._auto_fts_weight("how does the payment flow work") == 0.25
    # Generic ID hint (backticked code) also boosts, even without a configured pattern
    assert r._auto_fts_weight("what does `parse_markdown` do") == 1.0
    store.close()


def test_auto_fts_weight_no_patterns_uses_generic_hints_only(tmp_path: Path) -> None:
    store = ChunkStore(tmp_path / "fw2.db", embedding_dim=DIM)
    embedder = MagicMock()
    r = Retriever(store=store, embedder=embedder, reference_patterns=[])
    # No configured patterns — TKT-42 is just prose now
    assert r._auto_fts_weight("what happened with TKT-42") == 0.25
    # But quoted phrases still trigger the generic hint
    assert r._auto_fts_weight('search for "exact phrase"') == 1.0
    store.close()
