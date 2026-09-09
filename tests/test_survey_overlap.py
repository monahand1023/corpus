"""Tests for `corpus.survey.overlap` — estimating how much of a directory is
already indexed in an existing corpus database.

Builds a real (tiny) `ChunkStore` in `tmp_path` rather than mocking it —
this module's whole job is the interaction between phrase extraction and
`ChunkStore.fts_search` + substring confirmation, which a mock would hide
bugs in.
"""

from __future__ import annotations

import math
from pathlib import Path

from corpus.db.sqlite import ChunkStore
from corpus.survey.overlap import _extract_phrase, run_overlap_survey
from corpus.types import Chunk, ChunkKind, ChunkMetadata
from corpus.util.hash import chunk_id, sha256

DIM = 8


def _fake_embedding(seed: int) -> list[float]:
    return [math.sin((seed + 1) * (i + 1) * 0.001) for i in range(DIM)]


def _make_chunk(source_type: str, key: str, content: str) -> Chunk:
    return Chunk(
        id=chunk_id(source_type, key, ChunkKind.SECTION, 0),
        content=content,
        content_hash=sha256(content),
        metadata=ChunkMetadata(
            source_type=source_type, source_key=key, chunk_kind=ChunkKind.SECTION,
            chunk_index=0, title=key,
        ),
    )


def _build_store(db_path: Path, docs: dict[str, str]) -> None:
    store = ChunkStore(db_path, embedding_dim=DIM)
    try:
        items = [
            (_make_chunk("archive", key, content), _fake_embedding(i))
            for i, (key, content) in enumerate(docs.items())
        ]
        store.upsert_batch(items)
    finally:
        store.close()


def _touch(root: Path, rel: str, content: str) -> Path:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    return p


# ---------------------------------------------------------------------------
# _extract_phrase
# ---------------------------------------------------------------------------


def test_extract_phrase_picks_longest_qualifying_line(tmp_path: Path) -> None:
    p = _touch(
        tmp_path,
        "doc.txt",
        "short line\nThis is a genuinely long and distinctive sentence with many words\nmid length words here ok\n",
    )
    phrase = _extract_phrase(p, min_words=8, max_read_bytes=65536)
    assert phrase == "This is a genuinely long and distinctive sentence with many words"


def test_extract_phrase_returns_none_when_no_line_qualifies(tmp_path: Path) -> None:
    p = _touch(tmp_path, "doc.txt", "too short\nalso short\n")
    assert _extract_phrase(p, min_words=8, max_read_bytes=65536) is None


def test_extract_phrase_handles_undecodable_bytes(tmp_path: Path) -> None:
    p = tmp_path / "doc.txt"
    p.write_bytes(b"\xff\xfe\x00garbage bytes that are not valid utf8 at all here today")
    # Must not raise; may or may not find a qualifying line depending on
    # decode-with-errors="ignore" output — either is fine.
    _extract_phrase(p, min_words=8, max_read_bytes=65536)


# ---------------------------------------------------------------------------
# run_overlap_survey
# ---------------------------------------------------------------------------


def test_matched_document_is_confirmed_via_substring(tmp_path: Path) -> None:
    tree = tmp_path / "tree"
    db_path = tmp_path / "archive.db"
    phrase = "The quarterly planning review covered budget allocations for next year"
    _touch(tree, "already_indexed.txt", f"intro\n{phrase}\nmore text here\n")
    _build_store(db_path, {"already_indexed": f"Some preamble.\n{phrase}\nSome epilogue."})

    result = run_overlap_survey(tree, db_path, sample_size=10, min_words=8, seed=1)

    assert result.eligible_document_count == 1
    assert result.sample_size == 1
    assert result.matched_count == 1
    assert result.sample[0].matched_source == "archive::already_indexed"
    assert result.estimated_overlap_fraction == 1.0


def test_unmatched_document_is_not_confirmed(tmp_path: Path) -> None:
    tree = tmp_path / "tree"
    db_path = tmp_path / "archive.db"
    _touch(tree, "new_doc.txt", "This sentence has never appeared anywhere in the archive before\n")
    _build_store(db_path, {"other": "Completely unrelated content about a different topic entirely today"})

    result = run_overlap_survey(tree, db_path, sample_size=10, min_words=8, seed=1)

    assert result.matched_count == 0
    assert result.estimated_overlap_fraction == 0.0
    assert result.confidence_interval_95 is not None


def test_shared_vocabulary_without_exact_phrase_is_not_a_false_match(tmp_path: Path) -> None:
    """FTS5 OR-of-terms recall alone would call this a match (both share
    several words); the substring confirmation step must reject it."""
    tree = tmp_path / "tree"
    db_path = tmp_path / "archive.db"
    _touch(
        tree,
        "doc.txt",
        "The quarterly budget review meeting happened on a sunny Tuesday afternoon\n",
    )
    _build_store(
        db_path,
        {"other": "A completely different budget meeting review happened during a rainy Tuesday morning session"},
    )

    result = run_overlap_survey(tree, db_path, sample_size=10, min_words=8, seed=1)

    assert result.matched_count == 0


def test_no_eligible_documents_yields_empty_sample(tmp_path: Path) -> None:
    tree = tmp_path / "tree"
    tree.mkdir()
    db_path = tmp_path / "archive.db"
    _build_store(db_path, {"x": "irrelevant"})

    result = run_overlap_survey(tree, db_path)

    assert result.eligible_document_count == 0
    assert result.sample == []
    assert result.estimated_overlap_fraction is None
    assert result.confidence_interval_95 is None


def test_binary_and_unsupported_extensions_are_not_sampled(tmp_path: Path) -> None:
    tree = tmp_path / "tree"
    db_path = tmp_path / "archive.db"
    _touch(tree, "doc.pdf", "this looks like pdf content but has the wrong extension for sampling")
    _build_store(db_path, {"x": "irrelevant"})

    result = run_overlap_survey(tree, db_path)

    assert result.eligible_document_count == 0


def test_sample_size_caps_how_many_are_checked_against_db(tmp_path: Path) -> None:
    tree = tmp_path / "tree"
    db_path = tmp_path / "archive.db"
    for i in range(20):
        _touch(tree, f"doc{i}.txt", f"this is a sufficiently long distinctive sentence number {i} here\n")
    _build_store(db_path, {"x": "irrelevant content unrelated to any sampled document text"})

    result = run_overlap_survey(tree, db_path, sample_size=5, seed=1)

    assert result.eligible_document_count == 20
    assert result.sample_size == 5


def test_respects_excludes(tmp_path: Path) -> None:
    tree = tmp_path / "tree"
    db_path = tmp_path / "archive.db"
    _touch(tree, "keep.txt", "this is a sufficiently long distinctive line of real content here\n")
    _touch(tree, "skip/ignored.txt", "this is another sufficiently long distinctive line of content here\n")
    _build_store(db_path, {"x": "irrelevant"})

    result = run_overlap_survey(tree, db_path, excludes=("skip",))

    assert result.eligible_document_count == 1


def test_missing_db_raises_file_not_found(tmp_path: Path) -> None:
    tree = tmp_path / "tree"
    _touch(tree, "doc.txt", "this is a sufficiently long distinctive line of real content here\n")

    try:
        run_overlap_survey(tree, tmp_path / "does_not_exist.db")
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("expected FileNotFoundError for a missing database")
