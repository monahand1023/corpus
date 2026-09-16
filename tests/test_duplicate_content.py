"""The same passage indexed from more than one document.

Measured on live archives: 12,076 redundant chunks in one (6.5% of the index),
897 in another (1.2%). Traced in the transcript source to 168 recordings
present twice under different paths -- 17.8 GPU-hours of audio transcribed
twice, and a search that can return the same passage twice from two files.

REPORTED, NEVER DEDUPLICATED. Some duplication is legitimate and must not be
touched: an email thread quotes what it replies to, and shared boilerplate
repeats by design. The 1.2% archive is almost entirely that. Deciding which
copies are waste needs a person, so this hands them the evidence.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from corpus.survey.duplicates import find_duplicate_content

SCHEMA = """
CREATE TABLE chunks (
    id TEXT PRIMARY KEY,
    source_type TEXT NOT NULL,
    source_key TEXT NOT NULL,
    content TEXT NOT NULL,
    content_hash TEXT NOT NULL
);
"""


def _db(tmp_path: Path, rows: list[tuple[str, str, str]]) -> Path:
    path = tmp_path / "index.db"
    conn = sqlite3.connect(path)
    conn.executescript(SCHEMA)
    conn.executemany(
        "INSERT INTO chunks (id, source_type, source_key, content, content_hash)"
        " VALUES (?,?,?,?,?)",
        [(str(i), st, key, text, str(hash(text)))
         for i, (st, key, text) in enumerate(rows)],
    )
    conn.commit()
    conn.close()
    return path


def test_an_index_with_no_duplicates_reports_none(tmp_path: Path) -> None:
    db = _db(tmp_path, [
        ("notes", "a.md", "the first distinct passage"),
        ("notes", "b.md", "an entirely different passage"),
    ])
    report = find_duplicate_content(db)
    assert report.redundant_chunks == 0
    assert report.duplicate_documents == []


def test_the_same_passage_in_two_documents_is_counted(tmp_path: Path) -> None:
    db = _db(tmp_path, [
        ("notes", "a.md", "shared passage"),
        ("notes", "b.md", "shared passage"),
        ("notes", "c.md", "unique passage"),
    ])
    report = find_duplicate_content(db)
    assert report.redundant_chunks == 1
    assert report.shared_passages == 1


def test_a_passage_repeated_INSIDE_one_document_is_not_duplication(
    tmp_path: Path,
) -> None:
    """A refrain, a repeated heading, a boilerplate footer -- one document's
    own business, and not evidence of a duplicate file."""
    db = _db(tmp_path, [
        ("notes", "a.md", "same passage"),
        ("notes", "a.md", "same passage"),
    ])
    assert find_duplicate_content(db).redundant_chunks == 0


def test_documents_that_fully_duplicate_each_other_are_named(tmp_path: Path) -> None:
    """The actionable case: the same file present in two places."""
    db = _db(tmp_path, [
        ("tr", "/vol/one/clip.mov", "first half of the talk"),
        ("tr", "/vol/one/clip.mov", "second half of the talk"),
        ("tr", "/backup/clip.mov", "first half of the talk"),
        ("tr", "/backup/clip.mov", "second half of the talk"),
        ("tr", "/vol/other.mov", "something else entirely"),
    ])
    report = find_duplicate_content(db)
    pairs = {(a, b) for a, b, _ in report.duplicate_documents}
    assert ("/backup/clip.mov", "/vol/one/clip.mov") in pairs or \
           ("/vol/one/clip.mov", "/backup/clip.mov") in pairs
    assert report.redundant_chunks == 2


def test_coverage_is_reported_so_an_empty_index_cannot_read_as_clean(
    tmp_path: Path,
) -> None:
    db = _db(tmp_path, [])
    report = find_duplicate_content(db)
    assert report.coverage.vacuous is True


def test_the_share_is_reported_as_a_fraction_of_the_index(tmp_path: Path) -> None:
    db = _db(tmp_path, [
        ("notes", "a.md", "shared"), ("notes", "b.md", "shared"),
        ("notes", "c.md", "one"), ("notes", "d.md", "two"),
    ])
    report = find_duplicate_content(db)
    assert round(report.percent, 1) == 25.0
