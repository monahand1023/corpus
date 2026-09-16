"""Which duplicate is the one to drop?

`find_duplicate_content` reports duplicated PASSAGES, deliberately without
deleting: a quoted email and a copied file look identical to it. But it
leaves the operator with a percentage and no decision.

The decidable question is narrower. A document is WHOLLY duplicated when
every one of its chunks also appears under another document — dropping it
loses nothing. Measured on a live archive: 1,972 such documents holding
10,223 chunks (5.5% of the index).

THEY COME IN PAIRS, and that is the trap. Both members of a pair are wholly
contained in the other, so both appear in the list, and excluding the list
deletes the content entirely. Only ONE of each pair can go, which is a
choice about which path is canonical — and a path under a folder named
"Backup", or inside a `.zip` whose contents are also extracted, is the
answer often enough to be worth suggesting.
"""

from __future__ import annotations

import sqlite3

from corpus.survey.duplicates import duplicate_documents


def _index(tmp_path, rows):
    """rows: (source_key, [content_hash, ...])."""
    db = tmp_path / "c.db"
    conn = sqlite3.connect(db)
    conn.execute(
        "CREATE TABLE chunks (content_hash TEXT, source_type TEXT, source_key TEXT)"
    )
    for key, hashes in rows:
        for h in hashes:
            conn.execute(
                "INSERT INTO chunks VALUES (?, 'docs', ?)", (h, key)
            )
    conn.commit()
    conn.close()
    return db


def test_a_document_wholly_contained_in_another_is_paired_with_it(tmp_path):
    db = _index(
        tmp_path,
        [("live/report.docx", ["a", "b", "c"]),
         ("Backup/report.docx", ["a", "b", "c"])],
    )
    pairs = duplicate_documents(db)
    assert len(pairs) == 1
    assert pairs[0].shared == 3


def test_the_backup_copy_is_the_one_suggested_for_removal(tmp_path):
    db = _index(
        tmp_path,
        [("Work/report.docx", ["a", "b"]),
         ("Windows Backup/Desktop/report.docx", ["a", "b"])],
    )
    pair = duplicate_documents(db)[0]
    assert "Backup" in pair.drop
    assert "Backup" not in pair.keep


def test_the_zipped_copy_is_dropped_when_the_extracted_one_exists(tmp_path):
    db = _index(
        tmp_path,
        [("proj/models.txt", ["a", "b"]),
         ("proj.zip::proj/models.txt", ["a", "b"])],
    )
    pair = duplicate_documents(db)[0]
    assert ".zip::" in pair.drop


def test_a_pair_is_reported_once_not_twice(tmp_path):
    """Both members are wholly contained in the other. Listing both, and
    acting on the list, deletes the content."""
    db = _index(tmp_path, [("a/x.md", ["h1", "h2"]), ("b/x.md", ["h1", "h2"])])
    pairs = duplicate_documents(db)
    assert len(pairs) == 1
    assert {pairs[0].keep, pairs[0].drop} == {"a/x.md", "b/x.md"}


def test_a_partial_overlap_is_not_offered_for_removal(tmp_path):
    """Two versions of a document share most passages and differ in some.
    Dropping either loses the difference."""
    db = _index(
        tmp_path,
        [("report_v1.pdf", ["a", "b", "only-in-v1"]),
         ("report_v2.pdf", ["a", "b", "only-in-v2"])],
    )
    assert duplicate_documents(db) == []


def test_a_single_chunk_document_is_not_offered(tmp_path):
    """One shared passage is a stock sentence, not a copied file."""
    db = _index(tmp_path, [("a.md", ["boiler"]), ("b.md", ["boiler"])])
    assert duplicate_documents(db) == []


def test_three_copies_of_one_document_leave_exactly_one_survivor(tmp_path):
    """The arithmetic that matters: N copies must yield N-1 removals, never
    N, and never 1."""
    db = _index(
        tmp_path,
        [("live/x.md", ["a", "b"]),
         ("Backup/x.md", ["a", "b"]),
         ("Old Backup/x.md", ["a", "b"])],
    )
    pairs = duplicate_documents(db)
    dropped = {p.drop for p in pairs}
    kept = {p.keep for p in pairs}
    assert len(dropped) == 2
    assert kept == {"live/x.md"}
    assert not (dropped & kept), "a document was both kept and dropped"


def test_the_removable_chunk_count_is_reported(tmp_path):
    db = _index(
        tmp_path,
        [("live/x.md", ["a", "b", "c"]), ("Backup/x.md", ["a", "b", "c"])],
    )
    assert sum(p.shared for p in duplicate_documents(db)) == 3
