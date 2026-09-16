"""Tests for the post-ingest index quality scan.

The scan answers a question that only exists after ingest: we indexed it, is
any of it junk? Transcription artefacts are invisible from outside -- the
index reports a successful build, search returns results, and some of those
results are text no person ever said.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from corpus.survey.index_quality import NotACorpusIndexError, run_index_quality

SCHEMA = """
CREATE TABLE chunks (
    id TEXT PRIMARY KEY,
    source_type TEXT NOT NULL,
    source_key TEXT NOT NULL,
    content TEXT NOT NULL
);
"""


def _db(tmp_path: Path, rows: list[tuple[str, str, str]]) -> Path:
    path = tmp_path / "index.db"
    conn = sqlite3.connect(path)
    conn.executescript(SCHEMA)
    conn.executemany(
        "INSERT INTO chunks (id, source_type, source_key, content) VALUES (?,?,?,?)",
        [(str(i), st, key, text) for i, (st, key, text) in enumerate(rows)],
    )
    conn.commit()
    conn.close()
    return path


def test_a_clean_index_reports_clean(tmp_path: Path) -> None:
    db = _db(tmp_path, [
        ("transcripts", "a.mov", "We went to the park and fed the ducks."),
        ("notes", "b.md", "Buy milk on the way home."),
    ])
    result = run_index_quality(db)
    assert result.clean
    assert result.scanned_chunks == 2
    assert result.affected_chunks == 0


def test_a_chunk_that_is_only_a_sign_off_is_found(tmp_path: Path) -> None:
    # This should never have been indexed at all: its presence means the
    # connector is not filtering, which is a different fix from a glued tail.
    db = _db(tmp_path, [("transcripts", "a.mov", "Субтитры создавал DimaTorzok")])
    result = run_index_quality(db)
    assert len(result.whole_chunk) == 1
    assert result.tails == []


def test_a_sign_off_glued_to_speech_is_found_separately(tmp_path: Path) -> None:
    db = _db(tmp_path, [
        ("transcripts", "a.mov", "まって、まってご視聴ありがとうございました"),
    ])
    result = run_index_quality(db)
    assert len(result.tails) == 1
    assert result.whole_chunk == []
    assert result.tails[0].after == "まって、まって"


def test_real_speech_is_never_reported(tmp_path: Path) -> None:
    # The phrases that destroyed real content when matched too eagerly. A scan
    # that cries wolf on these would be worse than no scan.
    db = _db(tmp_path, [
        ("transcripts", "a.mov",
         "So speaking of budget, thank you very much for coming today."),
        ("transcripts", "b.mov", "Takk for ating medieting."),
        ("transcripts", "c.mov",
         "Субтитры создавал DimaTorzok 山が正面 見えてる?見えてる!"),
    ])
    result = run_index_quality(db)
    assert result.clean, [f.before for f in result.tails + result.whole_chunk]


def test_counts_are_exact_even_when_samples_are_capped(tmp_path: Path) -> None:
    # The point is a number you can act on, so the cap must limit only the
    # examples RETAINED, never the count.
    rows = [
        ("transcripts", f"{i}.mov", f"話しています{i}ご視聴ありがとうございました")
        for i in range(30)
    ]
    result = run_index_quality(_db(tmp_path, rows), sample_per_kind=3)
    assert len(result.tails) == 30
    assert sum(1 for f in result.tails if f.before) == 3
    assert len(result.documents_affected) == 30


def test_source_types_filter_limits_the_scan(tmp_path: Path) -> None:
    db = _db(tmp_path, [
        ("transcripts", "a.mov", "はいはいご視聴ありがとうございました"),
        ("notes", "b.md", "はいはいご視聴ありがとうございました"),
    ])
    assert run_index_quality(db).affected_chunks == 2
    scoped = run_index_quality(db, source_types=("transcripts",))
    assert scoped.affected_chunks == 1
    assert scoped.scanned_chunks == 1


def test_documents_are_counted_not_chunks(tmp_path: Path) -> None:
    # One recording producing several bad windows is ONE affected document;
    # conflating the two overstates how much of an archive is involved.
    rows = [
        ("transcripts", "same.mov", f"話{i}ご視聴ありがとうございました")
        for i in range(5)
    ]
    result = run_index_quality(_db(tmp_path, rows))
    assert result.affected_chunks == 5
    assert len(result.documents_affected) == 1


def test_blank_chunks_are_skipped_not_flagged(tmp_path: Path) -> None:
    db = _db(tmp_path, [("transcripts", "a.mov", "   ")])
    result = run_index_quality(db)
    assert result.clean
    assert result.scanned_chunks == 1


def test_a_database_that_is_not_an_index_says_so(tmp_path: Path) -> None:
    """A consumer repo holds several SQLite files and only one is the index.

    Pointing at a caption cache or a sidecar should report that plainly, not
    raise OperationalError out of the middle of a scan.
    """
    import pytest

    path = tmp_path / "captions.db"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE captions (id TEXT, text TEXT)")
    conn.commit()
    conn.close()

    with pytest.raises(NotACorpusIndexError, match="chunks"):
        run_index_quality(path)


# --- a scan that examined nothing is not a clean scan ------------------------


def test_an_empty_index_is_not_reported_as_clean(tmp_path: Path) -> None:
    """The defect: `clean` was `affected_chunks == 0`, trivially true at zero.

    Scanning an empty index printed "no transcription artefacts" -- a clean
    bill of health having examined nothing. Pointed at the wrong database, or
    at a source type that does not exist, the scan reassured instead of
    saying it had no idea.
    """
    db = _db(tmp_path, [])
    result = run_index_quality(db)

    assert result.scanned_chunks == 0
    assert result.coverage.vacuous is True
    assert result.clean is False, "a vacuous scan proves nothing"


def test_a_scan_of_real_chunks_reports_its_coverage(tmp_path: Path) -> None:
    db = _db(tmp_path, [
        ("notes", "a.md", "we drove up to the lake and fed the ducks all afternoon"),
        ("notes", "b.md", "the quarterly numbers came in ahead of plan this time"),
    ])
    result = run_index_quality(db)

    assert result.coverage.examined == 2
    assert result.coverage.describe() == "2 chunks"
    assert result.clean is True


def test_filtering_to_a_source_type_that_does_not_exist_is_vacuous(
    tmp_path: Path,
) -> None:
    # The realistic way to get a false clean: a typo in --source-type, or a
    # source that was renamed. The index is full; the scan still saw nothing.
    db = _db(tmp_path, [("notes", "a.md", "real text that is long enough to keep")])
    result = run_index_quality(db, source_types=("transcrpits",))

    assert result.coverage.vacuous is True
    assert result.clean is False
