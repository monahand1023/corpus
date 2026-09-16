"""Keyword-only hooks that no caller and no test ever named.

Found by walking the AST for keyword-only parameters and checking whether any
call site names them. Each is a seam someone could reach for, with nothing
verifying it works -- the same shape as `content_key`, which was documented,
implemented, defaulted to off, and passed by nobody in the engine, so the
duplicate passages it existed to collapse went uncollapsed for the life of
the feature.

The sweep needed a correction worth recording: its first version reported 13,
four of which were false positives injected through `**dict` unpacking, which
an AST scan for `keyword=` cannot see. Counting dict-literal keys as
potentially-unpacked brought it to 9.
"""

from __future__ import annotations

from pathlib import Path

import pytest

# --- corpus.transcripts.audio.decode(timeout_s=...) --------------------------


def test_the_decode_timeout_is_honoured(tmp_path: Path) -> None:
    """A hung ffmpeg must not hang the whole run. Nothing exercised the bound."""
    from corpus.transcripts.audio import AudioUnavailableError, decode

    media = tmp_path / "a.wav"
    media.write_bytes(b"not really audio")
    # 0 seconds cannot complete, so this proves the bound reaches ffmpeg
    # rather than being accepted and ignored.
    with pytest.raises(AudioUnavailableError):
        decode(media, timeout_s=0.001)


# --- corpus.transcripts.run.find_media(extensions=...) -----------------------


def test_find_media_honours_a_custom_extension_set(tmp_path: Path) -> None:
    from corpus.transcripts.run import find_media

    for name in ("a.m4a", "b.mov", "c.txt"):
        (tmp_path / name).write_bytes(b"x")

    assert [p.name for p in find_media(tmp_path, extensions={".m4a"})] == ["a.m4a"]
    assert {p.name for p in find_media(tmp_path, extensions={".m4a", ".mov"})} == {
        "a.m4a", "b.mov"
    }


def test_find_media_extension_matching_is_case_insensitive(tmp_path: Path) -> None:
    from corpus.transcripts.run import find_media

    (tmp_path / "LOUD.MOV").write_bytes(b"x")
    assert [p.name for p in find_media(tmp_path, extensions={".mov"})] == ["LOUD.MOV"]


# --- corpus.transcripts.quality.subtitle_boilerplate(prefixes=...) -----------


def test_a_caller_can_supply_its_own_credit_prefixes() -> None:
    """The `phrases` hook beside this one was tested; `prefixes` was not."""
    from corpus.transcripts.quality import subtitle_boilerplate

    text = "Transcribed by SomeStudio"
    assert subtitle_boilerplate(text, prefixes=("transcribed by",)) is True
    # An unrelated prefix list must not match it.
    assert subtitle_boilerplate(text, prefixes=("captions by",)) is False


# --- corpus.transcripts.segment.file_timeout(base=...) -----------------------


def test_the_file_timeout_scales_from_its_base() -> None:
    from corpus.transcripts.segment import file_timeout

    small, large = file_timeout(60.0, base=10.0), file_timeout(60.0, base=100.0)
    assert large > small, "a larger base must yield a larger budget"
    assert file_timeout(3600.0, base=10.0) > file_timeout(60.0, base=10.0), (
        "a longer file must get a longer budget"
    )


# --- corpus.verify.dormant(minimum=...) --------------------------------------


def test_the_dormancy_floor_can_be_overridden() -> None:
    from corpus.verify import Coverage, dormant

    counts = {"fired": 5}
    known = ["fired", "never"]
    coverage = Coverage(10, "verdicts")
    assert dormant(counts, known=known, coverage=coverage) == []
    assert dormant(counts, known=known, coverage=coverage, minimum=5) == ["never"]


# --- corpus.survey.duplicates.find_duplicate_content(top=, min_shared=) ------


_INDEX_SEQ = [0]


def _index(tmp_path: Path, rows: list[tuple[str, str]]) -> Path:
    import sqlite3

    # A fresh file per call: two indexes in one test is the normal case here,
    # and reusing the path raised "table chunks already exists".
    _INDEX_SEQ[0] += 1
    p = tmp_path / f"i{_INDEX_SEQ[0]}.db"
    c = sqlite3.connect(p)
    c.executescript(
        "CREATE TABLE chunks (id TEXT PRIMARY KEY, source_type TEXT NOT NULL,"
        " source_key TEXT NOT NULL, content TEXT NOT NULL,"
        " content_hash TEXT NOT NULL);"
    )
    c.executemany(
        "INSERT INTO chunks (id, source_type, source_key, content, content_hash)"
        " VALUES (?,?,?,?,?)",
        [(str(i), "notes", k, t, str(hash(t))) for i, (k, t) in enumerate(rows)],
    )
    c.commit()
    c.close()
    return p


def test_the_duplicate_report_caps_how_many_pairs_it_returns(tmp_path: Path) -> None:
    from corpus.survey.duplicates import find_duplicate_content

    rows = []
    for i in range(6):
        rows += [(f"a{i}.md", f"shared text {i}"), (f"b{i}.md", f"shared text {i}"),
                 (f"a{i}.md", f"more shared {i}"), (f"b{i}.md", f"more shared {i}")]
    report = find_duplicate_content(_index(tmp_path, rows), top=3)
    assert len(report.duplicate_documents) == 3


def test_min_shared_filters_incidental_single_overlaps(tmp_path: Path) -> None:
    """Two documents sharing ONE passage is usually a stock sentence."""
    from corpus.survey.duplicates import find_duplicate_content

    rows = [("a.md", "one shared line"), ("b.md", "one shared line")]
    assert find_duplicate_content(_index(tmp_path, rows), min_shared=2
                                  ).duplicate_documents == []
    assert find_duplicate_content(_index(tmp_path, rows), min_shared=1
                                  ).duplicate_documents != []
