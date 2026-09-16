"""Tests for the transcript connector.

Each case is a decision that was measured on a real 7,400-recording archive,
and several of them protect against a change that was tried and reverted.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from corpus.connectors.transcripts import (
    MIN_SEGMENT_CHARS,
    TranscriptChunker,
    TranscriptConnector,
    hms,
    worth_indexing,
)
from corpus.transcripts.store import SCHEMA


def _db(tmp_path: Path, rows: list[dict]) -> Path:
    path = tmp_path / "transcripts.db"
    conn = sqlite3.connect(path)
    conn.executescript(SCHEMA)
    for r in rows:
        segments = r["segments"]
        conn.execute(
            "INSERT INTO transcripts (path, duration_s, dropped_windows, text,"
            " languages, segments, model, policy, transcribed_at, elapsed_s)"
            " VALUES (?,?,?,?,?,?,?,?,?,?)",
            (
                r["path"], r.get("duration_s", 60.0), 0, r.get("text", "x"),
                json.dumps([s.get("lang") for s in segments]),
                json.dumps(segments), "m", "p", "2026-01-01T00:00:00Z", 1.0,
            ),
        )
    conn.commit()
    conn.close()
    return path


def _seg(start, end, text, lang="en"):
    return {"start": start, "end": end, "text": text, "lang": lang}


LONG_EN = "we went to the park and fed the ducks by the pond for an hour"


def _chunks(tmp_path, segments, text="something"):
    db = _db(tmp_path, [{"path": "/a.mov", "text": text, "segments": segments}])
    doc = next(iter(TranscriptConnector("transcripts", db).load()))
    return TranscriptChunker("transcripts").chunk(doc)


# --- one chunk per window, and NO merging ----------------------------------
# Merging adjacent windows was tried because these chunks average ~60 tokens
# against ~400 for document sources. It was much worse -- recall 0.625 ->
# 0.375 -- because the unit of interest in a recording is a MOMENT, not a
# topical section.


def test_each_window_becomes_its_own_chunk(tmp_path: Path) -> None:
    chunks = _chunks(tmp_path, [_seg(0, 30, LONG_EN), _seg(30, 60, LONG_EN)])
    assert len(chunks) == 2


def test_a_chunk_carries_its_language_and_timestamp(tmp_path: Path) -> None:
    chunks = _chunks(tmp_path, [_seg(3725.0, 3755.0, LONG_EN, "ja")])
    extra = chunks[0].metadata.extra
    assert extra["language"] == "ja"
    assert extra["start_s"] == 3725.0
    assert extra["timestamp"] == "1:02:05"


# --- the language-aware floor ----------------------------------------------
# A flat character count discriminates against CJK, on an archive that is
# roughly half Japanese.


def test_a_short_japanese_sentence_is_kept() -> None:
    assert worth_indexing("誕生日おめでとうございます、みなさん。")


def test_the_same_length_in_english_is_not() -> None:
    assert not worth_indexing("a" * 19)
    assert len("a" * 19) < MIN_SEGMENT_CHARS


# --- a sign-off is cut before the window is judged -------------------------


def test_a_sign_off_glued_to_speech_is_removed_but_the_speech_is_kept(
    tmp_path: Path,
) -> None:
    chunks = _chunks(tmp_path, [_seg(0, 30, "まって、まってご視聴ありがとうございました", "ja")])
    assert len(chunks) == 1
    assert chunks[0].content == "まって、まって"


def test_a_window_that_is_only_boilerplate_is_dropped(tmp_path: Path) -> None:
    chunks = _chunks(
        tmp_path,
        [_seg(0, 30, LONG_EN), _seg(30, 60, "Terima kasih telah menonton")],
    )
    assert len(chunks) == 1


# --- the rescue: a short window that IS the recording ----------------------
# Measured on 271 files, 17 produced NO chunks and were not junk -- short
# voice memos whose entire transcript is one sentence, counted as documents
# and indexed as nothing.


def test_a_recording_that_is_one_short_sentence_is_still_indexed(
    tmp_path: Path,
) -> None:
    chunks = _chunks(tmp_path, [_seg(0, 12, "I love you.")], text="I love you.")
    assert len(chunks) == 1


def test_the_rescue_never_restores_a_boilerplate_window(tmp_path: Path) -> None:
    # It used to take the longest segment unconditionally, so a recording whose
    # every window was caption boilerplate had its sign-off indexed as the
    # document -- reintroducing exactly what the filter exists to remove.
    chunks = _chunks(
        tmp_path, [_seg(0, 30, "Thanks for watching")], text="Thanks for watching"
    )
    assert chunks == []


# --- connector-level behaviour ---------------------------------------------


def test_a_missing_database_raises_rather_than_yielding_nothing(
    tmp_path: Path,
) -> None:
    # An empty enumeration makes the ingester treat every already-indexed
    # transcript as an orphan and delete it.
    with pytest.raises(FileNotFoundError):
        TranscriptConnector("transcripts", tmp_path / "absent.db")


def test_an_unparseable_row_counts_as_a_failure_not_a_skip(tmp_path: Path) -> None:
    # Counting it as failed suppresses pruning, rather than letting the run
    # delete chunks it could not read.
    db = _db(tmp_path, [{"path": "/a.mov", "segments": [_seg(0, 30, LONG_EN)]}])
    conn = sqlite3.connect(db)
    conn.execute("UPDATE transcripts SET segments = 'not json'")
    conn.commit()
    conn.close()

    connector = TranscriptConnector("transcripts", db)
    assert list(connector.load()) == []
    assert connector.failed_files == 1
    assert connector.skipped_files == 0


def test_exclusions_are_re_applied_at_ingest(tmp_path: Path) -> None:
    # Rules tighten while a long transcription run is in flight, so the
    # database holds rows written under the older ones. Enumeration-time
    # filtering cannot retract those; this can.
    db = _db(tmp_path, [
        {"path": "/music/song^KARAOKE.mp3", "segments": [_seg(0, 30, LONG_EN)]},
        {"path": "/videos/real.mov", "segments": [_seg(0, 30, LONG_EN)]},
    ])
    connector = TranscriptConnector("transcripts", db, exclude=["KARAOKE"])
    keys = [d.source_key for d in connector.load()]
    assert keys == ["/videos/real.mov"]
    assert connector.excluded_files == 1


def test_a_row_with_empty_text_is_skipped_not_failed(tmp_path: Path) -> None:
    db = _db(tmp_path, [{"path": "/a.mov", "text": "  ", "segments": []}])
    connector = TranscriptConnector("transcripts", db)
    assert list(connector.load()) == []
    assert connector.skipped_files == 1
    assert connector.failed_files == 0


def test_hms_renders_a_position_a_person_can_use() -> None:
    assert hms(0) == "0:00"
    assert hms(125) == "2:05"
    assert hms(3725) == "1:02:05"
    assert hms(None) is None


def test_a_looping_window_inside_a_real_recording_is_not_indexed() -> None:
    """Measured: 716 loop chunks in a live index, only 553 from bad transcripts.

    The other 163 were single looping windows inside GENUINE recordings, so a
    whole-transcript judgement could never reach them. Same argument as the
    boilerplate check beside it, same place to make it.
    """
    assert not worth_indexing("I am going to draw a small map. " * 9)


def test_a_real_window_that_repeats_a_little_is_still_indexed() -> None:
    assert worth_indexing(
        "we drove up to the lake and the kids fed the ducks all afternoon, and "
        "then the rain started and we had to run back to the car"
    )
    # Short and repetitive is a person, not a loop -- and below the unit floor.
    assert worth_indexing("Papa! Papa! Papa! Papa! Papa! Papa! Papa!")
