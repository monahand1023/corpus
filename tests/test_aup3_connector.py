"""Tests for the Audacity `.aup3` project connector and its companion
`extract_audio` API.

All fixture `.aup3` files are built programmatically with `sqlite3` in
`tmp_path` — no real recordings, per the project's public-repo constraints.
The schema mirrors what was hand-verified against a real Audacity 3 project
(see `corpus/connectors/aup3.py`'s module docstring): `project`, `autosave`,
and `sampleblocks` tables, with `sampleblocks` holding
`blockid, sampleformat, summin, summax, sumrms, summary256, summary64k,
samples`.
"""

from __future__ import annotations

import math
import shutil
import sqlite3
import struct
import wave
from pathlib import Path

import pytest

import corpus.connectors.aup3 as aup3
from corpus.connectors.aup3 import (
    DEFAULT_CHANNELS,
    DEFAULT_SAMPLE_RATE_HZ,
    FLOAT_SAMPLE_FORMAT,
    INT16_SAMPLE_FORMAT,
    Aup3Error,
    AupThreeConnector,
    FFmpegNotFoundError,
    UnsupportedSampleFormatError,
    extract_audio,
)
from corpus.connectors.registry import CONNECTOR_REGISTRY, DEFAULT_GLOBS


def _sine_samples(n: int, amplitude: float = 0.5, freq: float = 440.0, rate: int = 44100) -> list[float]:
    return [amplitude * math.sin(2 * math.pi * freq * i / rate) for i in range(n)]


def _block_stats(values: list[float]) -> tuple[float, float, float]:
    lo = min(values)
    hi = max(values)
    rms = math.sqrt(sum(v * v for v in values) / len(values))
    return lo, hi, rms


def _float_blob(values: list[float]) -> bytes:
    return struct.pack(f"<{len(values)}f", *values)


def _create_schema(conn: sqlite3.Connection) -> None:
    conn.execute("CREATE TABLE project (id INTEGER PRIMARY KEY, dict BLOB, doc BLOB)")
    conn.execute("CREATE TABLE autosave (id INTEGER PRIMARY KEY, dict BLOB, doc BLOB)")
    conn.execute(
        "CREATE TABLE sampleblocks ("
        "blockid INTEGER PRIMARY KEY AUTOINCREMENT, sampleformat INTEGER, "
        "summin REAL, summax REAL, sumrms REAL, "
        "summary256 BLOB, summary64k BLOB, samples BLOB)"
    )


def _make_aup3(
    path: Path,
    float_blocks: list[tuple[int, list[float]]] | None = None,
    int_blocks: list[tuple[int, int]] | None = None,  # (blockid, sample_count) -> int16Sample junk
    corrupt_block_index: int | None = None,
) -> Path:
    """Build a fixture `.aup3`. `float_blocks` are (blockid, samples) pairs
    written as verified floatSample blocks (correct summin/summax/sumrms),
    except the one at `corrupt_block_index` (into `float_blocks`), whose
    stored `summin` is deliberately wrong. `int_blocks` are (blockid,
    byte_count) pairs written as int16Sample junk — content never matters,
    since this connector never decodes non-float blocks."""
    conn = sqlite3.connect(path)
    _create_schema(conn)
    if float_blocks:
        for i, (blockid, values) in enumerate(float_blocks):
            lo, hi, rms = _block_stats(values)
            if corrupt_block_index is not None and i == corrupt_block_index:
                lo += 5.0  # deliberately wrong -- fails self-verification
            conn.execute(
                "INSERT INTO sampleblocks "
                "(blockid, sampleformat, summin, summax, sumrms, summary256, summary64k, samples) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (blockid, FLOAT_SAMPLE_FORMAT, lo, hi, rms, b"", b"", _float_blob(values)),
            )
    if int_blocks:
        for blockid, count in int_blocks:
            conn.execute(
                "INSERT INTO sampleblocks "
                "(blockid, sampleformat, summin, summax, sumrms, summary256, summary64k, samples) "
                "VALUES (?, ?, 0, 0, 0, ?, ?, ?)",
                (blockid, INT16_SAMPLE_FORMAT, b"", b"", b"\x00\x00" * count),
            )
    conn.commit()
    conn.close()
    return path


def _empty_aup3(path: Path) -> Path:
    conn = sqlite3.connect(path)
    _create_schema(conn)
    conn.commit()
    conn.close()
    return path


def _wrong_schema_db(path: Path) -> Path:
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE not_audacity (x INTEGER)")
    conn.commit()
    conn.close()
    return path


# ---------------------------------------------------------------------------
# Registry wiring
# ---------------------------------------------------------------------------


def test_aup3_is_registered() -> None:
    assert "aup3" in DEFAULT_GLOBS
    assert DEFAULT_GLOBS["aup3"] == "**/*.aup3"
    assert "aup3" in CONNECTOR_REGISTRY


def test_build_aup3(tmp_path: Path) -> None:
    from corpus.config import SourceConfig

    cfg = SourceConfig(name="recordings", type="aup3", path=str(tmp_path))
    connector, _chunker = CONNECTOR_REGISTRY["aup3"](cfg)
    assert isinstance(connector, AupThreeConnector)
    assert connector._sample_rate == DEFAULT_SAMPLE_RATE_HZ
    assert connector._channels == DEFAULT_CHANNELS


def test_build_aup3_honors_sample_rate_and_channel_overrides(tmp_path: Path) -> None:
    from corpus.config import SourceConfig

    cfg = SourceConfig(
        name="recordings", type="aup3", path=str(tmp_path), sample_rate=48000, channels=2
    )
    connector, _chunker = CONNECTOR_REGISTRY["aup3"](cfg)
    assert connector._sample_rate == 48000
    assert connector._channels == 2


# ---------------------------------------------------------------------------
# load() -- metadata document
# ---------------------------------------------------------------------------


def test_verified_float_project_reports_duration_and_assumptions(tmp_path: Path) -> None:
    samples = _sine_samples(4410)  # exactly 0.1s at the default 44100 Hz
    _make_aup3(tmp_path / "trip.aup3", float_blocks=[(1, samples)])

    docs = list(AupThreeConnector(source_type="recordings", path=tmp_path).load())
    assert len(docs) == 1
    doc = docs[0]
    assert doc.title == "trip"
    assert doc.source_key == "trip.aup3"
    body = doc.raw["body"]
    assert "Sample blocks: 1" in body
    assert "floatSample (verified" in body
    assert "Estimated duration: 0s (0.1s)" in body
    assert "Assumed sample rate: 44100 Hz" in body
    assert "Assumed channels: 1" in body
    assert "extract_audio()" in body


def test_blockids_need_not_be_contiguous(tmp_path: Path) -> None:
    """Real projects have gaps in blockid (deleted blocks) -- block COUNT,
    not MAX(blockid), must drive duration."""
    samples_a = _sine_samples(100)
    samples_b = _sine_samples(100)
    _make_aup3(tmp_path / "edited.aup3", float_blocks=[(1, samples_a), (922, samples_b)])

    docs = list(AupThreeConnector(source_type="recordings", path=tmp_path).load())
    assert "Sample blocks: 2" in docs[0].raw["body"]


def test_zero_sampleblocks_project_is_reported_not_dropped(tmp_path: Path) -> None:
    _empty_aup3(tmp_path / "blank.aup3")
    docs = list(AupThreeConnector(source_type="recordings", path=tmp_path).load())
    assert len(docs) == 1
    assert "Sample blocks: 0" in docs[0].raw["body"]


def test_corrupt_database_counts_as_failed(tmp_path: Path) -> None:
    (tmp_path / "damaged.aup3").write_bytes(b"not a sqlite database, just garbage bytes" * 4)
    conn = AupThreeConnector(source_type="recordings", path=tmp_path)
    docs = list(conn.load())
    assert docs == []
    assert conn.failed_files == 1
    assert conn.skipped_files == 0


def test_wrong_schema_database_counts_as_skipped_not_failed(tmp_path: Path) -> None:
    _wrong_schema_db(tmp_path / "unrelated.aup3")
    conn = AupThreeConnector(source_type="recordings", path=tmp_path)
    docs = list(conn.load())
    assert docs == []
    assert conn.skipped_files == 1
    assert conn.failed_files == 0


def test_mixed_format_project_reports_partial_extractability(tmp_path: Path) -> None:
    float_samples = _sine_samples(4410)
    _make_aup3(
        tmp_path / "mixed.aup3",
        float_blocks=[(1, float_samples)],
        int_blocks=[(2, 100)],
    )
    docs = list(AupThreeConnector(source_type="recordings", path=tmp_path).load())
    body = docs[0].raw["body"]
    assert "Sample blocks: 2" in body
    assert "mixed" in body
    assert "1 int16Sample block(s)" in body


def test_int_only_project_reports_unsupported_format(tmp_path: Path) -> None:
    _make_aup3(tmp_path / "legacy.aup3", int_blocks=[(1, 100), (2, 100)])
    docs = list(AupThreeConnector(source_type="recordings", path=tmp_path).load())
    body = docs[0].raw["body"]
    assert "Sample blocks: 2" in body
    assert "no floatSample blocks found" in body
    assert "2 int16Sample block(s)" in body
    assert "Estimated duration: unknown" in body


def test_verification_mismatch_is_reported_as_unverified(tmp_path: Path) -> None:
    samples = _sine_samples(4410)
    _make_aup3(tmp_path / "suspect.aup3", float_blocks=[(1, samples)], corrupt_block_index=0)

    docs = list(AupThreeConnector(source_type="recordings", path=tmp_path).load())
    body = docs[0].raw["body"]
    assert "did not match closely enough to trust" in body
    assert "Estimated duration: unknown (format unverified)" in body


def test_missing_directory_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        list(AupThreeConnector(source_type="recordings", path=tmp_path / "nope").load())


def test_failed_and_skipped_reset_between_runs(tmp_path: Path) -> None:
    (tmp_path / "damaged.aup3").write_bytes(b"garbage")
    conn = AupThreeConnector(source_type="recordings", path=tmp_path)
    list(conn.load())
    assert conn.failed_files == 1
    list(conn.load())
    assert conn.failed_files == 1, "must reset, not accumulate, across runs"


# ---------------------------------------------------------------------------
# extract_audio() -- the actual audio, and the read-only guarantee
# ---------------------------------------------------------------------------


def test_extract_audio_writes_wav_adjacent_to_source_by_default(tmp_path: Path) -> None:
    n = 4410  # 0.1s at 44100 Hz
    samples = _sine_samples(n)
    source = _make_aup3(tmp_path / "family.aup3", float_blocks=[(1, samples)])

    out = extract_audio(source)

    assert out == tmp_path / "family.wav"
    assert out.exists()
    with wave.open(str(out), "rb") as wf:
        assert wf.getnchannels() == 1
        assert wf.getframerate() == DEFAULT_SAMPLE_RATE_HZ
        assert wf.getsampwidth() == 2
        assert wf.getnframes() == n
        raw = wf.readframes(n)
    got = struct.unpack(f"<{n}h", raw)
    expected = [round(max(-1.0, min(1.0, s)) * 32767) for s in samples]
    assert got == tuple(expected)


def test_extract_audio_multiple_blocks_concatenated_in_blockid_order(tmp_path: Path) -> None:
    a = _sine_samples(1000, freq=220.0)
    b = _sine_samples(1000, freq=880.0)
    # Insert out of blockid order to prove ORDER BY blockid, not insertion order, wins.
    source = _make_aup3(tmp_path / "two_blocks.aup3", float_blocks=[(50, b), (1, a)])

    out = extract_audio(source)
    with wave.open(str(out), "rb") as wf:
        assert wf.getnframes() == 2000
        raw = wf.readframes(2000)
    got = struct.unpack("<2000h", raw)
    expected_a = [round(max(-1.0, min(1.0, s)) * 32767) for s in a]
    expected_b = [round(max(-1.0, min(1.0, s)) * 32767) for s in b]
    assert list(got) == expected_a + expected_b


def test_extract_audio_never_modifies_the_source(tmp_path: Path) -> None:
    samples = _sine_samples(4410)
    source = _make_aup3(tmp_path / "irreplaceable.aup3", float_blocks=[(1, samples)])
    before_stat = source.stat()
    before_bytes = source.read_bytes()

    extract_audio(source)

    after_stat = source.stat()
    assert after_stat.st_mtime == before_stat.st_mtime
    assert after_stat.st_size == before_stat.st_size
    assert source.read_bytes() == before_bytes
    # No -wal/-shm/journal companion files must appear either.
    siblings = {p.name for p in tmp_path.iterdir()}
    assert siblings == {"irreplaceable.aup3", "irreplaceable.wav"}


def test_extract_audio_zero_blocks_returns_none_and_writes_nothing(tmp_path: Path) -> None:
    source = _empty_aup3(tmp_path / "blank.aup3")
    result = extract_audio(source)
    assert result is None
    assert list(tmp_path.iterdir()) == [source]


def test_extract_audio_mixed_format_raises_and_writes_nothing(tmp_path: Path) -> None:
    samples = _sine_samples(100)
    source = _make_aup3(
        tmp_path / "mixed.aup3", float_blocks=[(1, samples)], int_blocks=[(2, 50)]
    )
    with pytest.raises(UnsupportedSampleFormatError, match="int16Sample"):
        extract_audio(source)
    assert list(tmp_path.iterdir()) == [source], "no partial/gapped file may be left behind"


def test_extract_audio_int_only_project_raises(tmp_path: Path) -> None:
    source = _make_aup3(tmp_path / "legacy.aup3", int_blocks=[(1, 50)])
    with pytest.raises(UnsupportedSampleFormatError):
        extract_audio(source)


def test_extract_audio_verification_mismatch_raises(tmp_path: Path) -> None:
    samples = _sine_samples(4410)
    source = _make_aup3(
        tmp_path / "suspect.aup3", float_blocks=[(1, samples)], corrupt_block_index=0
    )
    with pytest.raises(UnsupportedSampleFormatError, match="block 1"):
        extract_audio(source)
    assert list(tmp_path.iterdir()) == [source]


def test_extract_audio_missing_source_raises_file_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        extract_audio(tmp_path / "nope.aup3")


def test_extract_audio_corrupt_database_raises(tmp_path: Path) -> None:
    source = tmp_path / "damaged.aup3"
    source.write_bytes(b"not a database")
    with pytest.raises(sqlite3.DatabaseError):
        extract_audio(source)


def test_extract_audio_output_path_directory_override(tmp_path: Path) -> None:
    source = _make_aup3(tmp_path / "orig.aup3", float_blocks=[(1, _sine_samples(100))])
    out_dir = tmp_path / "extracted"
    out_dir.mkdir()

    out = extract_audio(source, output_path=out_dir)
    assert out == out_dir / "orig.wav"
    assert out.exists()


def test_extract_audio_output_path_exact_file_override(tmp_path: Path) -> None:
    source = _make_aup3(tmp_path / "orig.aup3", float_blocks=[(1, _sine_samples(100))])
    out_file = tmp_path / "renamed.wav"

    out = extract_audio(source, output_path=out_file)
    assert out == out_file
    assert out.exists()


def test_extract_audio_invalid_format_raises_value_error(tmp_path: Path) -> None:
    source = _make_aup3(tmp_path / "orig.aup3", float_blocks=[(1, _sine_samples(100))])
    with pytest.raises(ValueError, match="audio_format"):
        extract_audio(source, audio_format="ogg")


def test_extract_audio_channel_override_drops_uneven_tail(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    samples = _sine_samples(101)  # not divisible by 2
    source = _make_aup3(tmp_path / "stereo_guess.aup3", float_blocks=[(1, samples)])

    with caplog.at_level("WARNING"):
        out = extract_audio(source, channels=2)

    with wave.open(str(out), "rb") as wf:
        assert wf.getnchannels() == 2
        assert wf.getnframes() == 50  # 101 // 2, tail dropped
    assert "trailing sample" in caplog.text


# ---------------------------------------------------------------------------
# ffmpeg-backed formats
# ---------------------------------------------------------------------------


def test_extract_audio_flac_without_ffmpeg_raises_clear_error(tmp_path: Path) -> None:
    source = _make_aup3(tmp_path / "orig.aup3", float_blocks=[(1, _sine_samples(1000))])
    with pytest.raises(FFmpegNotFoundError):
        extract_audio(source, audio_format="flac", ffmpeg_path="/definitely/not/a/real/binary")
    assert list(tmp_path.iterdir()) == [source]


def test_extract_audio_flac_missing_ffmpeg_via_monkeypatched_which(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(aup3.shutil, "which", lambda _: None)
    source = _make_aup3(tmp_path / "orig.aup3", float_blocks=[(1, _sine_samples(1000))])
    with pytest.raises(FFmpegNotFoundError):
        extract_audio(source, audio_format="mp3")


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="requires a real ffmpeg on PATH")
def test_extract_audio_flac_with_real_ffmpeg(tmp_path: Path) -> None:
    samples = _sine_samples(44100)  # 1.0s
    source = _make_aup3(tmp_path / "family.aup3", float_blocks=[(1, samples)])

    out = extract_audio(source, audio_format="flac")

    assert out == tmp_path / "family.flac"
    assert out.exists()
    assert out.stat().st_size > 0
    # No leftover temp/intermediate files.
    siblings = {p.name for p in tmp_path.iterdir()}
    assert siblings == {"family.aup3", "family.flac"}


def test_aup3_error_hierarchy() -> None:
    assert issubclass(UnsupportedSampleFormatError, Aup3Error)
    assert issubclass(FFmpegNotFoundError, Aup3Error)
