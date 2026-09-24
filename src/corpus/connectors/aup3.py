"""Audacity 3 project (`.aup3`) connector.

An `.aup3` is a SQLite database holding a project's raw audio. Nothing in
corpus (or ffmpeg, or any general-purpose media tool) can read one today —
the audio inside is entirely invisible to search or transcription. This
module makes it reachable.

Verified structure (by hand, against a real project — not assumed):

  - Tables: `project`, `autosave`, `sampleblocks`, `sqlite_sequence`.
  - `sampleblocks` columns: `blockid, sampleformat, summin, summax, sumrms,
    summary256, summary64k, samples`.
  - `sampleformat` 262159 (0x0004000F) is Audacity's `floatSample` —
    `samples` is little-endian float32 in [-1, 1]. This is
    self-verifying: decoding a block and recomputing min/max/RMS matches the
    block's own stored `summin`/`summax`/`sumrms`. Both `load()` (once, on
    the first floatSample block found — a cheap per-file sanity check) and
    `extract_audio()` (on every block it writes, effectively free since it
    already decodes every sample to write it out) use that as an assertion:
    a mismatch means the format assumption doesn't hold for this data, and
    it's treated as unsupported rather than guessed at.
  - Blockids are NOT contiguous — deleting audio in the editor deletes rows,
    so a real project's blockids can span a much wider range than its block
    count. Every query here reads `ORDER BY blockid`, never assumes
    contiguity, and never assumes the block count equals `MAX(blockid)`.
  - `int16Sample` (0x00020001) and `int24Sample` (0x00040001) — Audacity's
    other on-disk sample formats — exist in the wild. This connector
    recognizes them by their format code but does not attempt to decode
    them: their on-disk byte layout inside `samples` was not independently
    verified the way floatSample's was, and guessing would risk silently
    misdecoding audio into noise. A block in one of these formats is
    reported, never decoded. See "What isn't supported" below.

The project's layout -- sample rate, tracks, clip positions, trims, mute --
lives in `project.doc`, Audacity's binary XML. `corpus.connectors.aup3_layout`
reads it, so the document reports the real duration and rate, and
`extract_audio` writes the audible tracks mixed to mono at the real rate.
A project whose layout cannot be read (no `project` row, or one that does
not decode) falls back to the old assumptions: 44100 Hz, one channel, every
block joined in id order. Those are overridable per source, and the document
says they are assumptions.

Transcription: `corpus-transcribe` treats an `.aup3` like any recording
(`corpus.transcripts.audio.decode` reads it through the layout), so what is
said in one becomes searchable through a `transcripts` source.

Why "connector + companion extraction API", not "extraction-only CLI
utility" or "connector that extracts automatically": an `.aup3` holds audio,
not text, so `AupThreeConnector.load()` alone can never produce indexable
content — but a lightweight per-project document (duration, block count,
sample-format verdict, source path) is genuinely useful on its own: it makes
a pile of unlabeled project files searchable/browsable ("which of these is
the 40-minute one from the trip") even before anything transcribes them.
That document costs nothing to produce — it needs only cheap SQL aggregates
(`COUNT`, `SUM(LENGTH(...))` per format), never a full sample decode — so
it's cheap enough to run on every `corpus-ingest`. Actually extracting audio
is a completely different cost profile: it means reading and writing the
entire sample data, potentially gigabytes per project. Doing that
automatically as a side effect of `load()` would make every ingest run
silently slow and disk-hungry the first time an archive of recordings is
pointed at corpus. Keeping extraction a separate, explicitly-called function
(`extract_audio`) means ingest stays fast and side-effect-free, and a person
(or a future transcription pipeline stage) decides when the heavier work
happens. No `[project.scripts]` CLI entry point is added for it in this
change — a concurrent change in this same checkout is already touching
`[project.scripts]`, so `extract_audio` ships as a plain, tested, importable
function for now; wiring a CLI around it is a trivial, separate follow-up.

Non-negotiable: the source `.aup3` is opened via a `mode=ro` SQLite URI
(never `sqlite3.connect(path)` directly) in both `load()` and
`extract_audio()`, and is never written, migrated, moved, or deleted. These
are personal recordings that cannot be regenerated if damaged.
`tests/test_aup3_connector.py` asserts the source file's mtime and size are
byte-for-byte unchanged after a real extraction run.

Handled explicitly (see the tests):
  - A corrupt or truncated database: `sqlite3.DatabaseError` opening or
    querying it is treated as `failed_files` (transient — could be
    mid-write or mid-copy), matching every other connector's "cannot open"
    convention.
  - A syntactically valid SQLite database that isn't an Audacity project at
    all (no `sampleblocks` table): `skipped_files` (permanent — retrying
    won't make it a project).
  - A project with zero sample blocks (valid, just empty — nothing was ever
    recorded/imported): `load()` yields a document saying so;
    `extract_audio` returns `None` (nothing to write, not an error).
  - A mixed-format project (some sampleblocks floatSample, some not):
    `load()`'s document reports the split; `extract_audio` refuses the
    WHOLE file with `UnsupportedSampleFormatError` rather than silently
    writing a shorter file with an unannounced gap where the unsupported
    blocks would have been.
"""

from __future__ import annotations

import array
import logging
import math
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import wave
from collections.abc import Iterable
from pathlib import Path

from corpus.connectors.aup3_layout import LayoutError, ProjectLayout, mix_to_mono, read_layout
from corpus.types import SourceDocument
from corpus.util.sqlite_ro import connect_ro

logger = logging.getLogger(__name__)

# Audacity's own SampleFormat.h enum values (public, from Audacity's
# open-source headers) — not reverse-engineered, just reused so this module
# recognizes them by name.
FLOAT_SAMPLE_FORMAT = 0x0004000F  # 262159 — the only format this connector decodes
INT16_SAMPLE_FORMAT = 0x00020001  # 131073 — recognized, not decoded (see module docstring)
INT24_SAMPLE_FORMAT = 0x00040001  # 262145 — recognized, not decoded (see module docstring)

_KNOWN_SAMPLE_FORMATS: dict[int, str] = {
    FLOAT_SAMPLE_FORMAT: "floatSample",
    INT16_SAMPLE_FORMAT: "int16Sample",
    INT24_SAMPLE_FORMAT: "int24Sample",
}

BYTES_PER_FLOAT_SAMPLE = 4

# Used only for a project whose layout cannot be read -- see the module
# docstring. Both are constructor / SourceConfig overrides.
DEFAULT_SAMPLE_RATE_HZ = 44100
DEFAULT_CHANNELS = 1

# Self-verification tolerance for recomputed min/max/RMS vs. a block's own
# stored summin/summax/sumrms. Not exact equality: Audacity's stored values
# are computed by its own (float32-precision, unknown summation order) code
# path, while this recomputes in Python double precision — close to but not
# guaranteed bit-identical even when the format assumption is entirely
# correct. A REAL mismatch (wrong format assumption, corrupted block) is
# orders of magnitude larger than this tolerance, so it doesn't mask genuine
# misdetection.
_VERIFY_REL_TOL = 1e-3
_VERIFY_ABS_TOL = 1e-5

# WAV output is 16-bit PCM, not 32-bit float — the one deliberately lossy
# step in this pipeline, chosen for maximum playback compatibility (every
# player/OS handles 16-bit PCM WAV; float WAV is common but not universal).
# 16-bit PCM still has ~96dB of dynamic range, far more than a voice/family
# recording needs, and more than downstream transcription needs (most ASR
# pipelines resample to 16kHz mono anyway). FLAC/MP3 output (via ffmpeg)
# transcodes from this same WAV, so they inherit this one quantization step,
# not an additional one.
_INT16_PEAK = 32767


class Aup3Error(Exception):
    """Base class for `.aup3`-specific extraction failures."""


class UnsupportedSampleFormatError(Aup3Error):
    """Raised by `extract_audio` when a project contains a sample block that
    isn't a verified floatSample block — a different Audacity sample format
    (int16Sample, int24Sample), or a floatSample block whose recomputed
    min/max/RMS doesn't match its own stored summary stats closely enough to
    trust. Raised for the WHOLE project, not just the offending block:
    silently dropping it and writing a shorter file would produce a WAV with
    an unannounced gap, which is worse than refusing outright and saying
    exactly which block and why."""


class FFmpegNotFoundError(Aup3Error):
    """Raised by `extract_audio` when `audio_format` is "flac"/"mp3" but no
    `ffmpeg` binary is on PATH. Never silently falls back to WAV — the
    caller asked for a specific format and gets a specific, actionable
    reason it can't have it."""



def _has_sampleblocks_table(conn: sqlite3.Connection) -> bool:
    rows = conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
    return "sampleblocks" in {str(row[0]) for row in rows}


def _format_breakdown(conn: sqlite3.Connection) -> dict[int, tuple[int, int]]:
    """`{sampleformat: (block_count, total_sample_bytes)}` for the whole
    project, via one cheap SQL aggregate — no per-row Python decoding, so
    this is safe to run on every `load()` regardless of project size."""
    rows = conn.execute(
        "SELECT sampleformat, COUNT(*), COALESCE(SUM(LENGTH(samples)), 0) "
        "FROM sampleblocks GROUP BY sampleformat"
    ).fetchall()
    return {int(fmt): (int(count), int(total_bytes)) for fmt, count, total_bytes in rows}


def _decode_float_samples(blob: bytes) -> array.array[float]:
    """`samples` blob (little-endian float32) -> an `array.array('f')`,
    byte-order-corrected for the (rare, but possible) big-endian host."""
    samples: array.array[float] = array.array("f")
    samples.frombytes(blob)
    if sys.byteorder != "little":
        samples.byteswap()
    return samples


def _block_stats(samples: array.array[float]) -> tuple[float, float, float]:
    if not samples:
        return (0.0, 0.0, 0.0)
    lo = min(samples)
    hi = max(samples)
    rms = math.sqrt(sum(float(s) * float(s) for s in samples) / len(samples))
    return (lo, hi, rms)


def _stats_match(
    lo: float, summin: float, hi: float, summax: float, rms: float, sumrms: float
) -> bool:
    return (
        math.isclose(lo, summin, rel_tol=_VERIFY_REL_TOL, abs_tol=_VERIFY_ABS_TOL)
        and math.isclose(hi, summax, rel_tol=_VERIFY_REL_TOL, abs_tol=_VERIFY_ABS_TOL)
        and math.isclose(rms, sumrms, rel_tol=_VERIFY_REL_TOL, abs_tol=_VERIFY_ABS_TOL)
    )


def _verify_block(summin: float, summax: float, sumrms: float, blob: bytes) -> bool:
    lo, hi, rms = _block_stats(_decode_float_samples(blob))
    return _stats_match(lo, summin, hi, summax, rms, sumrms)


def _format_name(fmt: int) -> str:
    return _KNOWN_SAMPLE_FORMATS.get(fmt, f"unknown format 0x{fmt:X}")


def _format_counts_text(counts: dict[int, int]) -> str:
    return ", ".join(
        f"{count} {_format_name(fmt)} block(s)" for fmt, count in sorted(counts.items())
    )


def _format_duration(seconds: float) -> str:
    total = max(round(seconds), 0)
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    parts = []
    if hours:
        parts.append(f"{hours}h")
    if hours or minutes:
        parts.append(f"{minutes}m")
    parts.append(f"{secs}s")
    return " ".join(parts)


def _describe_project(
    *,
    path: Path,
    breakdown: dict[int, tuple[int, int]],
    verified: bool,
    sample_rate: int,
    channels: int,
) -> str:
    total_blocks = sum(count for count, _ in breakdown.values())
    non_float = {fmt: count for fmt, (count, _) in breakdown.items() if fmt != FLOAT_SAMPLE_FORMAT}
    float_count, float_bytes = breakdown.get(FLOAT_SAMPLE_FORMAT, (0, 0))

    lines = [
        f"Audacity project: {path.name}",
        f"Source path: {path}",
        f"Sample blocks: {total_blocks}",
    ]

    if float_count == 0:
        lines.append(
            "Sample format: no floatSample blocks found "
            f"({_format_counts_text(non_float)}) — extraction is not "
            "supported for this project; see corpus.connectors.aup3's "
            "module docstring"
        )
        lines.append("Estimated duration: unknown (unsupported sample format)")
    elif not verified:
        lines.append(
            "Sample format: a floatSample block was found, but "
            "self-verification (recomputed min/max/RMS vs. the block's own "
            "stored summary stats) did not match closely enough to trust — "
            "treating this project as unsupported rather than guessing"
        )
        lines.append("Estimated duration: unknown (format unverified)")
    else:
        total_samples = float_bytes // BYTES_PER_FLOAT_SAMPLE
        duration = total_samples / (sample_rate * max(channels, 1))
        if non_float:
            lines.append(
                "Sample format: mixed — floatSample (verified, extractable) "
                f"plus {_format_counts_text(non_float)} (not supported by "
                "this connector's extraction path); duration below reflects "
                "the floatSample portion only"
            )
        else:
            lines.append(
                "Sample format: floatSample (verified against the block's "
                "own stored summary stats)"
            )
        lines.append(f"Estimated duration: {_format_duration(duration)} ({duration:.1f}s)")

    lines.append(
        f"Assumed sample rate: {sample_rate} Hz — this project has no readable "
        "layout; override via `sample_rate` on this source in corpus.toml if "
        "you know the real value"
    )
    lines.append(
        f"Assumed channels: {channels} — this project has no readable layout; "
        "override via `channels` on this source in corpus.toml if you know "
        "the real value"
    )
    lines.append(
        "Note: audio content itself is not indexed here — this document "
        "only describes the recording. Call "
        "corpus.connectors.aup3.extract_audio() to write a playable "
        "WAV/FLAC/MP3 file for transcription."
    )
    return "\n".join(lines)


def _describe_layout(*, path: Path, layout: ProjectLayout, total_blocks: int) -> str:
    audible = len(layout.audible_tracks())
    tracks = f"{len(layout.tracks)}" + (
        f" ({audible} audible)" if audible != len(layout.tracks) else ""
    )
    return "\n".join(
        [
            f"Audacity project: {path.name}",
            f"Source path: {path}",
            f"Duration: {_format_duration(layout.duration_s)} ({layout.duration_s:.1f}s)",
            f"Sample rate: {layout.rate} Hz",
            f"Tracks: {tracks}",
            f"Sample blocks: {total_blocks}",
            "Note: what is said in this recording is not in this document. "
            "Run corpus-transcribe over this folder to transcribe it; the "
            "transcript is indexed through a `transcripts` source.",
        ]
    )


def _describe_empty_project(path: Path) -> str:
    return "\n".join(
        [
            f"Audacity project: {path.name}",
            f"Source path: {path}",
            "Sample blocks: 0 (no recorded or imported audio in this project)",
        ]
    )


class AupThreeConnector:
    """Walks `path` for `.aup3` files and yields one metadata
    `SourceDocument` per project — never the audio itself. See the module
    docstring for why extraction is a separate, explicitly-called function
    (`extract_audio`) rather than something `load()` triggers."""

    def __init__(
        self,
        source_type: str,
        path: Path | str,
        glob: str = "**/*.aup3",
        sample_rate: int = DEFAULT_SAMPLE_RATE_HZ,
        channels: int = DEFAULT_CHANNELS,
    ):
        self.source_type = source_type
        self._root = Path(os.path.expanduser(str(path))).resolve()
        self._glob = glob
        self._sample_rate = sample_rate
        self._channels = channels
        self.failed_files = 0
        self.skipped_files = 0

    def load(self) -> Iterable[SourceDocument]:
        # Reset per run — see the identical note in every other connector:
        # a reused instance must not suppress orphan pruning forever on the
        # strength of an earlier run's failures.
        self.failed_files = 0
        self.skipped_files = 0

        from corpus.connectors.discovery import discover_files

        if not self._root.is_dir():
            raise FileNotFoundError(
                f"Aup3 source '{self.source_type}': directory not found: {self._root}"
            )

        for path in discover_files(self._root, self._glob):
            doc = self._load_one(path)
            if doc is not None:
                yield doc

    def _load_one(self, path: Path) -> SourceDocument | None:
        conn: sqlite3.Connection | None = None
        try:
            conn = connect_ro(path, immutable=True)
            return self._build_document(conn, path)
        except sqlite3.DatabaseError as e:
            logger.warning(
                "%s: cannot open '%s' as a SQLite database: %s", self.source_type, path, e
            )
            self.failed_files += 1
            return None
        finally:
            if conn is not None:
                conn.close()

    def _build_document(self, conn: sqlite3.Connection, path: Path) -> SourceDocument | None:
        if not _has_sampleblocks_table(conn):
            logger.info(
                "%s: skipping '%s' — no `sampleblocks` table; not an "
                "Audacity 3 project (permanent, not retried)",
                self.source_type,
                path.name,
            )
            self.skipped_files += 1
            return None

        source_key = str(path.relative_to(self._root))
        breakdown = _format_breakdown(conn)
        total_blocks = sum(count for count, _ in breakdown.values())

        if total_blocks == 0:
            return SourceDocument(
                source_type=self.source_type,
                source_key=source_key,
                title=path.stem,
                url=None,
                raw={"body": _describe_empty_project(path), "path": str(path)},
            )

        layout_error = None
        try:
            layout = read_layout(conn)
        except LayoutError as e:
            layout, layout_error = None, str(e)
            logger.warning("%s: layout of '%s' could not be read: %s", self.source_type, path, e)
        if layout is not None:
            return SourceDocument(
                source_type=self.source_type,
                source_key=source_key,
                title=path.stem,
                url=None,
                raw={
                    "body": _describe_layout(path=path, layout=layout, total_blocks=total_blocks),
                    "path": str(path),
                },
            )

        verified = False
        if FLOAT_SAMPLE_FORMAT in breakdown:
            row = conn.execute(
                "SELECT summin, summax, sumrms, samples FROM sampleblocks "
                "WHERE sampleformat = ? ORDER BY blockid LIMIT 1",
                (FLOAT_SAMPLE_FORMAT,),
            ).fetchone()
            if row is not None:
                summin, summax, sumrms, blob = row
                verified = _verify_block(float(summin), float(summax), float(sumrms), bytes(blob))

        body = _describe_project(
            path=path,
            breakdown=breakdown,
            verified=verified,
            sample_rate=self._sample_rate,
            channels=self._channels,
        )
        if layout_error is not None:
            body += (
                f"\nNote: the project's layout could not be read ({layout_error}), "
                "so the rate, channels and duration above are assumptions."
            )
        return SourceDocument(
            source_type=self.source_type,
            source_key=source_key,
            title=path.stem,
            url=None,
            raw={"body": body, "path": str(path)},
        )


# ---------------------------------------------------------------------------
# Extraction API — writes a real, playable audio file. Never called by
# `load()`; see the module docstring for why.
# ---------------------------------------------------------------------------


def _resolve_output_path(source: Path, output_path: Path | str | None, audio_format: str) -> Path:
    default_name = f"{source.stem}.{audio_format}"
    if output_path is None:
        # Adjacent to the source by default — the point is that a person
        # can find and play it.
        return source.with_name(default_name)
    out = Path(os.path.expanduser(str(output_path)))
    if out.is_dir():
        return out / default_name
    return out


def _quantize_int16(sample: float) -> int:
    clipped = -1.0 if sample < -1.0 else (1.0 if sample > 1.0 else sample)
    return round(clipped * _INT16_PEAK)


def _write_pcm16_frames(wf: wave.Wave_write, chunk: array.array[float]) -> None:
    ints: array.array[int] = array.array("h", (_quantize_int16(s) for s in chunk))
    wf.writeframesraw(ints.tobytes())


def _write_wav(path: Path, conn: sqlite3.Connection, sample_rate: int, channels: int) -> int:
    """Stream every floatSample block (blockid order) into a 16-bit PCM WAV
    at `path`. Verifies each block against its own stored summary stats as
    it decodes — effectively free, since decoding every sample is required
    anyway to write it out — and raises `UnsupportedSampleFormatError`
    immediately on a mismatch, before this function returns, so the caller
    (which writes to a temp path and only renames it into place afterward)
    never leaves a partial file at the final path.

    Returns the number of frames written. Any sample count not evenly
    divisible by `channels` has its trailing remainder dropped (logged), not
    padded — see the module docstring's channel-layout caveat.
    """
    frames_written = 0
    pending: array.array[float] = array.array("f")
    safe_channels = max(channels, 1)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(safe_channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        cursor = conn.execute(
            "SELECT blockid, summin, summax, sumrms, samples FROM sampleblocks "
            "WHERE sampleformat = ? ORDER BY blockid",
            (FLOAT_SAMPLE_FORMAT,),
        )
        for blockid, summin, summax, sumrms, blob in cursor:
            samples = _decode_float_samples(bytes(blob))
            lo, hi, rms = _block_stats(samples)
            if not _stats_match(lo, float(summin), hi, float(summax), rms, float(sumrms)):
                raise UnsupportedSampleFormatError(
                    f"block {blockid}: recomputed min/max/RMS "
                    f"({lo:.6f}/{hi:.6f}/{rms:.6f}) does not match this "
                    f"block's own stored summary stats "
                    f"({float(summin):.6f}/{float(summax):.6f}/{float(sumrms):.6f}) — "
                    "refusing to trust this block's decode"
                )
            pending.extend(samples)
            usable = (len(pending) // safe_channels) * safe_channels
            if usable:
                _write_pcm16_frames(wf, pending[:usable])
                frames_written += usable // safe_channels
                del pending[:usable]

    if pending:
        logger.warning(
            "%s: %d trailing sample(s) dropped — total sample count wasn't "
            "evenly divisible by channels=%d (see the module docstring's "
            "channel-layout caveat)",
            path,
            len(pending),
            channels,
        )
    return frames_written


def _write_mixed_wav(path: Path, conn: sqlite3.Connection, layout: ProjectLayout) -> int:
    """Write the project's audible tracks, mixed to mono at their real rate,
    as 16-bit PCM. Returns the number of frames written."""
    import numpy as np

    try:
        samples, rate = mix_to_mono(conn, layout)
    except LayoutError as e:
        raise UnsupportedSampleFormatError(str(e)) from e
    pcm = np.round(np.clip(samples, -1.0, 1.0) * _INT16_PEAK).astype("<i2")
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(pcm.tobytes())
    return int(pcm.size)


def _write_project_wav(
    path: Path, conn: sqlite3.Connection, sample_rate: int, channels: int
) -> int:
    """The project's own layout when it can be read; otherwise the flat,
    assumed-rate stream, which is all a project without one supports."""
    try:
        layout = read_layout(conn)
    except LayoutError as e:
        logger.warning("%s: layout could not be read (%s); using assumed rate/channels", path, e)
        layout = None
    if layout is not None:
        return _write_mixed_wav(path, conn, layout)
    return _write_wav(path, conn, sample_rate, channels)


def extract_audio(
    aup3_path: Path | str,
    output_path: Path | str | None = None,
    audio_format: str = "wav",
    sample_rate: int = DEFAULT_SAMPLE_RATE_HZ,
    channels: int = DEFAULT_CHANNELS,
    ffmpeg_path: str = "ffmpeg",
) -> Path | None:
    """Decode every floatSample sample block of `aup3_path`, in blockid
    order, and write it out as a standard, playable audio file — the
    companion to `AupThreeConnector`, whose own `load()` never writes
    anything (see the module docstring).

    The source is opened strictly read-only (`mode=ro`) and never written,
    moved, or deleted. Output is written adjacent to the source by default
    (`recording.aup3` -> `recording.wav`); `output_path` may instead name an
    exact file or an existing directory. The write is atomic — data is
    staged at a temp path and only `os.replace`d into the final target once
    fully written, so a crash mid-extraction cannot leave a truncated file
    that looks complete.

    With a readable layout the output is the audible tracks mixed to mono
    at the project's real rate (needs numpy); `sample_rate` and `channels`
    apply only to a project without one.

    `audio_format="wav"` needs nothing beyond the stdlib for a project
    without a layout. "flac"/"mp3"
    transcode from that same WAV via `ffmpeg` on PATH — if it's missing,
    raises `FFmpegNotFoundError` rather than silently falling back to WAV.

    Returns the path written, or `None` if the project has zero sample
    blocks (nothing to extract — not an error). Raises
    `UnsupportedSampleFormatError` if any block isn't a verified floatSample
    block (see the module docstring on mixed-format projects).
    """
    if audio_format not in ("wav", "flac", "mp3"):
        raise ValueError(f"audio_format must be 'wav', 'flac', or 'mp3' — got {audio_format!r}")

    source = Path(os.path.expanduser(str(aup3_path))).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"aup3 source not found: {source}")

    target = _resolve_output_path(source, output_path, audio_format)

    conn = connect_ro(source, immutable=True)
    try:
        if not _has_sampleblocks_table(conn):
            raise Aup3Error(f"{source}: no `sampleblocks` table — not an Audacity 3 project")

        breakdown = _format_breakdown(conn)
        total_blocks = sum(count for count, _ in breakdown.values())
        if total_blocks == 0:
            logger.info("%s: 0 sample blocks — nothing to extract", source)
            return None

        non_float = {
            fmt: count for fmt, (count, _) in breakdown.items() if fmt != FLOAT_SAMPLE_FORMAT
        }
        if non_float:
            raise UnsupportedSampleFormatError(
                f"{source}: {_format_counts_text(non_float)} — only "
                "floatSample is supported by this connector's extraction "
                "path (see corpus.connectors.aup3's module docstring); "
                "refusing to write a file with an unannounced gap where "
                "these blocks would have been"
            )

        if audio_format == "wav":
            wav_tmp = target.with_name(target.name + ".tmp")
            try:
                _write_project_wav(wav_tmp, conn, sample_rate, channels)
            except BaseException:
                wav_tmp.unlink(missing_ok=True)
                raise
            os.replace(wav_tmp, target)
            logger.info("%s: extracted audio to %s", source, target)
            return target

        if shutil.which(ffmpeg_path) is None:
            raise FFmpegNotFoundError(
                f"audio_format={audio_format!r} requires ffmpeg on PATH; "
                f"none found (looked for {ffmpeg_path!r}). Install ffmpeg, "
                "or call with audio_format='wav' (needs nothing beyond the "
                "standard library)."
            )

        fd, tmp_wav_name = tempfile.mkstemp(suffix=".wav", prefix="corpus-aup3-")
        os.close(fd)
        tmp_wav = Path(tmp_wav_name)
        final_tmp = target.with_name(target.name + ".tmp")
        try:
            _write_project_wav(tmp_wav, conn, sample_rate, channels)
            proc = subprocess.run(
                # `-f audio_format` names the muxer explicitly rather than
                # letting ffmpeg guess it from `final_tmp`'s extension --
                # that guess would see the intermediate `.tmp` suffix (the
                # atomic-write staging name, not the real target) and fail
                # to pick a format at all.
                [ffmpeg_path, "-y", "-i", str(tmp_wav), "-f", audio_format, str(final_tmp)],
                capture_output=True,
                text=True,
                check=False,
            )
            if proc.returncode != 0:
                raise Aup3Error(
                    f"ffmpeg failed to transcode {source} -> {target}: "
                    f"{proc.stderr.strip()[-2000:]}"
                )
            os.replace(final_tmp, target)
        finally:
            tmp_wav.unlink(missing_ok=True)
            final_tmp.unlink(missing_ok=True)

        logger.info("%s: extracted audio to %s (via ffmpeg)", source, target)
        return target
    finally:
        conn.close()
