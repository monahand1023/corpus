"""Media duration survey: hours are the planning number, not file count.

A directory of 4,000 voice memos and a directory of 40 lecture recordings
can have the same file count and wildly different transcription cost —
count alone is useless for planning. `ffprobe`-ing every audio/video file in
a large tree is too slow to run casually, so this samples per extension
(reservoir sampling — see `corpus.survey.sampling` — so the full file list
is never held in memory) and extrapolates: sample mean duration for that
type, multiplied by that type's full file count. The result is explicit
about being an extrapolation (sample size vs. population, per type) rather
than presenting a bare number that looks like a census.

Degrades gracefully without `ffmpeg` installed: file counts and sizes are
still reported; duration is reported as unavailable, not guessed at.
"""

from __future__ import annotations

import os
import random
import shutil
import sqlite3
import statistics
import subprocess
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from corpus.survey.sampling import ReservoirSampler
from corpus.survey.walk import WalkStats, walk_files

AUDIO_EXTENSIONS: frozenset[str] = frozenset(
    {".mp3", ".wav", ".flac", ".aac", ".ogg", ".wma", ".m4a", ".aiff", ".aif", ".opus",
     # An Audacity project: a SQLite database, not a stream ffmpeg reads.
     # `_probe_duration_seconds` and `transcripts.audio.decode` read it directly.
     ".aup3"}
)
VIDEO_EXTENSIONS: frozenset[str] = frozenset(
    {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v", ".wmv", ".flv", ".mpg", ".mpeg", ".3gp"}
)
MEDIA_EXTENSIONS: frozenset[str] = AUDIO_EXTENSIONS | VIDEO_EXTENSIONS

DEFAULT_SAMPLE_SIZE_PER_TYPE = 20
DEFAULT_FFPROBE_TIMEOUT_SECONDS = 10.0

#: Concurrency for probing a known file list. Each probe is a subprocess that
#: spends nearly all its time waiting, so overlapping them is close to free;
#: the cap exists because the spawns themselves are not, and a few hundred at
#: once buys nothing. Bounded by the file count at the call.
DEFAULT_PROBE_WORKERS = min(16, max(4, (os.cpu_count() or 4)))


def probe_all(
    paths: Sequence[Path],
    probe: Callable[[Path], float | None],
    workers: int | None = None,
) -> dict[Path, float | None]:
    """Read every path's duration, overlapping the probes.

    This is for the case where the full file list is already in hand and each
    one must be priced -- `--dry-run` with a duration floor, and the floor on
    a real run. `run_media_survey` above SAMPLES instead, because it walks
    trees too large to probe at all; this does not get to sample, because the
    caller needs a per-file answer.

    Serial probing here cost real time: pricing one library was still running
    after nine minutes while a plain walk of the same tree listed 50,000
    files in 0.035 seconds. The work was never the disk.

    A `None` is kept, not dropped. Callers read "duration unknown" as "long
    enough to keep", so losing the key would silently change a verdict.
    """
    ordered = list(paths)
    if not ordered:
        return {}
    limit = DEFAULT_PROBE_WORKERS if workers is None else workers
    limit = max(1, min(limit, len(ordered)))
    if limit == 1:
        return {path: probe(path) for path in ordered}
    with ThreadPoolExecutor(max_workers=limit) as pool:
        return dict(zip(ordered, pool.map(probe, ordered), strict=True))


@dataclass
class TypeSurvey:
    extension: str
    count: int = 0
    total_bytes: int = 0
    sample_size: int = 0  # how many were selected for probing
    probe_failed: int = 0  # selected but ffprobe couldn't read them
    durations_seconds: list[float] = field(default_factory=list)  # successfully probed

    @property
    def probed_ok(self) -> int:
        return len(self.durations_seconds)

    @property
    def mean_duration_seconds(self) -> float | None:
        return statistics.fmean(self.durations_seconds) if self.durations_seconds else None

    @property
    def stdev_duration_seconds(self) -> float | None:
        if len(self.durations_seconds) < 2:
            return None
        return statistics.pstdev(self.durations_seconds)

    @property
    def estimated_total_seconds(self) -> float | None:
        mean = self.mean_duration_seconds
        return mean * self.count if mean is not None else None


@dataclass
class MediaSurveyResult:
    root: str
    excludes: tuple[str, ...]
    use_default_excludes: bool
    ffprobe_available: bool
    rate: float | None
    types: list[TypeSurvey]
    walk_stats: WalkStats

    @property
    def total_files(self) -> int:
        return sum(t.count for t in self.types)

    @property
    def total_bytes(self) -> int:
        return sum(t.total_bytes for t in self.types)

    @property
    def total_sampled(self) -> int:
        return sum(t.sample_size for t in self.types)

    @property
    def total_probed_ok(self) -> int:
        return sum(t.probed_ok for t in self.types)

    @property
    def estimated_total_hours(self) -> float | None:
        totals = [t.estimated_total_seconds for t in self.types]
        known = [s for s in totals if s is not None]
        if not known or len(known) != len(totals):
            # A type with files but zero successfully-probed samples makes
            # the grand total an undercount, not just "missing a type" — say
            # so by refusing to produce a number rather than silently
            # excluding that type's (unknown, possibly large) contribution.
            return None
        return sum(known) / 3600.0

    @property
    def estimated_processing_hours(self) -> float | None:
        if self.rate is None or self.rate <= 0:
            return None
        total = self.estimated_total_hours
        return total / self.rate if total is not None else None


def _probe_duration_seconds(
    path: Path, ffprobe_bin: str, timeout: float
) -> float | None:
    """Run `ffprobe` against one file, returning its duration in seconds, or
    `None` if ffprobe can't read it (corrupt/unsupported/times out) — never
    raises. A module-level function so tests can monkeypatch it directly
    without needing a real `ffprobe` binary or real media files.

    An Audacity project is read from its own layout instead: ffprobe cannot
    open one, and a None here gives it the flat base deadline."""
    if path.suffix.lower() == ".aup3":
        return _aup3_duration_seconds(path)
    try:
        proc = subprocess.run(
            [
                ffprobe_bin,
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "csv=p=0",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    try:
        return float(proc.stdout.strip())
    except ValueError:
        return None


def _aup3_duration_seconds(path: Path) -> float | None:
    from corpus.connectors.aup3_layout import LayoutError, read_layout
    from corpus.util.sqlite_ro import connect_ro

    try:
        conn = connect_ro(path, immutable=True)
    except sqlite3.Error:
        return None
    try:
        layout = read_layout(conn)
    except (LayoutError, sqlite3.Error):
        return None
    finally:
        conn.close()
    return layout.duration_s if layout is not None else None


def run_media_survey(
    root: Path,
    excludes: tuple[str, ...] = (),
    use_default_excludes: bool = True,
    sample_size_per_type: int = DEFAULT_SAMPLE_SIZE_PER_TYPE,
    rate: float | None = None,
    ffprobe_timeout: float = DEFAULT_FFPROBE_TIMEOUT_SECONDS,
    ffprobe_path: str | None = None,
    seed: int | None = None,
) -> MediaSurveyResult:
    rng = random.Random(seed)
    stats = WalkStats()
    type_surveys: dict[str, TypeSurvey] = {}
    samplers: dict[str, ReservoirSampler[Path]] = {}

    for wf in walk_files(root, excludes, use_default_excludes, stats=stats):
        ext = wf.path.suffix.lower()
        if ext not in MEDIA_EXTENSIONS:
            continue
        t = type_surveys.setdefault(ext, TypeSurvey(extension=ext))
        t.count += 1
        t.total_bytes += wf.size
        sampler = samplers.setdefault(ext, ReservoirSampler(sample_size_per_type, rng))
        sampler.offer(wf.path)

    ffprobe_bin = ffprobe_path or shutil.which("ffprobe")
    ffprobe_available = ffprobe_bin is not None

    for ext, t in type_surveys.items():
        sample = samplers[ext].sample
        t.sample_size = len(sample)
        if not ffprobe_available:
            continue
        assert ffprobe_bin is not None  # narrowed by ffprobe_available check
        for path in sample:
            duration = _probe_duration_seconds(path, ffprobe_bin, ffprobe_timeout)
            if duration is None:
                t.probe_failed += 1
            else:
                t.durations_seconds.append(duration)

    types = sorted(type_surveys.values(), key=lambda t: (-t.total_bytes, t.extension))
    return MediaSurveyResult(
        root=str(root),
        excludes=excludes,
        use_default_excludes=use_default_excludes,
        ffprobe_available=ffprobe_available,
        rate=rate,
        types=types,
        walk_stats=stats,
    )
