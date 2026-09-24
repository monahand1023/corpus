"""Transcribing a folder of recordings into a sidecar, resumably.

`pipeline.transcribe_file` handles one file. This handles the run: which files
to visit, which to skip because a previous run already answered for them, and
writing every outcome down -- including the negative ones.

RESUMPTION IS THE POINT. A real archive is hours to days of compute, and it
WILL be interrupted: a crash, a reboot, a laptop lid. A run that cannot resume
is a run that never finishes, and the expensive mistake is subtler than losing
transcripts. Files that produced NO usable text are the ones a naive restart
re-does, because they leave no row behind to find. One interrupted pass
re-decoded and re-transcribed 889 already-examined silent clips before the
`no_text` table existed.

WHAT A POLICY CHANGE DOES. Every stored negative carries the fingerprint of
the rules that produced it. Change a threshold and those rows stop matching,
so the files are retried rather than inheriting a verdict made under different
rules. That is why `Settings.as_policy` has to list everything -- a setting
omitted from it silently keeps stale answers alive.
"""

from __future__ import annotations

import contextlib
import contextvars
import json
import logging
import sqlite3
from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from corpus.survey.media import MEDIA_EXTENSIONS
from corpus.survey.walk import WalkStats, walk_files
from corpus.transcripts import store
from corpus.transcripts.audio import NoAudioStreamError
from corpus.transcripts.pipeline import (
    Outcome,
    Settings,
    filter_windows,
    transcribe_file,
)
from corpus.transcripts.segment import file_timeout, join_windows
from corpus.transcripts.worker import WorkerTimeout

logger = logging.getLogger(__name__)


@dataclass
class RunStats:
    considered: int = 0
    skipped_done: int = 0
    rejudged: int = 0
    demoted: int = 0
    transcribed: int = 0
    empty: int = 0
    failed: int = 0
    # Counted separately from `failed` (which it is also part of), because a
    # hang and a broken decode need different responses: one is about this
    # run, the other about the file.
    timed_out: int = 0
    seconds_of_audio: float = 0.0
    errors: list[tuple[str, str]] = field(default_factory=list)

    @property
    def attempted(self) -> int:
        return self.transcribed + self.empty + self.failed


# An archive's own "which media is in scope" policy, supplied by the consumer.
#
# `--exclude GLOB` cannot stand in for it. `fnmatch` is case-sensitive on
# POSIX, so `*karaoke*` misses `Karaoke Track.m4a` where the archive's own
# `pat in path.lower()` does not -- and the policies that live here include
# "never transcribe media from another person's iMessage or WhatsApp backup".
# Translating that into globs silently changes it, and a silent change there
# means transcribing someone who never agreed.
#
# A ContextVar for the same reason `source_excludes` is one: the consumer
# wraps the engine's entry point, and every walk inside the block sees it
# without the engine growing a parameter for a rule it cannot evaluate.
_media_scope: contextvars.ContextVar[Callable[[Path], bool] | None] = (
    contextvars.ContextVar("corpus_media_scope", default=None)
)


@contextmanager
def media_scope(predicate: Callable[[Path], bool]) -> Iterator[None]:
    """Keep only media the predicate returns True for, inside this block."""
    token = _media_scope.set(predicate)
    try:
        yield
    finally:
        _media_scope.reset(token)


def find_media(
    root: Path | str,
    *,
    extensions: Iterable[str] = MEDIA_EXTENSIONS,
    excludes: Sequence[str] = (),
    stats: WalkStats | None = None,
) -> Iterator[Path]:
    """Every media file under `root`, in a stable order.

    Sorted so an interrupted run resumes in the same sequence rather than
    re-walking in whatever order the filesystem returns, which makes progress
    legible across restarts.

    Pass `stats` to find out what the walk WITHHELD. A `.photoslibrary` is
    pruned by default -- right for indexing documents, wrong here, where its
    `originals/` is the home video -- and without this the caller cannot tell
    an empty folder from a folder it declined to open. Read it after the
    iterator is exhausted; it fills in as a side effect.
    """
    stats = stats if stats is not None else WalkStats()
    wanted = {e.lower() for e in extensions}
    in_scope = _media_scope.get()
    found = []
    for walked in walk_files(Path(root), excludes=tuple(excludes), stats=stats):
        if walked.path.suffix.lower() not in wanted:
            continue
        # Counted, not silently dropped: "0 files found" has to stay a
        # question anyone can answer afterwards.
        if in_scope is not None and not in_scope(walked.path):
            stats.files_excluded += 1
            continue
        found.append(walked.path)
    yield from sorted(found)


def rejudge_stored(
    conn: sqlite3.Connection,
    *,
    policy: str,
    settings: Settings,
    paths: set[str] | None = None,
    model_name: str = "",
) -> tuple[int, int]:
    """Re-apply the current rules to transcripts kept under older ones.

    Returns (rejudged, demoted).

    Costs no model time: the text is already stored, and judging text is what
    `quality` does. That is the whole reason this can run on every pass rather
    than being a migration someone has to remember.

    Without it the policy fingerprint kept only half its promise. A rule change
    invalidated stored "no speech" verdicts and retried those files, but a
    stored TRANSCRIPT was equally a verdict -- "this text is real" -- and was
    inherited forever. Measured: adding the loop signal correctly re-examined
    all 126 rejected files in an archive and left the six looping transcripts
    it was written to catch sitting in the index.
    """
    stale = store.stale_transcripts(conn, policy=policy)
    rejudged = demoted = 0
    for path, text, duration_s, languages in stale:
        if paths is not None and path not in paths:
            continue
        # This applies the WHOLE-TRANSCRIPT rules only. It cannot apply the
        # per-window ones to a row whose stored windows did not come from this
        # pipeline -- a restoration record holds one synthetic span covering
        # the file. Restamping such a row says "this faced the current rules",
        # and half of them did.
        #
        # Measured live: 104 rows were restamped that way and every one still
        # held loop text at 0.80-0.82 that the per-window rule would have
        # stripped. Stamped current, they would never have been redone. Left
        # stale, a real re-transcribe picks them up.
        row = conn.execute(
            "SELECT segments FROM transcripts WHERE path = ?", (path,)
        ).fetchone()
        if row is not None:
            windows = _windows_from_segments(row[0])
            if windows and max(d for _t, d, _c in windows) > settings.window_s + 1e-6:
                continue
        rejudged += 1
        verdict = settings.judge(text, duration_s=duration_s, languages=languages)
        if verdict.keep:
            store.restamp_transcript(conn, path, policy=policy)
        else:
            # The window geometry travels with the rejection: without it
            # the decision cannot be re-judged, because the rate rule
            # divides by the WINDOW's duration and only the file's survives.
            stored_segments = ""
            row = conn.execute(
                "SELECT segments FROM transcripts WHERE path = ?", (path,)
            ).fetchone()
            if row is not None and row[0]:
                stored_segments = row[0]
            store.demote_transcript(
                conn, path, duration_s=duration_s, policy=policy,
                reason=verdict.reason or "no_text", rejected_text=text,
                segments=stored_segments,
            )
            demoted += 1
    return rejudged, demoted


@dataclass
class RefilterStats:
    """What a re-filter did, and — just as important — what it did not."""

    rejudged: int = 0
    text_changed: int = 0
    windows_dropped: int = 0
    demoted: int = 0
    skipped_other_model: int = 0
    skipped_no_segments: int = 0
    skipped_wide_windows: int = 0

    def describe(self) -> str:
        parts = [
            f"{self.rejudged:,} re-filtered",
            f"{self.text_changed:,} shortened",
            f"{self.windows_dropped:,} windows dropped",
            f"{self.demoted:,} demoted",
        ]
        skipped = (
            self.skipped_other_model
            + self.skipped_no_segments
            + self.skipped_wide_windows
        )
        if skipped:
            parts.append(
                f"{skipped:,} skipped "
                f"({self.skipped_other_model:,} from another model, "
                f"{self.skipped_no_segments:,} with no stored windows, "
                f"{self.skipped_wide_windows:,} not windowed by this pipeline)"
            )
        return ", ".join(parts)


def _windows_from_segments(
    segments: str | None,
) -> list[tuple[str, float, bool]]:
    """Stored segments as `(text, duration, continues_previous)`.

    `continues_previous` is not stored and IS load-bearing on rejoin, so it is
    reconstructed from the geometry. Windows inside one speech region overlap
    by `window_s - step`, so `start < previous end` means a cut made
    mid-speech, whose duplicated text is an artefact to trim. A window that
    begins after a gap follows real silence, and its repetition may be real.
    """
    try:
        parsed = json.loads(segments or "[]")
    except (TypeError, ValueError):
        return []
    out: list[tuple[str, float, bool]] = []
    prev_end: float | None = None
    for seg in parsed if isinstance(parsed, list) else []:
        if not isinstance(seg, dict):
            continue
        start = float(seg.get("start") or 0.0)
        end = float(seg.get("end") or 0.0)
        continues = prev_end is not None and start < prev_end - 1e-9
        prev_end = end
        out.append(((seg.get("text") or "").strip(), max(end - start, 1e-3), continues))
    return out


def refilter_stored(
    conn: sqlite3.Connection,
    *,
    policy: str,
    settings: Settings,
    model_name: str,
    paths: set[str] | None = None,
) -> RefilterStats:
    """Re-apply the current TEXT filters to stored windows. No model time.

    A re-transcribe of one live archive was measured at 61.7 GPU-hours from
    its own recorded timings, to re-run what amounts to a set of regexes. The
    per-window text is already in `segments`, so it does not have to be
    decoded again.

    WHAT MAKES THIS EQUIVALENT. Stored segments are the windows that survived
    the PREVIOUS filter, so applying a stricter one reaches exactly the state a
    re-decode would. It holds only while the DECODE is unchanged — `window_s`,
    `overlap_s`, the VAD threshold and the model decide which audio becomes
    which window, and none of that can be re-derived from text.

    So: a row from another model is skipped and counted, never restamped; a
    row with no stored windows is skipped; and a `window_s` that could not have
    produced the stored windows raises rather than quietly restamping a policy
    that was never applied. "I could not do this" and "I did this and nothing
    changed" must not look alike — the one rule this codebase keeps relearning.
    """
    # EVERY row, not only ones stamped with another policy. The stamp is
    # computed from SETTINGS, so a behavioural fix in `filter_windows`
    # leaves it unchanged and rows written by the old code read as
    # current. Watched live: a decode-loop fix shipped, a 7-hour
    # re-transcribe ran with the old code, and this reported
    # "0 re-filtered" while three recordings still held a window of 138
    # chars/s -- from the one command whose whole purpose is applying a
    # filter change without re-decoding.
    #
    # Safe to widen because re-judging is text-only and idempotent: a row
    # that does not change is not rewritten.
    stale = store.stale_transcripts(conn, policy=policy, include_current=True)
    stats = RefilterStats()
    for path, text, duration_s, languages in stale:
        if paths is not None and path not in paths:
            continue
        row = conn.execute(
            "SELECT segments, model FROM transcripts WHERE path = ?", (path,)
        ).fetchone()
        if row is None:
            continue
        segments, stored_model = row[0], row[1]
        if stored_model and model_name and stored_model != model_name:
            stats.skipped_other_model += 1
            continue
        windows = _windows_from_segments(segments)
        if not windows:
            stats.skipped_no_segments += 1
            continue
        # A window wider than `window_s` did not come from this geometry.
        # Found on a live sidecar: 104 rows held ONE "window" spanning the
        # whole file, over half an hour of it, because they were RESTORATION
        # records -- text recovered from `no_text` and written back with a
        # synthetic window. Re-judging a whole file as a single window would
        # be a different operation wearing this one's name.
        #
        # Skipped per row, not raised: one unusable row must not abandon the
        # rest, and must not be quietly restamped either.
        if max(d for _t, d, _c in windows) > settings.window_s + 1e-6:
            stats.skipped_wide_windows += 1
            continue

        stats.rejudged += 1
        kept, dropped = filter_windows(windows, settings)
        stats.windows_dropped += len(dropped)
        new_text = join_windows(kept)
        verdict = settings.judge(new_text, duration_s=duration_s, languages=languages)
        if not new_text or not verdict.keep:
            store.demote_transcript(
                conn, path, duration_s=duration_s, policy=policy,
                reason=verdict.reason or "no_text", rejected_text=new_text or text,
                segments=segments or "",
            )
            stats.demoted += 1
            continue
        if new_text != text:
            stats.text_changed += 1
            kept_segments = [
                seg
                for i, seg in enumerate(json.loads(segments or "[]"))
                if i not in {idx for idx, _r in dropped}
            ]
            conn.execute(
                "UPDATE transcripts SET text = ?, segments = ? WHERE path = ?",
                (new_text, json.dumps(kept_segments), path),
            )
            conn.commit()
        store.restamp_transcript(conn, path, policy=policy)
    return stats


def below_duration_floor(duration_s: float | None, min_seconds: float) -> bool:
    """Whether a recording is too short to be worth transcribing.

    An UNKNOWN duration is never below the floor. ffprobe failing is not
    evidence that a recording is short, and skipping on an unknown would
    silently drop real speech -- the failure mode every threshold in this
    project is tuned against.
    """
    if min_seconds <= 0 or duration_s is None:
        return False
    return duration_s < min_seconds


def partition_by_duration(
    paths: Sequence[Path],
    *,
    min_seconds: float,
    probe: Callable[[Path], float | None],
) -> tuple[list[Path], list[Path]]:
    """Split `paths` into (long enough, too short).

    WHY A FLOOR EXISTS AT ALL. Measured on a real photo library of
    photo-library videos: 81% are under four seconds -- the clip Apple stores
    beside each Live Photo. Transcribing them adds four fifths of the library for tens of hours
    of ambience and floods the index with near-empty text that dilutes every
    search. The dry run cannot warn about it either: it reports total hours,
    which cannot show that four fifths of them are four seconds long.

    Probing is skipped entirely when there is no floor. It costs a subprocess
    per file, and paying for all of them to decide nothing is the kind of
    cost that gets a feature switched off.
    """
    if min_seconds <= 0:
        return list(paths), []
    keep: list[Path] = []
    skip: list[Path] = []
    for path in paths:
        (skip if below_duration_floor(probe(path), min_seconds) else keep).append(path)
    return keep, skip


# How many identical timeouts before a file is treated as settled.
#
# A timeout is a failure and failures are retried, because a hang is usually
# about the RUN -- thermal state, memory pressure, a transient decode loop --
# rather than the file. That holds for the first hang and stops holding for
# the third: one recording on a live archive hung three passes running (45
# minutes before the deadline existed, then 162 seconds, then 162 again), and
# retrying it forever bought nothing while costing the deadline every time.
#
# Three, not one, because writing a file off on its first bad night is exactly
# the mistake this is trying not to make. The resulting verdict carries a
# policy fingerprint like any other, so a rule change brings it back: this is
# "we asked three times", not "this file is worthless".
MAX_TIMEOUT_ATTEMPTS = 3


def stale_paths(conn: sqlite3.Connection, *, policy: str) -> list[Path]:
    """Files the current policy invalidated, read from the STORE.

    After a threshold change, re-walking the media roots is the wrong
    operation: the files to redo are already known, and a walk rediscovers
    everything the archive deliberately excluded. Measured on a live archive,
    whose roots hold a photo library of which 81% are the
    sub-4-second clip Apple stores beside each Live Photo -- its own filters
    exclude those, plus karaoke backing tracks and a 15-second floor, and none
    of those rules live in corpus. A blind re-walk would have queued four fifths of it
    near-empty clips for ~33 hours of room tone.

    The sidecar's contents already encode every one of those decisions.

    Rejections count. A `no_text` row is a verdict too, and redoing only the
    transcripts leaves every rejection frozen under rules that no longer
    apply -- the half-kept promise the policy fingerprint exists to close.
    """
    seen: dict[str, None] = {}
    for table in ("transcripts", "no_text"):
        try:
            rows = conn.execute(
                f"SELECT path FROM {table} WHERE policy IS NOT ?", (policy,)
            )
        except sqlite3.Error:
            continue
        for (path,) in rows:
            if path:
                seen.setdefault(path, None)
    return [Path(p) for p in seen]


def stale_paths_present(
    conn: sqlite3.Connection, *, policy: str
) -> tuple[list[Path], int]:
    """`stale_paths`, minus what is not on disk right now, and how many.

    This archive spans external drives. A path that is not mounted is not a
    broken file, and recording thousands of failures for an unplugged volume
    would bury the real ones -- so they are counted and reported, not tried.
    """
    present, missing = [], 0
    for path in stale_paths(conn, policy=policy):
        if path.exists():
            present.append(path)
        else:
            missing += 1
    return present, missing


def _call_bounded(
    transcribe: Callable[..., Outcome],
    path: Path,
    backend: object,
    settings: Settings,
    timeout_s: float,
    run_one: Callable[[Path, float], tuple[str, Any]] | None,
) -> Outcome:
    """Transcribe one file under a deadline, raising `WorkerTimeout` if it blows.

    `run_one` is the strong form: a subprocess that can be KILLED. The CLI
    supplies one, because Whisper's decode loop is inside Metal kernels where
    neither a signal nor a thread cancellation ever arrives -- a real archive
    watched a 170-second clip hold the pipeline for 75 minutes.

    Without one, the deadline is applied with a thread, which bounds how long
    the RUN waits but not how long the work takes: the abandoned thread keeps
    going. That is enough for an injected fake, which cannot hang a GPU, and
    is deliberately not what production uses.
    """
    if run_one is not None:
        kind, payload = run_one(path, timeout_s)
        if kind == "timeout":
            raise WorkerTimeout(f"timed out after {timeout_s:.0f}s")
        if kind == "error":
            detail = str(payload)
            # The worker reports exceptions as text across the pipe, so the
            # one the run loop treats specially has to be recognised again.
            if detail.startswith("NoAudioStreamError"):
                raise NoAudioStreamError(detail)
            raise RuntimeError(detail)
        return payload  # type: ignore[no-any-return]

    from concurrent.futures import ThreadPoolExecutor
    from concurrent.futures import TimeoutError as FutureTimeout

    pool = ThreadPoolExecutor(max_workers=1)
    try:
        future = pool.submit(transcribe, path, backend, settings=settings)
        try:
            return future.result(timeout=timeout_s)
        except FutureTimeout:
            raise WorkerTimeout(f"timed out after {timeout_s:.0f}s") from None
    finally:
        # wait=False: waiting for the abandoned thread would reintroduce
        # exactly the stall this is bounding.
        pool.shutdown(wait=False, cancel_futures=True)


def transcribe_directory(
    root: Path | str,
    db_path: Path | str,
    backend: object,
    *,
    settings: Settings | None = None,
    excludes: Sequence[str] = (),
    limit: int | None = None,
    on_progress: Callable[[int, int, Path, Outcome | None], None] | None = None,
    transcribe: Callable[..., Outcome] = transcribe_file,
    only: Sequence[Path] | None = None,
    file_timeout_s: float | None = None,
    run_one: Callable[[Path, float], tuple[str, Any]] | None = None,
    duration_of: Callable[[Path], float | None] | None = None,
) -> RunStats:
    """Transcribe everything under `root` into the sidecar at `db_path`.

    Safe to interrupt and re-run: anything already answered for under the same
    policy is skipped. `transcribe` is injectable so the run loop can be
    tested without a model.

    `only` replaces the directory walk with an explicit file list -- see
    `stale_paths`, which reads the work list from the store so an archive's
    own exclusion rules are not rediscovered and overturned.

    `duration_of` reads a file's length when the sidecar has none, which is
    every file seen for the first time. Without it such a file gets the flat
    base deadline, too short for any long recording.
    """
    settings = settings or Settings()
    model_name = getattr(backend, "model_name", "unknown")
    policy = store.policy_fingerprint(settings.as_policy(model_name))
    stats = RunStats()

    files = list(only) if only is not None else list(find_media(root, excludes=excludes))
    stats.considered = len(files)

    with store.open_store(db_path) as conn:
        # Before deciding what to skip: anything kept under OLD rules has to
        # face the current ones, or a tightened filter never reaches the
        # material it was written for.
        stats.rejudged, stats.demoted = rejudge_stored(
            conn, policy=policy, settings=settings,
            paths={str(f) for f in files}, model_name=model_name,
        )
        # Sweep failure rows for files that already have a verdict. They can
        # never be reached by processing, because `already_done` skips them.
        swept = store.clear_settled_failures(conn, policy=policy)
        if swept:
            logger.info("cleared %d stale failure row(s)", swept)

        done = store.already_done(conn, policy=policy)
        todo = [p for p in files if str(p) not in done]
        stats.skipped_done = len(files) - len(todo)
        if limit is not None:
            todo = todo[:limit]

        for index, path in enumerate(todo, start=1):
            # `file_timeout` has computed this bound since the pipeline
            # landed, and nothing ever called it: the value was ported into
            # corpus and the mechanism that enforces it was left behind. A
            # bound with no way to enforce it reads exactly like one that
            # works, which is how a 17-minute stall went unnoticed on a work
            # list whose longest recording should take ten.
            deadline = file_timeout_s
            if deadline is None:
                recorded = conn.execute(
                    "SELECT duration_s FROM transcripts WHERE path = ?"
                    " UNION SELECT duration_s FROM no_text WHERE path = ?",
                    (str(path), str(path)),
                ).fetchone()
                duration = recorded[0] if recorded else None
                if not duration and duration_of is not None:
                    duration = duration_of(path)
                deadline = file_timeout(float(duration or 0.0))
            try:
                outcome = _call_bounded(
                    transcribe, path, backend, settings, deadline, run_one
                )
            except WorkerTimeout as exc:
                # A FAILURE, not a settled verdict: a hang is usually about
                # this run -- thermal state, memory pressure, a transient
                # decode loop -- so the file comes back on the next pass
                # rather than being written off.
                stats.timed_out += 1
                stats.failed += 1
                stats.errors.append((str(path), str(exc)))
                store.save_failure(conn, str(path), str(exc))
                if store.failure_attempts(conn, str(path)) >= MAX_TIMEOUT_ATTEMPTS:
                    # It has never once completed. Settle it so it stops
                    # costing the deadline on every future pass, keeping the
                    # policy fingerprint so a rule change brings it back.
                    store.save_no_text(
                        conn, str(path), duration_s=0.0,
                        reason="repeatedly_timed_out", policy=policy,
                    )
                    logger.warning(
                        "%s has now timed out %d times; recording it as "
                        "repeatedly_timed_out so it is not retried until a "
                        "rule changes",
                        path,
                        MAX_TIMEOUT_ATTEMPTS,
                    )
                logger.warning("transcription %s for %s", exc, path)
                if on_progress:
                    on_progress(index, len(todo), path, None)
                continue
            except NoAudioStreamError:
                # A settled fact about the FILE, not a failure of this run.
                # Recorded as no_text so it is skipped next time: measured on
                # a live photo library, 10% of files have no audio track, and
                # as failures they were re-decoded on every run forever while
                # putting an error in front of the operator that they could
                # never act on. An error nobody can act on hides the ones
                # they can.
                stats.empty += 1
                store.clear_failure(conn, str(path))
                store.save_no_text(
                    conn, str(path), duration_s=0.0,
                    reason="no_audio_stream", policy=policy,
                )
                if on_progress:
                    # NOT None. None is what a crash reports, and this is a
                    # settled verdict -- 94 of these in a row printed as ERR
                    # and looked exactly like a run falling over.
                    on_progress(
                        index,
                        len(todo),
                        path,
                        Outcome(path=str(path), empty_reason="no_audio_stream"),
                    )
                continue
            except Exception as exc:  # one bad file must not end the run
                # A failure is recorded and NOT skipped on the next run: it
                # usually means a broken decode or a transient resource
                # problem, not a settled verdict about the audio.
                stats.failed += 1
                stats.errors.append((str(path), f"{type(exc).__name__}: {exc}"))
                store.save_failure(conn, str(path), f"{type(exc).__name__}: {exc}")
                logger.warning("transcription failed for %s: %s", path, exc)
                if on_progress:
                    on_progress(index, len(todo), path, None)
                continue

            # RECORDING the answer can fail too, and until this was wrapped
            # one bad value took the whole pass with it: a NaN avg_logprob --
            # which SQLite cannot store, so a NOT NULL column rejects it --
            # aborted a live run 59 files into a long one. Transcription was
            # protected per file while the write beside it was not, so one
            # unstorable window discarded every file after it.
            try:
                _record(conn, path, outcome, policy, stats)
            except Exception as exc:
                detail = f"could not record result: {type(exc).__name__}: {exc}"
                stats.failed += 1
                stats.errors.append((str(path), detail))
                with contextlib.suppress(Exception):
                    store.save_failure(conn, str(path), detail)
                logger.warning("%s for %s", detail, path)
                if on_progress:
                    on_progress(index, len(todo), path, None)
                continue


            if on_progress:
                on_progress(index, len(todo), path, outcome)

    return stats



def _record(
    conn: sqlite3.Connection,
    path: Path,
    outcome: Outcome,
    policy: str,
    stats: RunStats,
) -> None:
    """Write one file's result to the store.

    Split out of the run loop so the loop can protect it: recording an
    answer can fail for reasons that have nothing to do with the next
    file, and a failure here must cost one file rather than the pass.
    """
    stats.seconds_of_audio += outcome.duration_s
    if outcome.dropped:
        store.save_dropped_windows(
            conn,
            str(path),
            [
                {
                    "window_start": d.window_start,
                    "no_speech": d.no_speech,
                    "avg_logprob": d.avg_logprob,
                    "text": d.text,
                    # WHICH rule fired. Computed per window and then
                    # dropped here, which left the store unable to
                    # answer whether any filter is dead.
                    "reason": d.reason,
                }
                for d in outcome.dropped
            ],
            policy=policy,
        )

    # This file now has an answer, so any `failures` row from an
    # earlier attempt has stopped being true. Clearing it here covers
    # every verdict below in one place.
    store.clear_failure(conn, str(path))

    if outcome.transcript is not None:
        outcome.transcript.policy = policy
        store.save_transcript(conn, outcome.transcript)
        stats.transcribed += 1
    else:
        # The row that makes a restart cheap. Without it this file is
        # simply absent, and the next run pays for it again to reach
        # the same answer.
        store.save_no_text(
            conn,
            str(path),
            duration_s=outcome.duration_s,
            policy=policy,
            reason=outcome.empty_reason or "no_text",
            rejected_text=outcome.rejected_text,
        )
        stats.empty += 1
