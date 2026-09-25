"""corpus-transcribe: turn a folder of recordings into an indexable sidecar.

    corpus-transcribe ~/Videos --dry-run     # how much audio, how long, what it costs
    corpus-transcribe ~/Videos               # do it (resumable)
    corpus-transcribe ~/Videos --limit 20    # a sample first, to see the quality

This is the step `corpus-index` cannot do for you, and deliberately so: it is
hours of local compute rather than seconds of I/O, and it belongs behind its
own confirmation. Afterwards the sidecar is a source like any other --
`type = "transcripts"`, pointed at the database.

SAFE TO INTERRUPT. Every outcome is written as it happens, including the
negative ones, so a second run skips what the first already answered. That
matters more than it sounds: files that produce NO usable text are the ones a
naive restart re-does, because they leave nothing behind to find.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Container, Mapping, Sequence
from pathlib import Path

from corpus.cli._common import configure_logging
from corpus.survey.format import human_count
from corpus.survey.walk import WalkStats
from corpus.transcripts.pipeline import Settings
from corpus.transcripts.run import find_media, partition_by_duration, transcribe_directory
from corpus.util.priority import DEFAULT_NICE, be_nice

DEFAULT_DB = "data/transcripts.db"


def _duration_probe() -> Callable[[Path], float | None]:
    """ffprobe-backed duration lookup, or a no-op when ffprobe is missing.

    Returning None for everything is the safe degradation: `partition_by_duration`
    keeps a file whose duration it cannot read, so a missing ffprobe means the
    floor simply does not apply rather than silently dropping the archive.
    """
    import shutil

    from corpus.survey.media import DEFAULT_FFPROBE_TIMEOUT_SECONDS, _probe_duration_seconds

    binary = shutil.which("ffprobe")
    if binary is None:
        return lambda _path: None
    return lambda path: _probe_duration_seconds(
        path, binary, DEFAULT_FFPROBE_TIMEOUT_SECONDS
    )


def _report_withheld(stats: WalkStats) -> None:
    """Say what the walk refused to open, and how to open it.

    A `.photoslibrary` is pruned from every walk by default, which is right
    for indexing -- the bundle is Apple's SQLite, plists and derivatives --
    and wrong here, because `originals/` inside it is the home video. On a
    reference archive that is the long tail at 15 seconds or more.

    Printing "0 media files found" for a folder holding a photo library is
    the same failure this pipeline keeps producing in other forms: a refusal
    to look, rendered identically to a look that found nothing. Pointing the
    root AT the bundle already works -- nothing said so, and nobody should
    have to derive it from the exclusion rules.
    """
    if not stats.media_bundles_pruned:
        return
    n = len(stats.media_bundles_pruned)
    print(
        f"  photo libraries   : {n} skipped by default "
        "(bundle internals are not documents)"
    )
    print("    Their originals/ folders ARE home video. To transcribe one,")
    print("    point the root at the bundle itself:")
    for bundle in stats.media_bundles_pruned[:3]:
        print(f'      corpus-transcribe "{bundle}" --min-seconds 15')
    if n > 3:
        print(f"      ... and {n - 3} more")


def _format_duration(hours: float) -> str:
    """Render a duration in a unit that still carries information.

    Hours are right for the number this command usually prints and wrong once
    the work shrinks -- and it shrinks precisely when the skip set is large,
    which is the common case on a second run. "~0.0 h" is true for 75 files
    and reads as "nothing to do" rather than "two minutes".
    """
    if hours >= 1.0:
        return f"~{hours:.1f} h"
    minutes = hours * 60.0
    if minutes >= 1.0:
        return f"~{minutes:.0f} min"
    return f"~{hours * 3600.0:.0f} s"


def _runtime_hours(
    files: Sequence[Path],
    durations: Mapping[Path, float | None],
    done: Container[str],
    *,
    done_is_exact: bool,
) -> tuple[float | None, int]:
    """Hours this run will actually decode, and how many files that is.

    Returns `(None, n)` when any surviving file's duration could not be read:
    a partial sum reads exactly like a total, and this number is the only
    warning before hours of compute.

    The already-done set is subtracted ONLY when it is exact. Without a model
    there is no policy to scope it by, so it is an upper bound -- and
    subtracting an upper bound understates the run, which is the direction
    that costs someone an unexpected night. Overstating is not free either:
    it talks people out of a job that would have taken twenty minutes.
    """
    if not files:
        return (None, 0)
    todo = (
        [f for f in files if str(f) not in done] if done_is_exact else list(files)
    )
    known = [d for f in todo if (d := durations.get(f)) is not None]
    if len(known) != len(todo):
        return (None, len(todo))
    return (sum(known) / 3600.0, len(todo))


def _dry_run(
    root: Path,
    db: Path,
    excludes: Sequence[str],
    rate: float,
    settings: Settings,
    min_seconds: float = 0.0,
) -> int:
    """Price the run without transcribing anything.

    Deliberately the documented first step. This spends hours of local compute
    and the estimate is the only warning before it does.

    It resolves the backend even though it will not transcribe, for two
    reasons: the count of already-done files is scoped to the policy, and the
    policy names the model -- and a missing extra is cheapest to discover
    here, before anyone commits to the run.
    """
    from corpus.survey.media import probe_all, run_media_survey
    from corpus.transcripts.backends import BackendUnavailableError, default_backend

    model_name: str | None = None
    backend_problem: str | None = None
    try:
        model_name = default_backend().model_name
    except BackendUnavailableError as exc:
        backend_problem = str(exc)

    walked = WalkStats()
    files = list(find_media(root, excludes=excludes, stats=walked))
    print(f"corpus-transcribe: {root}")
    print(f"  media files found : {human_count(len(files))}")
    _report_withheld(walked)
    # When a floor is set, every file has just been probed, so the exact
    # duration of what WILL run is known -- better than the sampled estimate
    # below, and it must not contradict the line above it. Printing "204
    # skipped" and then quoting hours for all 857 is worse than printing no
    # number at all: the dry run is the one warning before hours of compute.
    durations: dict[Path, float | None] = {}
    if min_seconds > 0 and files:
        durations = probe_all(files, _duration_probe())
        files, too_short = partition_by_duration(
            files, min_seconds=min_seconds, probe=lambda f: durations.get(f)
        )
        print(
            f"  under {min_seconds:g}s          : {human_count(len(too_short))} "
            "skipped (a duration that cannot be read counts as long enough)"
        )
    if not files:
        print("\n  Nothing to transcribe. `corpus-survey census` shows what IS here.")
        return 1

    # The skip set is resolved BEFORE the estimate, because the estimate is
    # about the files that will actually be decoded. Reporting it afterwards
    # put "estimated runtime ~14.4 h" directly above "already done: 5,407 of
    # these (skipped)" on a real archive where 363 files needed transcribing.
    done: set[str] = set()
    exact = False
    if db.exists():
        from corpus.transcripts import store

        # Counts files judged to hold NO SPEECH as done, not just transcribed
        # ones. Reporting transcripts alone understated the skip set badly: on
        # a real archive most files produce no text, and those rows exist
        # precisely so a second run does not pay for them again.
        with store.open_store(db, read_only=True) as conn:
            if model_name is not None:
                policy = store.policy_fingerprint(settings.as_policy(model_name))
                done = set(store.already_done(conn, policy=policy))
                exact = True
            else:
                # Without a model there is no policy, so the best available
                # answer ignores it -- an UPPER bound, since a row recorded
                # under different rules is retried rather than inherited.
                done = {r[0] for r in conn.execute("SELECT path FROM transcripts")}
                done |= {r[0] for r in conn.execute("SELECT path FROM no_text")}
                exact = False
        # Intersected with what this run would actually visit: rows left by
        # some other directory are not work this run gets to skip.
        already = len({str(f) for f in files} & done)
        if already:
            qualifier = "" if exact else " at most"
            print(
                f"  already done      :{qualifier} {human_count(already)} of these "
                f"(skipped; includes files found to hold no speech)"
            )

    # Exact when the floor already required probing every file. When it did
    # not, or when any one duration could not be read, this DEGRADES to the
    # sampled estimate rather than replacing it -- an exact number that
    # sometimes prints nothing is worse than an approximate one that always
    # prints something, and "nothing" here reads as "no work to do".
    hours, todo = _runtime_hours(files, durations, done, done_is_exact=exact)
    priced_remaining = hours is not None
    if hours is None:
        survey = run_media_survey(root, excludes=tuple(excludes), rate=rate)
        hours = survey.estimated_total_hours
        todo = len(files)
    if hours is None:
        print(
            "  duration          : unknown (ffprobe unavailable, so length "
            "could not be sampled)"
        )
    else:
        scope = "left to do" if priced_remaining and exact and done else "to process"
        print(
            f"  audio {scope:11s} : {_format_duration(hours)} across "
            f"{human_count(todo)} file(s)"
        )
        print(
            f"  estimated runtime : {_format_duration(hours / rate)} at "
            f"{rate:g}x realtime (local compute; no API spend)"
        )

    print(
        "\n  Estimates come from sampling file durations, not from decoding "
        "every file. Treat the runtime as an order of magnitude: it moves with "
        "how much of the audio is actually speech, because silence is skipped "
        "without the model running."
    )
    if backend_problem:
        print(f"\n  NOTE: transcription cannot run yet -- {backend_problem}")

    print("\n(dry run — nothing transcribed)")
    return 0 if not backend_problem else 2


def _dry_run_redo_stale(
    db: Path, settings: Settings, rate: float, since: str | None
) -> int:
    """Price a --redo-stale run from the sidecar: the same work list the real
    run reads, and durations already stored, so nothing is walked or probed."""
    from corpus.transcripts import store
    from corpus.transcripts.backends import BackendUnavailableError, default_backend
    from corpus.transcripts.run import stale_paths_present

    try:
        model_name = default_backend().model_name
    except BackendUnavailableError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    policy = store.policy_fingerprint(settings.as_policy(model_name))
    decode_policy = store.policy_fingerprint(settings.as_decode_policy(model_name))
    with store.open_store(db, read_only=True) as conn:
        only, missing, out_of_scope = stale_paths_present(
            conn, policy=policy, since=since, decode_policy=decode_policy
        )
        durations: dict[str, float] = {}
        for table in ("no_text", "transcripts"):
            for path, duration in conn.execute(f"SELECT path, duration_s FROM {table}"):
                durations[path] = float(duration or 0.0)
    seconds = sum(durations.get(str(p), 0.0) for p in only)
    print(f"corpus-transcribe --redo-stale --dry-run: {db}")
    print(f"  {human_count(len(only))} file(s) invalidated by the current policy")
    if missing:
        print(f"  {human_count(missing)} more are recorded but not on disk right now")
    if out_of_scope:
        print(f"  {human_count(out_of_scope)} more are outside this archive's media policy")
    hours = seconds / 3600
    print(f"  audio to redo      : {_format_duration(hours)}")
    print(
        f"  estimated runtime  : {_format_duration(hours / rate)} at {rate:g}x "
        "realtime (local compute; no API spend)"
    )
    print("(dry run — nothing transcribed)")
    return 0


def _refilter(db: Path, settings: Settings) -> int:
    """Re-apply text filters to stored windows. No model, no audio, no GPU."""
    from corpus.transcripts import store
    from corpus.transcripts.run import refilter_stored

    if not db.is_file():
        print(f"error: no sidecar at {db}", file=sys.stderr)
        return 1
    conn = store.connect(db)
    try:
        model = conn.execute(
            "SELECT model FROM transcripts WHERE model != '' "
            "GROUP BY model ORDER BY count(*) DESC LIMIT 1"
        ).fetchone()
        model_name = model[0] if model else ""
        policy = store.policy_fingerprint(settings.as_policy(model_name))
        print(f"corpus-transcribe --refilter: {db}")
        print(f"  model {model_name or '(unrecorded)'}, policy {policy}")
        print("  no audio is decoded; only stored window text is re-judged\n")
        try:
            stats = refilter_stored(
                conn, policy=policy, settings=settings, model_name=model_name
            )
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        print(f"  {stats.describe()}")
        if stats.demoted:
            print(
                f"\n  {stats.demoted:,} transcript(s) no longer pass and moved to "
                "no_text; their text is retained there, so this is auditable."
            )
        if stats.text_changed or stats.demoted:
            print("\n  Re-run corpus-ingest to update the index.")
        return 0
    finally:
        conn.close()


def main_argv(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="corpus-transcribe",
        description="Transcribe a folder of audio/video into an indexable sidecar",
    )
    parser.add_argument("path", help="Directory of recordings")
    parser.add_argument(
        "--db", default=DEFAULT_DB,
        help=f"Sidecar database to write (default: {DEFAULT_DB})",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Report how much audio there is and how long it would take. "
             "The intended first step: this spends hours of local compute.",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Stop after this many files. Use it to sample the quality on a "
             "handful before committing to the whole archive.",
    )
    parser.add_argument(
        "--exclude", action="append", default=[], dest="excludes", metavar="GLOB",
        help="Skip paths matching this pattern (repeatable)",
    )
    parser.add_argument(
        "--min-seconds",
        type=float,
        default=0.0,
        metavar="N",
        help=(
            "Skip recordings shorter than N seconds. Measured on a real "
            "real photo library: 81%% of its videos are under four "
            "seconds -- the clip Apple stores beside each Live Photo -- which "
            "is four fifths of the library and tens of hours of room tone. A file whose "
            "duration cannot be read is KEPT, never skipped."
        ),
    )
    parser.add_argument(
        "--rate", type=float, default=15.0,
        help="Assumed speed as a multiple of realtime, for the estimate only",
    )
    parser.add_argument(
        "--language", action="append", default=[], dest="languages", metavar="CODE",
        help="A language actually spoken in these recordings (repeatable). "
             "A window the model labels as any other language is decoded "
             "again in the likeliest of these, instead of coming back as "
             "text in a language nobody speaks.",
    )
    parser.add_argument(
        "--refilter",
        action="store_true",
        help=(
            "Re-apply the current TEXT filters to windows already stored in "
            "the sidecar, without decoding any audio. Use after changing a "
            "quality threshold: re-transcribing one 380-hour archive was "
            "measured at 61.7 GPU-hours to re-run what amounts to a regex. "
            "Refuses on rows from a different model or a different window "
            "geometry, which only a real re-transcribe can redo."
        ),
    )
    parser.add_argument(
        "--transcribed-since",
        metavar="DATE",
        default=None,
        help=(
            "With --redo-stale: only rows written on or after DATE (YYYY-MM-DD). "
            "A pipeline change invalidates every row's policy, but only the "
            "rows that pipeline wrote carry its defect."
        ),
    )
    parser.add_argument(
        "--redo-stale",
        action="store_true",
        help=(
            "Re-transcribe exactly the files the current policy invalidated, "
            "taking the work list from the sidecar instead of walking the "
            "media roots. After a threshold change that is the right "
            "operation: a re-walk rediscovers everything the archive chose to "
            "exclude, and those rules live in the archive, not in corpus."
        ),
    )
    parser.add_argument(
        "--nice",
        type=int,
        default=DEFAULT_NICE,
        metavar="N",
        help=(
            f"Lower this job's scheduling priority by N (default {DEFAULT_NICE}). "
            "It is long background work and something interactive is probably "
            "sharing the machine. Children inherit it. Use 0 to leave priority "
            "alone; it cannot be raised again afterwards."
        ),
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)
    if args.transcribed_since and not args.redo_stale:
        parser.error("--transcribed-since only applies with --redo-stale")
    be_nice(args.nice)

    configure_logging(args.verbose)
    root = Path(args.path).expanduser()
    if not root.is_dir() and not args.redo_stale:
        print(f"error: not a directory: {root}", file=sys.stderr)
        return 1
    db = Path(args.db).expanduser()

    settings = Settings(expected_languages=frozenset(args.languages))
    if args.dry_run and args.redo_stale:
        return _dry_run_redo_stale(db, settings, args.rate, args.transcribed_since)
    if args.dry_run:
        return _dry_run(root, db, args.excludes, args.rate, settings, args.min_seconds)
    if args.refilter:
        return _refilter(db, settings)

    from corpus.survey.media import probe_all
    from corpus.transcripts.audio import ffmpeg_available
    from corpus.transcripts.backends import BackendUnavailableError, default_backend

    if not ffmpeg_available():
        print(
            "error: ffmpeg is not on PATH. Transcription decodes audio with "
            "it; install it (macOS: `brew install ffmpeg`) and retry.",
            file=sys.stderr,
        )
        return 2
    try:
        backend = default_backend()
    except BackendUnavailableError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    only = None
    if args.redo_stale:
        from corpus.transcripts import store
        from corpus.transcripts.run import stale_paths_present

        with store.open_store(db, read_only=True) as conn:
            policy = store.policy_fingerprint(
                settings.as_policy(getattr(backend, "model_name", "unknown"))
            )
            decode_policy = store.policy_fingerprint(
                settings.as_decode_policy(getattr(backend, "model_name", "unknown"))
            )
            only, missing, out_of_scope = stale_paths_present(
                conn, policy=policy, since=args.transcribed_since,
                decode_policy=decode_policy,
            )
        total = len(only)
        print(f"corpus-transcribe --redo-stale: {db}")
        print(f"  {human_count(total)} file(s) invalidated by the current policy")
        if missing:
            # An unmounted volume is not a broken file, and recording
            # thousands of failures for one would bury the real ones.
            print(
                f"  {human_count(missing)} more are recorded but not on disk "
                "right now (unmounted volume?) and are left alone"
            )
        if out_of_scope:
            print(
                f"  {human_count(out_of_scope)} more are outside this archive's "
                "media policy and are left alone"
            )
        print()
        if not total:
            print("  Nothing to redo.")
            return 0
    else:
        walked = WalkStats()
        found = list(find_media(root, excludes=args.excludes, stats=walked))
        print(f"corpus-transcribe: {root} -> {db}")
        _report_withheld(walked)
        if args.min_seconds > 0:
            durations = probe_all(found, _duration_probe())
            found, too_short = partition_by_duration(
                found, min_seconds=args.min_seconds, probe=lambda f: durations.get(f)
            )
            # Pass the survivors as the explicit work list, so the floor is
            # applied BEFORE decoding -- which is the cost it exists to avoid.
            only = found
            print(
                f"  {human_count(len(too_short))} file(s) under "
                f"{args.min_seconds:g}s skipped before decoding"
            )
        print(
            f"  {human_count(len(found))} media file(s); already-done files "
            "are skipped\n"
        )

    def progress(index: int, todo: int, path: Path, outcome: object) -> None:
        mark = "ok " if getattr(outcome, "produced_text", False) else "-- "
        if outcome is None:
            mark = "ERR"
        # flush because this is the ONLY sign of life on a multi-hour run, and
        # Python block-buffers stdout when it is not a terminal. Redirected to
        # a log or a pipe -- which is how a long run is actually started -- the
        # entire progress stream stayed in the buffer, so the log showed the
        # banner and then nothing for hours.
        print(f"  [{index}/{todo}] {mark} {path.name[:68]}", flush=True)

    # A SUBPROCESS, not a thread. Whisper can enter an unbounded decode loop
    # on non-speech audio -- a real archive watched a 170-second clip hold the
    # pipeline for 75 minutes, GPU busy, producing nothing. Signals arrive at
    # Python bytecode boundaries and the loop is inside Metal kernels; threads
    # cannot be killed. Only a process can be terminated.
    #
    # corpus computed the deadline for this from the day the pipeline landed
    # and never enforced it: `file_timeout` was imported, tested, and called
    # by nothing.
    from corpus.transcripts.worker import TranscribeWorker

    worker = TranscribeWorker()
    try:
        stats = transcribe_directory(
            root, db, backend,
            settings=settings, excludes=args.excludes, limit=args.limit,
            on_progress=progress, only=only,
            run_one=lambda path, timeout: worker.run(path, timeout, settings),
            duration_of=_duration_probe(),
            redo=bool(args.redo_stale),
        )
    finally:
        worker.close()

    if stats.timed_out:
        print(
            f"\n  {human_count(stats.timed_out)} file(s) exceeded their deadline and "
            "were cut off mid-decode. Recorded as FAILURES, not settled verdicts, so "
            "the next run retries them: a hang is usually about the run (thermal "
            "state, memory pressure, a decode loop) rather than about the file."
        )
    if stats.demoted:
        print(
            f"\n  {human_count(stats.demoted)} transcript(s) kept by the previous "
            f"rules no longer pass the current ones and were moved to no_text "
            f"(their text is retained there, so the change can be audited)."
        )
    print(
        f"\n  transcribed : {human_count(stats.transcribed)}"
        f"\n  no speech   : {human_count(stats.empty)}"
        f"\n  failed      : {human_count(stats.failed)}"
        f"\n  skipped     : {human_count(stats.skipped_done)} (done by an earlier run)"
        f"\n  audio seen  : {_format_duration(stats.seconds_of_audio / 3600).lstrip("~")}"
    )
    for path, error in stats.errors[:5]:
        print(f"    ERR {Path(path).name}: {error}")

    if stats.transcribed:
        print(
            f"\nAdd to corpus.toml, then run corpus-ingest:\n"
            f'\n  [[sources]]\n  name = "transcripts"\n  type = "transcripts"\n'
            f'  path = "{db}"\n'
        )
    return 0 if not stats.failed else 3


def main() -> int:
    return main_argv(sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
