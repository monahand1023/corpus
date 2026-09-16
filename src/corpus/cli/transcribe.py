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
from collections.abc import Sequence
from pathlib import Path

from corpus.cli._common import configure_logging
from corpus.survey.format import human_count
from corpus.transcripts.pipeline import Settings
from corpus.transcripts.run import find_media, transcribe_directory

DEFAULT_DB = "data/transcripts.db"


def _humanise_hours(seconds: float) -> str:
    hours = seconds / 3600
    if hours < 1:
        return f"{seconds / 60:.0f} min"
    return f"{hours:.1f} h"


def _dry_run(
    root: Path, db: Path, excludes: Sequence[str], rate: float, settings: Settings
) -> int:
    """Price the run without transcribing anything.

    Deliberately the documented first step. This spends hours of local compute
    and the estimate is the only warning before it does.

    It resolves the backend even though it will not transcribe, for two
    reasons: the count of already-done files is scoped to the policy, and the
    policy names the model -- and a missing extra is cheapest to discover
    here, before anyone commits to the run.
    """
    from corpus.survey.media import run_media_survey
    from corpus.transcripts.backends import BackendUnavailableError, default_backend

    model_name: str | None = None
    backend_problem: str | None = None
    try:
        model_name = default_backend().model_name
    except BackendUnavailableError as exc:
        backend_problem = str(exc)

    files = list(find_media(root, excludes=excludes))
    print(f"corpus-transcribe: {root}")
    print(f"  media files found : {human_count(len(files))}")
    if not files:
        print("\n  Nothing to transcribe. `corpus-survey census` shows what IS here.")
        return 1

    survey = run_media_survey(root, excludes=tuple(excludes), rate=rate)
    hours = survey.estimated_total_hours
    if hours is None:
        print(
            "  duration          : unknown (ffprobe unavailable, so length "
            "could not be sampled)"
        )
    else:
        print(f"  audio to process  : ~{hours:.1f} h")
        print(
            f"  estimated runtime : ~{hours / rate:.1f} h at {rate:g}x realtime "
            "(local compute; no API spend)"
        )

    if db.exists():
        from corpus.transcripts import store

        # Counts files judged to hold NO SPEECH as done, not just transcribed
        # ones. Reporting transcripts alone understated the skip set badly: on
        # a real archive most files produce no text, and those rows exist
        # precisely so a second run does not pay for them again.
        with store.open_store(db, read_only=True) as conn:
            if model_name is not None:
                policy = store.policy_fingerprint(settings.as_policy(model_name))
                done = store.already_done(conn, policy=policy)
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
        "--rate", type=float, default=15.0,
        help="Assumed speed as a multiple of realtime, for the estimate only",
    )
    parser.add_argument(
        "--language", action="append", default=[], dest="languages", metavar="CODE",
        help="A language you actually speak (repeatable). Transcribers "
             "mislabel silence as languages nobody in the recording speaks, "
             "and naming yours lets that be used as a rejection signal.",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)

    configure_logging(args.verbose)
    root = Path(args.path).expanduser()
    if not root.is_dir():
        print(f"error: not a directory: {root}", file=sys.stderr)
        return 1
    db = Path(args.db).expanduser()

    settings = Settings(expected_languages=frozenset(args.languages))
    if args.dry_run:
        return _dry_run(root, db, args.excludes, args.rate, settings)

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

    total = len(list(find_media(root, excludes=args.excludes)))
    print(f"corpus-transcribe: {root} -> {db}")
    print(f"  {human_count(total)} media file(s); already-done files are skipped\n")

    def progress(index: int, todo: int, path: Path, outcome: object) -> None:
        mark = "ok " if getattr(outcome, "produced_text", False) else "-- "
        if outcome is None:
            mark = "ERR"
        print(f"  [{index}/{todo}] {mark} {path.name[:68]}")

    stats = transcribe_directory(
        root, db, backend,
        settings=settings, excludes=args.excludes, limit=args.limit,
        on_progress=progress,
    )

    print(
        f"\n  transcribed : {human_count(stats.transcribed)}"
        f"\n  no speech   : {human_count(stats.empty)}"
        f"\n  failed      : {human_count(stats.failed)}"
        f"\n  skipped     : {human_count(stats.skipped_done)} (done by an earlier run)"
        f"\n  audio seen  : {_humanise_hours(stats.seconds_of_audio)}"
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
