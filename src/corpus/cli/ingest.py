"""corpus-ingest: load configured sources into the chunk store.

Usage:
  corpus-ingest --source notes
  corpus-ingest --source notes --source papers
  corpus-ingest --all
"""

from __future__ import annotations

import argparse
import sys

from corpus.cli._common import configure_logging, load_config_or_exit
from corpus.connectors.registry import DEFAULT_GLOBS
from corpus.credentials import resolve_dotenv
from corpus.ingester import Ingester
from corpus.util.autodetect import detect_sources
from corpus.util.priority import DEFAULT_NICE, be_nice

# Completed, but a guard reported something a human should look at: a yield
# collapse, a source name reused for a different path, or files newly
# reclassified as permanently unreadable while their chunks were pruned.
EXIT_ANOMALY = 3


def _print_tokens(*, tokens_used: int, counted: bool) -> None:
    """Report embedding spend, distinguishing "none" from "not measured".

    A provider that returns no usage figure leaves the counter at 0 forever.
    Printing that as `tokens billed: 0` says the run was free, whatever it
    actually embedded -- and says it identically for a run that genuinely
    embedded nothing because every chunk was unchanged, which is a real and
    useful thing to be able to see.
    """
    if counted:
        print(f"  tokens billed:    {tokens_used:,}")
    else:
        print("  tokens billed:    not reported by this embedding provider")


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest a configured source")
    parser.add_argument(
        "--source",
        action="append",
        default=None,
        help="Source name from corpus.toml (repeatable)",
    )
    parser.add_argument("--all", action="store_true", help="Ingest every configured source")
    parser.add_argument(
        "--path",
        default=None,
        metavar="DIR",
        help=(
            "Ingest whatever is in DIR: detect which built-in connectors apply "
            "and ingest each matching file type as its own source. Needs no "
            "[[sources]] block; corpus.toml still supplies the database path "
            "and embedder. Cannot be combined with --source or --all."
        ),
    )
    parser.add_argument(
        "--prune-anyway",
        action="store_true",
        help=(
            "Delete orphaned chunks even when the connector reported files it "
            "could not read, OR when the blast-radius guard would otherwise "
            "refuse a prune that looks too large relative to the source's "
            "existing chunks (see [pruning] in corpus.toml). Requires "
            "--source; refused with --all."
        ),
    )
    parser.add_argument("--config", default=None, help="Path to corpus.toml (default: ./corpus.toml)")
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
    args = parser.parse_args()
    be_nice(args.nice)

    # Resolve credentials now that --config is known (env var > .env beside
    # --config > .env in cwd). See corpus.credentials for the precedence.
    resolve_dotenv(args.config)

    configure_logging(args.verbose)

    if args.path and (args.all or args.source):
        print("--path cannot be combined with --source or --all.")
        return 2

    # --prune-anyway forces a destructive sweep past the safety gate, so it is
    # deliberately not available across every source at once: the operator has
    # to have looked at which files failed, for a source they named.
    if args.prune_anyway and args.all:
        print(
            "Refusing --prune-anyway with --all: it would force a destructive "
            "prune across every source. Re-run with --source NAME for the "
            "specific source whose unreadable files you have reviewed."
        )
        return 2

    config = load_config_or_exit(args.config)
    if args.path:
        # Replace the configured sources with what is actually in the folder.
        # corpus.toml still provides db_path, embedder and retriever settings;
        # only [[sources]] is superseded.
        try:
            detected = detect_sources(args.path)
        except FileNotFoundError as e:
            print(f"ERROR: {e}")
            return 1
        if not detected:
            print(
                f"Nothing ingestable found in {args.path}. Supported types: "
                + ", ".join(sorted(DEFAULT_GLOBS))
            )
            return 1
        config = config.model_copy(update={"sources": detected})
        names = [s.name for s in detected]
        print(f"Detected {len(detected)} source(s) in {args.path}:")
        for s_ in detected:
            print(f"  {s_.name:32s} {s_.type}")
        print()
    elif args.all:
        names = [s.name for s in config.sources]
    elif args.source:
        names = args.source
    else:
        parser.error("specify --source NAME (repeatable), --all, or --path DIR")

    if not names:
        print("No sources configured in corpus.toml. Add a [[sources]] block.")
        return 1

    ingester = Ingester(config)
    # 0 clean, 1 a source failed outright, EXIT_ANOMALY completed but
    # something needs a human. A warning nobody reads is theatre for an
    # unattended run: cron and CI see an exit code, not stderr. Kept distinct
    # from 1 so a caller can tell "this did not work" from "this worked and
    # you should look at it", and so an existing script checking `!= 0` still
    # notices either.
    exit_code = 0
    try:
        for name in names:
            print(f"=== Ingesting {name} ===")
            try:
                r = ingester.ingest(name, prune_anyway=args.prune_anyway)
            except (ValueError, OSError, ImportError) as e:
                # A source that cannot be enumerated (missing directory,
                # unmounted volume — FileNotFoundError is an OSError subclass)
                # or whose connector's optional extra isn't installed
                # (ImportError, e.g. `pip install 'corpus-rag[docx]'`) fails
                # THIS source only; --all continues. Orphan pruning never ran
                # for it, so its indexed content is left intact rather than
                # deleted.
                print(f"  ERROR: {e}  (index left intact; skipping this source)")
                exit_code = 1
                continue
            print(f"  documents:        {r.documents:,}")
            print(f"  chunks seen:      {r.chunks_seen:,}")
            print(f"  chunks upserted:  {r.chunks_upserted:,}")
            print(f"  chunks unchanged: {r.chunks_skipped:,}")
            anomalies = [
                d
                for d in (
                    r.yield_drop_detail,
                    r.path_change_detail,
                    r.skip_rise_detail,
                    # Documents disappearing is an anomaly in its own right,
                    # not merely colour on another warning: the two shapes it
                    # exists to catch -- a drop under the yield ratio, and
                    # one-for-one substitution -- fire NO other check, so
                    # leaving it out here would report them and still exit 0.
                    r.vanished_detail,
                )
                if d
            ]
            # --prune-anyway IS the acknowledgement: the operator looked at
            # the warnings and re-ran to accept them. Exiting anomalous after
            # that would leave no way to clear the signal, which trains people
            # to ignore it.
            if anomalies and exit_code == 0 and not args.prune_anyway:
                exit_code = EXIT_ANOMALY
            if r.skip_rise_detail:
                print(f"  WARNING: {r.skip_rise_detail}")
            if r.path_change_detail:
                print(f"  WARNING: {r.path_change_detail}")
            if r.yield_drop_detail:
                # Printed, not just logged: the whole point is that this run
                # otherwise looks completely normal.
                print(f"  WARNING: {r.yield_drop_detail}")
            if r.vanished_detail:
                # The names behind whichever warning fired. Without these the
                # operator is told a magnitude and left to diff directory
                # listings by hand to find out what actually moved.
                print(f"  WARNING: {r.vanished_detail}")
            if r.files_skipped:
                print(f"  files skipped:    {r.files_skipped:,}  (unsupported/unreadable by design)")
            if r.prune_refused:
                # A refusal is a failure the operator must see and act on, not
                # just a note in the log — same reasoning as the exit_code=1
                # branch below for a source that couldn't be enumerated.
                print("  orphans deleted:  0  (pruning REFUSED — blast-radius guard tripped)")
                print(f"  {r.prune_refused_detail}")
                exit_code = 1
            elif r.pruning_performed:
                print(f"  orphans deleted:  {r.orphans_deleted:,}")
            else:
                print(f"  files unreadable: {r.files_failed:,}")
                print("  orphans deleted:  0  (pruning SKIPPED — see warnings above)")
            _print_tokens(tokens_used=r.tokens_used, counted=r.tokens_counted)
            print(f"  elapsed:          {r.elapsed_seconds:.1f}s")
            print()
        if exit_code == EXIT_ANOMALY:
            print(
                "Completed with warnings above. Re-run the affected source with "
                "--prune-anyway once you have checked them, which accepts the "
                f"new state and clears this (exit {EXIT_ANOMALY})."
            )
        return exit_code
    finally:
        ingester.close()


if __name__ == "__main__":
    sys.exit(main())
