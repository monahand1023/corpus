"""corpus-ingest: load configured sources into the chunk store.

Usage:
  corpus-ingest --source notes
  corpus-ingest --source notes --source papers
  corpus-ingest --all
"""

from __future__ import annotations

import argparse
import logging
import sys

from dotenv import load_dotenv

from corpus.cli._common import load_config_or_exit
from corpus.connectors.registry import DEFAULT_GLOBS
from corpus.ingester import Ingester
from corpus.util.autodetect import detect_sources

load_dotenv()


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
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

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
            print(f"  tokens billed:    {r.tokens_used:,}")
            print(f"  elapsed:          {r.elapsed_seconds:.1f}s")
            print()
        return exit_code
    finally:
        ingester.close()


if __name__ == "__main__":
    sys.exit(main())
