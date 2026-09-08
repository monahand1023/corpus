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
from corpus.ingester import Ingester

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
        "--prune-anyway",
        action="store_true",
        help=(
            "Delete orphaned chunks even when the connector reported files it "
            "could not read. Requires --source; refused with --all."
        ),
    )
    parser.add_argument("--config", default=None, help="Path to corpus.toml (default: ./corpus.toml)")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

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
    if args.all:
        names = [s.name for s in config.sources]
    elif args.source:
        names = args.source
    else:
        parser.error("specify --source NAME (repeatable) or --all")

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
            if r.pruning_performed:
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
