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

from corpus.config import CorpusConfig
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
    parser.add_argument("--config", default=None, help="Path to corpus.toml (default: ./corpus.toml)")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    config = CorpusConfig.load(args.config)
    if args.all:
        names = [s.name for s in config.sources]
    elif args.source:
        names = args.source
    else:
        parser.error("specify --source NAME (repeatable) or --all")
        return 2  # unreachable

    if not names:
        print("No sources configured in corpus.toml. Add a [[sources]] block.")
        return 1

    ingester = Ingester(config)
    exit_code = 0
    try:
        for name in names:
            print(f"=== Ingesting {name} ===")
            try:
                r = ingester.ingest(name)
            except (ValueError, FileNotFoundError) as e:
                print(f"  ERROR: {e}")
                exit_code = 1
                continue
            print(f"  documents:        {r.documents:,}")
            print(f"  chunks seen:      {r.chunks_seen:,}")
            print(f"  chunks upserted:  {r.chunks_upserted:,}")
            print(f"  chunks unchanged: {r.chunks_skipped:,}")
            print(f"  orphans deleted:  {r.orphans_deleted:,}")
            print(f"  tokens billed:    {r.tokens_used:,}")
            print(f"  elapsed:          {r.elapsed_seconds:.1f}s")
            print()
        return exit_code
    finally:
        ingester.close()


if __name__ == "__main__":
    sys.exit(main())
