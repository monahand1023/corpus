"""corpus-list: enumerate configured sources and their ingested chunk counts.

Usage:
  corpus-list
  corpus-list --config path/to/corpus.toml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from corpus.config import CorpusConfig
from corpus.db.sqlite import ChunkStore


def main() -> int:
    parser = argparse.ArgumentParser(description="List configured sources + chunk counts")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    config = CorpusConfig.load(args.config)

    if not config.sources:
        print("No [[sources]] configured in corpus.toml.")
        return 1

    # Chunk counts are only available if the DB exists.
    counts: dict[str, int] = {}
    db_exists = Path(config.db_path).exists()
    if db_exists:
        store = ChunkStore(config.db_path, embedding_dim=config.embedder.dim)
        try:
            counts = store.stats()["by_source"]
        finally:
            store.close()

    print(f"corpus.toml: {len(config.sources)} source(s)")
    print(f"database:    {config.db_path}" + ("" if db_exists else "  (not yet created)"))
    print()
    print(f"  {'name':<20} {'type':<10} {'chunks':>10}  path")
    print(f"  {'-' * 20} {'-' * 10} {'-' * 10}  {'-' * 30}")
    for s in config.sources:
        n = counts.get(s.name)
        n_str = f"{n:,}" if n is not None else ("0" if db_exists else "—")
        print(f"  {s.name:<20} {s.type:<10} {n_str:>10}  {s.path}")

    # Surface anything in the DB that ISN'T in the current config — likely a
    # renamed/removed source whose chunks are now orphaned.
    if db_exists:
        configured = {s.name for s in config.sources}
        stale = sorted(set(counts) - configured)
        if stale:
            print()
            print("  Chunks in DB with no matching [[sources]] entry (stale):")
            for name in stale:
                print(f"    {name}: {counts[name]:,} chunks  — `corpus-reset --source {name}` to drop")
    return 0


if __name__ == "__main__":
    sys.exit(main())
