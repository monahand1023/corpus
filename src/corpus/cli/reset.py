"""corpus-reset: drop a single source's chunks, or delete the whole DB.

Usage:
  corpus-reset --source notes        # drop just the 'notes' source's chunks
  corpus-reset --all                 # delete the entire DB file
  corpus-reset --all --yes           # skip the confirmation prompt

`--source` is the surgical option — useful when you change a connector's
chunking logic and want to re-ingest just that source cleanly. `--all` is
the nuclear option, needed when you change embedder dim (the schema is
fixed-width and can't be migrated in place).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from corpus.config import CorpusConfig
from corpus.db.sqlite import ChunkStore


def main() -> int:
    parser = argparse.ArgumentParser(description="Drop a source's chunks or delete the DB")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--source", help="Drop only this source type's chunks")
    group.add_argument("--all", action="store_true", help="Delete the entire DB file")
    parser.add_argument("--yes", action="store_true", help="Skip confirmation prompt")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    config = CorpusConfig.load(args.config)
    db_path = Path(config.db_path)

    if args.all:
        if not db_path.exists():
            print(f"No DB at {db_path} — nothing to delete.")
            return 0
        if not args.yes:
            resp = input(f"Delete the entire database at {db_path}? [y/N] ").strip().lower()
            if resp != "y":
                print("Aborted.")
                return 1
        # Remove the DB plus any WAL/SHM sidecar files.
        for suffix in ("", "-journal", "-wal", "-shm"):
            sidecar = Path(str(db_path) + suffix)
            if sidecar.exists():
                sidecar.unlink()
        print(f"Deleted {db_path} (and any WAL/SHM sidecars).")
        return 0

    # --source path
    if not db_path.exists():
        print(f"No DB at {db_path} — nothing to drop.")
        return 0

    store = ChunkStore(config.db_path, embedding_dim=config.embedder.dim)
    try:
        before = store.stats()["by_source"].get(args.source, 0)
        if before == 0:
            print(f"Source '{args.source}' has no chunks in the DB.")
            return 0
        if not args.yes:
            resp = input(f"Drop {before:,} chunks for source '{args.source}'? [y/N] ").strip().lower()
            if resp != "y":
                print("Aborted.")
                return 1
        deleted = store.delete_by_source(args.source)
        print(f"Dropped {deleted:,} chunks for source '{args.source}'.")
        print(f"Re-ingest with: corpus-ingest --source {args.source}")
        return 0
    finally:
        store.close()


if __name__ == "__main__":
    sys.exit(main())
