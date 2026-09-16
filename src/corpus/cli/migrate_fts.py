"""corpus-migrate-fts: rebuild a large store's full-text index deliberately.

A small store rebuilds itself when opened. A large one does not, because the
rebuild is a single transaction holding SQLite's only writer lock from first
delete to final commit — minutes, on a million-chunk store, during which any
concurrent ingest fails on its busy timeout. That is a maintenance operation,
and a library constructor is not the place for one. See
`ChunkStore._migrate_fts`.

    corpus-migrate-fts --config corpus.toml --dry-run
    corpus-migrate-fts --config corpus.toml

**Deploy the current code everywhere before running this.** An older writer
appending to a freshly migrated index writes unnormalized rows into a store
stamped as current — the migration reports success and the defect comes back
silently for everything written afterwards. Nothing in SQLite can catch that;
it is an ordering you have to get right operationally.

Interruption is safe. The rebuild is one transaction, so a killed process
rolls back to the old index and the old version stamp, and the next attempt
starts clean. There is no durable half-migrated state.
"""

from __future__ import annotations

import argparse
import sys
import time

from corpus.cli._common import configure_logging, load_config_or_exit
from corpus.db.sqlite import FTS_VERSION, ChunkStore


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Rebuild the full-text index after a normalization change"
    )
    parser.add_argument("--config", default=None, help="Path to corpus.toml")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report whether a rebuild is needed and how large it would be, then stop.",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    configure_logging(args.verbose)
    config = load_config_or_exit(args.config)

    # A project that has not indexed anything yet has no store to migrate.
    # Without this the read-only open below raises FileNotFoundError straight
    # out of the CLI: a traceback for an ordinary mistake, where every other
    # command that opens the store by path says it in one line.
    if not config.db_path.exists():
        print(f"error: database not found: {config.db_path}", file=sys.stderr)
        return 1

    # Read-only first: opening read-write is what would trigger the very
    # migration being reported on, so a dry run must not be able to cause one.
    probe = ChunkStore(
        config.db_path, embedding_dim=config.embedder.dim, read_only=True
    )
    try:
        stamped = probe.fts_version()
        chunks = probe.stats()["total"]
    finally:
        probe.close()

    print(f"database:      {config.db_path}")
    print(f"chunks:        {chunks:,}")
    print(f"fts_version:   {stamped!r} (current: {FTS_VERSION!r})")

    if stamped == FTS_VERSION:
        print("\nIndex is current. Nothing to do.")
        return 0

    print("\nA rebuild is needed. It re-reads text already in the database —")
    print("nothing is re-embedded and no API is called.")
    if args.dry_run:
        print("\n(dry run — nothing rebuilt)")
        return 0

    print("\nBefore continuing, make sure no other process is writing to this")
    print("database: the rebuild holds SQLite's only writer lock throughout,")
    print("and a concurrent ingest will fail on its busy timeout.")
    print("Interrupting this is safe — it rolls back to the current index.\n")

    start = time.monotonic()
    store = ChunkStore(
        config.db_path,
        embedding_dim=config.embedder.dim,
        allow_expensive_migration=True,
    )
    try:
        after = store.fts_version()
    finally:
        store.close()

    if after != FTS_VERSION:
        print(f"ERROR: rebuild did not complete (fts_version is still {after!r}).")
        return 1
    print(f"Rebuilt {chunks:,} chunk(s) in {time.monotonic() - start:.1f}s.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
