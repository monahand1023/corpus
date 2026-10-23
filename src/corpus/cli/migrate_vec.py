"""corpus-migrate-vec: partition a large store's vector index deliberately.

`chunks_vec` is partitioned by source type so a source-filtered search scans
only that type's vectors. A small store migrates itself when opened; a large
one does not, because the rebuild is one transaction holding SQLite's only
writer lock throughout. See `ChunkStore._migrate_vec`.

    corpus-migrate-vec --config corpus.toml --dry-run
    corpus-migrate-vec --config corpus.toml

Nothing is re-embedded and no API is called: the stored vectors are copied
out and back in with each chunk's type.

**Deploy the current code everywhere before running this.** An older writer
files new vectors with no type, and they then drop out of every filtered
search until current code next opens the store read-write and re-files them.

Interruption is safe. The rebuild is one transaction, so a killed process
rolls back to the old layout.
"""

from __future__ import annotations

import argparse
import sys
import time

from corpus.cli._common import configure_logging, load_config_or_exit
from corpus.db.sqlite import ChunkStore


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Partition the vector index by source type for fast filtered search"
    )
    parser.add_argument("--config", default=None, help="Path to corpus.toml")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report whether a migration is needed and how large it would be, then stop.",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    configure_logging(args.verbose)
    config = load_config_or_exit(args.config)

    if not config.db_path.exists():
        print(f"error: database not found: {config.db_path}", file=sys.stderr)
        return 1

    # Read-only first: a read-write open is what migrates a small store, so a
    # dry run must not be able to cause the migration it reports on.
    probe = ChunkStore(config.db_path, embedding_dim=config.embedder.dim, read_only=True)
    try:
        partitioned = probe.vec_partitioned()
        chunks = probe.stats()["total"]
    finally:
        probe.close()

    print(f"database:      {config.db_path}")
    print(f"chunks:        {chunks:,}")
    print(f"partitioned:   {'yes' if partitioned else 'no'}")

    if partitioned:
        print("\nVector index is already partitioned. Nothing to do.")
        return 0

    print("\nA migration is needed. It copies the stored vectors -- nothing is")
    print("re-embedded and no API is called.")
    if args.dry_run:
        print("\n(dry run -- nothing migrated)")
        return 0

    print("\nBefore continuing, make sure no other process is writing to this")
    print("database: the migration holds SQLite's only writer lock throughout.")
    print("Interrupting this is safe -- it rolls back to the current layout.\n")

    start = time.monotonic()
    store = ChunkStore(
        config.db_path, embedding_dim=config.embedder.dim, allow_expensive_migration=True
    )
    try:
        after = store.vec_partitioned()
    finally:
        store.close()

    if not after:
        print("ERROR: migration did not complete (vector index is still unpartitioned).")
        return 1
    print(f"Partitioned {chunks:,} chunk(s) in {time.monotonic() - start:.1f}s.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
