"""corpus-rename: change a source's name without re-ingesting it.

`source_type` is the first field of every chunk id, so a rename done with an
UPDATE statement leaves ids that no longer match what the next ingest
computes -- and that ingest silently inserts a second copy of the whole
source at full embedding cost. See `corpus.rename` for the details.

    corpus-rename old_name new_name --dry-run
    corpus-rename old_name new_name

Remember to change the `[[sources]]` name in corpus.toml to match, or the
next ingest will treat the renamed chunks as orphans.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from corpus.cli._common import configure_logging, load_config_or_exit
from corpus.rename import SourceNotFound, TargetExists, rename_source


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Rename an indexed source in place (no re-embedding)"
    )
    parser.add_argument("old", help="Current source name")
    parser.add_argument("new", help="New source name")
    parser.add_argument("--config", default=None, help="Path to corpus.toml")
    parser.add_argument(
        "--dry-run", action="store_true", help="Report what would move, change nothing"
    )
    parser.add_argument("--yes", action="store_true", help="Skip the confirmation")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    configure_logging(args.verbose)
    config = load_config_or_exit(args.config)
    db_path = Path(config.db_path)
    if not db_path.exists():
        print(f"error: database not found: {db_path}", file=sys.stderr)
        return 1

    import sqlite3

    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        count = conn.execute(
            "SELECT count(*) FROM chunks WHERE source_type = ?", (args.old,)
        ).fetchone()[0]
    finally:
        conn.close()

    print(f"corpus-rename: {db_path}")
    print(f"  {args.old!r} -> {args.new!r}: {count:,} chunk(s)")
    print("  Vectors and BM25 rows are keyed by rowid and are not rebuilt, so")
    print("  this costs no embedding spend.")

    if args.dry_run:
        print("\n(dry run — nothing changed)")
        return 0
    if not count:
        print(f"\nerror: no chunks with source_type={args.old!r}", file=sys.stderr)
        return 1

    if not args.yes:
        answer = input(f"\nRename {count:,} chunk(s)? [y/N]: ")
        if answer.strip().lower() not in ("y", "yes"):
            print("Nothing was changed.")
            return 1

    try:
        moved = rename_source(db_path, args.old, args.new)
    except SourceNotFound as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except TargetExists as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(f"renamed {moved:,} chunk(s).")
    print(
        f"\nNow change the [[sources]] entry named {args.old!r} to {args.new!r} in "
        "corpus.toml.\nUntil you do, the next ingest sees the renamed chunks as "
        "orphans."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
