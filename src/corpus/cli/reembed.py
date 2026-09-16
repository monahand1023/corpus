"""corpus-reembed: change embedding provider, model or dim in place.

WHY THIS COMMAND EXISTS. There was no other way. A re-ingest re-embeds
nothing -- `ChunkStore.upsert` returns early when the content hash is
unchanged -- and the dim guard's error says "Re-ingest from scratch with the
new dim", which means `corpus-reset --all`. That destroys everything stored
only in the database: contextualization blurbs and per-document summaries,
both of them Anthropic spend with no on-disk backup, in the hundreds of
thousands on a real archive.

Only the VECTORS change here. `chunks`, `chunks_fts`, `summaries`, every
context blurb and all metadata are left exactly as they are.

    corpus-reembed --dry-run        # what it would cost, spends nothing
    corpus-reembed                  # asks first
    corpus-reembed --stage-only     # buy the embeddings, swap later

THE TWO PHASES ARE SEPARABLE ON PURPOSE. Staging is the long, billed pass and
is resumable to the batch. The swap is one transaction that cannot run until
every chunk is staged, because a half-swapped vector table is an archive that
returns wrong neighbours and says nothing about it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from corpus.cli._common import configure_logging, load_config_or_exit
from corpus.credentials import resolve_dotenv
from corpus.reembed import (
    ReembedIncomplete,
    reembed_plan,
    stage_embeddings,
    staged_count,
    swap_in_staged,
)
from corpus.util.priority import DEFAULT_NICE, be_nice


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Re-embed an archive in place, keeping context and summaries"
    )
    parser.add_argument("--config", default=None, help="Path to corpus.toml")
    parser.add_argument(
        "--dry-run", action="store_true", help="Price it; write nothing"
    )
    parser.add_argument(
        "--stage-only",
        action="store_true",
        help="Buy the embeddings but do not swap them in yet",
    )
    parser.add_argument(
        "--swap-only",
        action="store_true",
        help="Swap in embeddings staged by an earlier run",
    )
    parser.add_argument("--yes", action="store_true", help="Skip the confirmation")
    parser.add_argument("--batch-size", type=int, default=128)
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

    resolve_dotenv(args.config)
    configure_logging(args.verbose)
    config = load_config_or_exit(args.config)

    db_path = Path(config.db_path)
    if not db_path.exists():
        print(f"error: database not found: {db_path}", file=sys.stderr)
        print("There is nothing to re-embed yet.", file=sys.stderr)
        return 1

    new_dim = config.embedder.dim
    plan = reembed_plan(db_path, new_dim=new_dim)
    print(f"corpus-reembed: {db_path}")
    print(f"provider           : {config.embedder.provider} "
          f"{config.embedder.model or ''}".rstrip())
    print(plan.describe())
    if not plan.chunks:
        print("\nNothing to do.")
        return 0

    if args.dry_run:
        print("\n(dry run — nothing embedded, nothing changed)")
        return 0

    if args.swap_only:
        return _swap(db_path, new_dim)

    if not args.yes:
        # An irreversible, billed operation must not start because someone
        # pressed enter. The estimate above is the only warning.
        answer = input("\nRe-embed? This calls your embedding provider [y/N]: ")
        if answer.strip().lower() not in ("y", "yes"):
            print("Nothing was embedded.")
            return 1

    from corpus.embedder.factory import make_embedder

    embedder = make_embedder(
        provider=config.embedder.provider,
        model=config.embedder.model,
        dim=new_dim,
    )

    def progress(done: int, total: int) -> None:
        # flush: on a run this long the only sign of life is this line, and
        # Python block-buffers stdout when it is redirected to a log.
        print(f"  staged {done:,}/{total:,}", flush=True)

    added = stage_embeddings(
        db_path,
        new_dim=new_dim,
        embed=embedder.embed_documents,
        batch_size=args.batch_size,
        on_progress=progress,
    )
    print(f"\nstaged {added:,} new embedding(s); "
          f"{staged_count(db_path, new_dim=new_dim):,} ready")

    if args.stage_only:
        print("\n(--stage-only — the archive is unchanged; "
              "re-run with --swap-only to apply)")
        return 0
    return _swap(db_path, new_dim)


def _swap(db_path: Path, new_dim: int) -> int:
    try:
        moved = swap_in_staged(db_path, new_dim=new_dim)
    except ReembedIncomplete as exc:
        print(f"\nerror: {exc}", file=sys.stderr)
        return 1
    print(f"swapped in {moved:,} vector(s); embedding_dim is now {new_dim}")
    print("Context blurbs, summaries and BM25 rows were not touched.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
