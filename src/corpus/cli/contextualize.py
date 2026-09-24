"""corpus-contextualize: add a Contextual-Retrieval blurb to indexed chunks.

Usage:
  corpus-contextualize --source notes --dry-run
  corpus-contextualize --source notes --source papers
  corpus-contextualize --all              # every source with contextualize = true
  corpus-contextualize --source notes --clear

Each chunk is sent to a cheap model along with its parent document, which
writes one sentence situating it; that sentence is stored and the chunk is
re-embedded as context + content. See `corpus.contextual` for why this helps
and what it costs.

`--dry-run` prices the run without submitting anything, and is the intended
first step: this spends real money per chunk, and the estimate is the only
warning before it does.
"""

from __future__ import annotations

import argparse
import sys
import time

from corpus.cli._common import configure_logging, load_config_or_exit
from corpus.config import CorpusConfig
from corpus.contextual.batch_runner import BatchContextualizer
from corpus.contextual.contextualizer import (
    estimate_chunk_cost_tokens,
    should_contextualize,
)
from corpus.credentials import resolve_dotenv
from corpus.db.sqlite import ChunkStore
from corpus.embedder.factory import make_embedder
from corpus.util.priority import DEFAULT_NICE, be_nice


def _min_tokens_for(config: CorpusConfig) -> dict[str, int]:
    """Each source's floor: its own `context_min_tokens` (0 included), else the global one."""
    return {
        s.name: (
            s.context_min_tokens
            if s.context_min_tokens is not None
            else config.contextual.min_tokens
        )
        for s in config.sources
    }


def _estimate(store: ChunkStore, source: str, min_tokens: int) -> tuple[int, int, int]:
    """(eligible chunks, uncached input tokens, output tokens) for one source.

    Deliberately counts the parent document ONCE per document rather than
    once per chunk: the document is sent as a cached prefix, so per-chunk
    accounting would overstate a big document's cost by roughly its chunk
    count. This is an estimate over the chars/4 heuristic — non-English text
    runs materially higher — and the CLI says so rather than presenting it as
    a quote.
    """
    chunks = store.chunks_missing_context(source)
    eligible = [
        c for c in chunks if should_contextualize((c.metadata or {}).get("token_count"), min_tokens)
    ]
    if not eligible:
        return 0, 0, 0
    seen_docs: set[str] = set()
    in_tokens = out_tokens = 0
    for chunk in eligible:
        doc_chars = 0
        if chunk.source_key not in seen_docs:
            seen_docs.add(chunk.source_key)
            doc_chars = len(store.doc_body(source, chunk.source_key))
        chunk_in, chunk_out = estimate_chunk_cost_tokens(doc_chars, len(chunk.content))
        in_tokens += chunk_in
        out_tokens += chunk_out
    return len(eligible), in_tokens, out_tokens


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Add Contextual-Retrieval context to indexed chunks"
    )
    parser.add_argument("--source", action="append", default=None, help="Source name (repeatable)")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Every source with contextualize = true in corpus.toml",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Price the run and stop. Submits nothing, spends nothing.",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help=(
            "Delete stored contexts for the named sources and restore their "
            "plain-text FTS rows. Does NOT restore the pre-context embeddings "
            "— those were overwritten in place and need a re-ingest."
        ),
    )
    parser.add_argument("--config", default=None, help="Path to corpus.toml")
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

    if args.source and args.all:
        print("--source cannot be combined with --all.")
        return 2
    if args.all:
        names = [s.name for s in config.sources if s.contextualize]
        if not names:
            print(
                "No sources have contextualize = true in corpus.toml. "
                "Add it to the [[sources]] blocks you want contextualized, "
                "or name them with --source."
            )
            return 1
    elif args.source:
        names = args.source
    else:
        parser.error("specify --source NAME (repeatable) or --all")

    store = ChunkStore.from_config(config)
    try:
        if args.clear:
            for name in names:
                cleared = store.clear_context(name)
                print(f"{name}: cleared {cleared:,} contexts")
            print(
                "\nEmbeddings still reflect the cleared contexts — re-ingest "
                "or re-contextualize these sources to rebuild them."
            )
            return 0

        min_tokens_for = _min_tokens_for(config)

        if args.dry_run:
            print(f"{'source':<28}{'chunks':>10}{'in (Mtok)':>12}{'out (Mtok)':>12}")
            total_in = total_out = total_chunks = 0
            for name in names:
                n, tin, tout = _estimate(
                    store, name, min_tokens_for.get(name, config.contextual.min_tokens)
                )
                print(f"{name:<28}{n:>10,}{tin/1e6:>12.2f}{tout/1e6:>12.2f}")
                total_chunks += n
                total_in += tin
                total_out += tout
            print(f"{'TOTAL':<28}{total_chunks:>10,}{total_in/1e6:>12.2f}{total_out/1e6:>12.2f}")
            print(
                "\nInput tokens count each parent document once (it is sent as a "
                f"cached prefix, billed at ~10% on re-reads). Model: "
                f"{config.contextual.model}. Chunks are also RE-EMBEDDED after "
                "contextualization, which your embedding provider bills separately. "
                "Estimated with the chars/4 heuristic — non-English text runs "
                "materially higher. Check current provider pricing before a large run."
            )
            print("(dry run — nothing submitted)")
            return 0

        embedder = make_embedder(
            provider=config.embedder.provider,
            model=config.embedder.model,
            dim=config.embedder.dim,
        )
        start = time.monotonic()
        total = 0
        for name in names:
            runner = BatchContextualizer(
                store=store,
                embedder=embedder,
                window_size=config.contextual.window_size,
                min_tokens=min_tokens_for.get(name, config.contextual.min_tokens),
                state_path=config.db_path.parent / f".contextualize-state-{name}.json",
            )
            print(f"=== Contextualizing {name} ===")
            def doc_body_fn(source_key: str, source: str = name) -> str:
                return store.doc_body(source, source_key)

            result = runner.run(name, doc_body_fn=doc_body_fn)
            print(f"  batch requests:        {result.requests_submitted:,}")
            print(f"  chunks contextualized: {result.chunks_contextualized:,}")
            if result.skipped_ineligible:
                print(f"  skipped (too short):   {result.skipped_ineligible:,}")
            if result.errors:
                print(f"  errors:                {result.errors:,}")
            total += result.chunks_contextualized

        print(f"\nTotal contextualized: {total:,} in {time.monotonic() - start:.1f}s")
        return 0
    finally:
        store.close()


if __name__ == "__main__":
    sys.exit(main())
