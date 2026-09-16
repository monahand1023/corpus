"""corpus-summarize: pre-compute per-doc summaries via Claude Haiku.

Usage:
  corpus-summarize --source notes --dry-run    # show cost estimate
  corpus-summarize --source notes -v           # run it
  corpus-summarize --all --concurrency 16      # all sources, faster

Requires `pip install corpus-rag[summarizer]` (or `uv add anthropic` if
working from the template) and ANTHROPIC_API_KEY in your environment.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING

from corpus.cli._common import (
    configure_logging,
    load_config_or_exit,
    open_store_read_only,
)
from corpus.credentials import resolve_dotenv
from corpus.db.sqlite import ChunkStore, StoredChunk

if TYPE_CHECKING:
    from corpus.summarizer.anthropic_summarizer import SummaryResult

# Haiku 4.5 pricing as of 2026, per token. Check current published pricing
# before a large run -- these are hardcoded and will drift.
PRICE_INPUT = 1.0 / 1_000_000
PRICE_OUTPUT = 5.0 / 1_000_000
PRICE_CACHED = 0.10 / 1_000_000

# What the summarizer sends alongside every document, regardless of its size:
# SYSTEM_PROMPT + the per-source guidance + the title and framing around the
# body. Measured from `AnthropicSummarizer.summarize`, chars/4. An archive of
# many small documents is mostly this, so leaving it out of an estimate
# understates exactly the runs where it matters most.
PROMPT_OVERHEAD_TOKENS = 260

# `max_tokens=400`, and the prompt asks for ~120 words, hard cap 200.
EST_OUTPUT_TOKENS_PER_DOC = 180


def run_cost(*, input_tokens: int, output_tokens: int, cached_tokens: int) -> float:
    """What a completed run cost, in dollars.

    The three counts are DISJOINT, which is the part this got wrong. The
    Anthropic API reports `input_tokens` as the uncached input only, with
    `cache_read_input_tokens` beside it -- not inside it. Treating the cached
    count as a subset and subtracting a discount for it scored every cached
    token as a ~90-cent refund per million instead of a 10-cent charge, so a
    run with a working prompt cache reported a cost below the true one and,
    with enough hits, below zero.
    """
    return (
        input_tokens * PRICE_INPUT
        + cached_tokens * PRICE_CACHED
        + output_tokens * PRICE_OUTPUT
    )


def estimate_doc_tokens(*, body_chars: int) -> tuple[int, int]:
    """Rough (input, output) tokens for summarizing one document.

    Capped at `MAX_INPUT_CHARS` because that is what `summarize()` actually
    sends: a 5 MB document is one 20k-token request, not a 1.25M-token one.
    Estimating the untruncated length priced documents that can never be sent
    in full, which on an archive with a few huge files dominated the total.

    The chars/4 heuristic, same as `corpus.util.tokens` -- the real tokenizer
    shifts individual requests by ~10% and non-English text considerably
    more, so a caller reporting this should say so rather than quote it.
    """
    from corpus.summarizer.anthropic_summarizer import MAX_INPUT_CHARS

    body = min(body_chars, MAX_INPUT_CHARS)
    return body // 4 + PROMPT_OVERHEAD_TOKENS, EST_OUTPUT_TOKENS_PER_DOC


def _reconstruct_doc(chunks: list[StoredChunk]) -> tuple[str, str]:
    title = chunks[0].title or chunks[0].source_key
    return title, "\n\n".join(c.content for c in chunks)


def main() -> int:
    parser = argparse.ArgumentParser(description="Per-doc summarization via Claude Haiku")
    parser.add_argument("--source", action="append", default=None, help="Source name (repeatable)")
    parser.add_argument("--all", action="store_true", help="Summarize all configured sources")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--config", default=None)
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    # Resolve credentials now that --config is known (env var > .env beside
    # --config > .env in cwd). See corpus.credentials for the precedence.
    resolve_dotenv(args.config)

    configure_logging(args.verbose)

    config = load_config_or_exit(args.config)
    if args.all:
        source_names = [s.name for s in config.sources]
    elif args.source:
        source_names = args.source
    else:
        parser.error("specify --source NAME (repeatable) or --all")

    # A dry run is the "what would this cost me" path, so it must not be able
    # to change anything -- including the two things a read-write open does
    # silently: create a database that is not there (turning a typo in
    # db_path into "0 docs, $0.00", read as "nothing to do") and rebuild a
    # stale FTS index from the store's constructor.
    if args.dry_run:
        store = open_store_read_only(config)
    else:
        if not config.db_path.exists():
            print(f"error: database not found: {config.db_path}", file=sys.stderr)
            print("Run `corpus-index` first.", file=sys.stderr)
            return 1
        store = ChunkStore(
            config.db_path,
            embedding_dim=config.embedder.dim,
            cache_size_mb=config.performance.cache_size_mb,
            mmap_size_mb=config.performance.mmap_size_mb,
            temp_store_memory=config.performance.temp_store_memory,
        )

    summarizer = None
    if not args.dry_run:
        from corpus.summarizer.anthropic_summarizer import (
            DEFAULT_MODEL,
            MAX_INPUT_CHARS,
            AnthropicSummarizer,
            doc_hash,
        )
        summarizer = AnthropicSummarizer()
    else:
        from corpus.summarizer.anthropic_summarizer import (
            DEFAULT_MODEL,
            MAX_INPUT_CHARS,
            doc_hash,
        )

    exit_code = 0
    grand = {"docs": 0, "failed": 0, "input": 0, "output": 0, "cached": 0}
    for name in source_names:
        keys = store.list_source_keys(name)
        if args.limit:
            keys = keys[: args.limit]
        known = store.known_summary_hashes(name)

        to_do: list[tuple[str, str, str, str]] = []
        skipped = 0
        for key in keys:
            chunks = store.get_by_source_key(name, key)
            if not chunks:
                continue
            title, body = _reconstruct_doc(chunks)
            h = doc_hash(body)
            if known.get(key) == h:
                skipped += 1
                continue
            to_do.append((key, title, body, h))

        print(f"=== {name} ===")
        print(f"  total docs: {len(keys):,}")
        print(f"  already summarized: {skipped:,}")
        print(f"  to summarize: {len(to_do):,}")

        if args.dry_run:
            est_input = est_output = 0
            for _, _, body, _ in to_do:
                doc_in, doc_out = estimate_doc_tokens(body_chars=len(body))
                est_input += doc_in
                est_output += doc_out
            # Priced as if nothing is cached. The cache only makes it cheaper,
            # and an estimate that assumes a hit rate it cannot know would err
            # toward encouraging the spend.
            est_cost = run_cost(
                input_tokens=est_input, output_tokens=est_output, cached_tokens=0
            )
            print(f"  est. input tokens: {est_input:,}")
            print(f"  est. output tokens: {est_output:,}")
            print(f"  est. cost: ${est_cost:.2f}")
            grand["docs"] += len(to_do)
            grand["input"] += est_input
            grand["output"] += est_output
            continue

        assert summarizer is not None
        start = time.monotonic()
        totals = {"input": 0, "output": 0, "cached": 0}

        def _do_one(
            item: tuple[str, str, str, str],
            source_name: str = name,
        ) -> tuple[str, str, SummaryResult | None, Exception | None]:
            key, title, body, h = item
            try:
                return key, h, summarizer.summarize(source_name, title, body), None
            except Exception as e:
                return key, h, None, e

        completed = 0
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = {pool.submit(_do_one, item): item for item in to_do}
            for fut in as_completed(futures):
                key, h, result, err = fut.result()
                completed += 1
                if err is not None or result is None:
                    logging.error("summarize %s:%s failed: %s", name, key, err)
                    exit_code = 1
                    grand["failed"] += 1
                    continue
                store.upsert_summary(
                    source_type=name,
                    source_key=key,
                    summary=result.summary,
                    doc_hash=h,
                    model=DEFAULT_MODEL,
                    token_count=result.input_tokens + result.output_tokens,
                )
                totals["input"] += result.input_tokens
                totals["output"] += result.output_tokens
                totals["cached"] += result.cached_input_tokens
                if completed % 50 == 0:
                    elapsed = time.monotonic() - start
                    rate = completed / elapsed if elapsed > 0 else 0
                    print(f"  ...summarized {completed}/{len(to_do)} ({rate:.1f}/s)")

        elapsed = time.monotonic() - start
        cost = run_cost(
            input_tokens=totals["input"],
            output_tokens=totals["output"],
            cached_tokens=totals["cached"],
        )
        print(
            f"  done in {elapsed:.0f}s. tokens: in={totals['input']:,} "
            f"out={totals['output']:,} cached={totals['cached']:,}. cost: ${cost:.2f}"
        )
        for k in ("input", "output", "cached"):
            grand[k] += totals[k]
        grand["docs"] += len(to_do)

    total_cost = run_cost(
        input_tokens=grand["input"],
        output_tokens=grand["output"],
        cached_tokens=grand["cached"],
    )
    if args.dry_run:
        # `--all --dry-run` over 44 sources printed 44 numbers and no sum, on
        # the one path whose entire job is answering "how much".
        print(f"\nTOTAL (estimated): {grand['docs']:,} docs, ${total_cost:.2f}")
        print(
            "Estimated with the chars/4 heuristic and priced as if nothing is "
            "cached -- non-English text runs materially higher, and the prompt "
            "cache only makes it cheaper. Documents are truncated at "
            f"{MAX_INPUT_CHARS:,} characters. Check current model pricing "
            "before a large run."
        )
        print("(dry run -- nothing submitted)")
    else:
        failed = f", {grand['failed']:,} failed" if grand["failed"] else ""
        print(
            f"\nTOTAL: {grand['docs'] - grand['failed']:,} docs summarized"
            f"{failed}, ${total_cost:.2f}"
        )

    store.close()
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
