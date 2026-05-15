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

from dotenv import load_dotenv

from corpus.config import CorpusConfig
from corpus.db.sqlite import ChunkStore

load_dotenv()

# Haiku 4.5 pricing as of 2026:
PRICE_INPUT = 1.0 / 1_000_000
PRICE_OUTPUT = 5.0 / 1_000_000
PRICE_CACHED = 0.10 / 1_000_000


def _reconstruct_doc(chunks: list) -> tuple[str, str]:
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

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    config = CorpusConfig.load(args.config)
    if args.all:
        source_names = [s.name for s in config.sources]
    elif args.source:
        source_names = args.source
    else:
        parser.error("specify --source NAME (repeatable) or --all")
        return 2  # unreachable

    store = ChunkStore(config.db_path, embedding_dim=config.embedder.dim)

    summarizer = None
    if not args.dry_run:
        from corpus.summarizer.anthropic_summarizer import (
            DEFAULT_MODEL,
            AnthropicSummarizer,
            doc_hash,
        )
        summarizer = AnthropicSummarizer()
    else:
        from corpus.summarizer.anthropic_summarizer import DEFAULT_MODEL, doc_hash

    exit_code = 0
    grand = {"docs": 0, "input": 0, "output": 0, "cached": 0}
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
            est_input = sum(len(body) // 4 for _, _, body, _ in to_do)
            est_output = len(to_do) * 150
            est_cost = est_input * PRICE_INPUT + est_output * PRICE_OUTPUT
            print(f"  est. input tokens: {est_input:,}")
            print(f"  est. output tokens: {est_output:,}")
            print(f"  est. cost: ${est_cost:.2f}")
            continue

        assert summarizer is not None
        start = time.monotonic()
        totals = {"input": 0, "output": 0, "cached": 0}

        def _do_one(
            item: tuple[str, str, str, str],
            source_name: str = name,
        ) -> tuple[str, str, object | None, Exception | None]:
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
        cost = (
            totals["input"] * PRICE_INPUT
            + totals["output"] * PRICE_OUTPUT
            - totals["cached"] * (PRICE_INPUT - PRICE_CACHED)
        )
        print(
            f"  done in {elapsed:.0f}s. tokens: in={totals['input']:,} "
            f"out={totals['output']:,} cached={totals['cached']:,}. cost: ${cost:.2f}"
        )
        for k in ("input", "output", "cached"):
            grand[k] += totals[k]
        grand["docs"] += len(to_do)

    if not args.dry_run:
        total_cost = (
            grand["input"] * PRICE_INPUT
            + grand["output"] * PRICE_OUTPUT
            - grand["cached"] * (PRICE_INPUT - PRICE_CACHED)
        )
        print(f"\nTOTAL: {grand['docs']:,} docs, ${total_cost:.2f}")

    store.close()
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
