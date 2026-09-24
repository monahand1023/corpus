"""corpus-query: ad-hoc retrieval against the ingested corpus.

Usage:
  corpus-query "How do I add a new connector?"
  corpus-query "RC3 timeline" --source tickets -k 10
  corpus-query "your question" --rerank
"""

from __future__ import annotations

import argparse
import sys
import textwrap

from corpus.cli._common import load_config_or_exit
from corpus.credentials import resolve_dotenv
from corpus.db.sqlite import ChunkStore
from corpus.embedder.factory import make_embedder
from corpus.retriever import Retriever


def main() -> int:
    parser = argparse.ArgumentParser(description="Ad-hoc query against the corpus")
    parser.add_argument("question", help="Natural-language question")
    parser.add_argument("--top-k", "-k", type=int, default=5)
    parser.add_argument(
        "--source",
        dest="sources",
        action="append",
        default=None,
        help="Filter to a source type (repeatable)",
    )
    parser.add_argument("--no-hybrid", action="store_true", help="Vector-only (skip BM25 fusion)")
    parser.add_argument("--rerank", action="store_true", help="Enable local BGE reranker")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    # Resolve credentials now that --config is known (env var > .env beside
    # --config > .env in cwd). See corpus.credentials for the precedence.
    resolve_dotenv(args.config)

    config = load_config_or_exit(args.config)
    # read_only=True: corpus-query only ever reads. Opening the store any
    # other way risks silently running a schema migration (see
    # ChunkStore._migrate_fts) as a side effect of an ad-hoc query. This does
    # mean a store that doesn't exist yet is now a clean error instead of a
    # silently auto-created empty DB -- there's nothing to query before the
    # first `corpus-ingest` anyway.
    try:
        store = ChunkStore.from_config(config, read_only=True)
    except FileNotFoundError:
        print(f"error: no database at {config.db_path} yet. Run corpus-ingest first.")
        return 1
    embedder = make_embedder(
        provider=config.embedder.provider,
        model=config.embedder.model,
        dim=config.embedder.dim,
    )
    reranker = None
    if args.rerank:
        from corpus.reranker.local import BGEReranker
        reranker = BGEReranker(device=config.reranker.device)

    retriever = Retriever(
        store=store,
        embedder=embedder,
        reranker=reranker,
        reference_patterns=config.compiled_references(),
    )
    try:
        result = retriever.query(
            args.question,
            top_k=args.top_k,
            filter_sources=args.sources,
            hybrid=not args.no_hybrid,
            rerank=args.rerank,
            # Documented in configuration.md, the README and
            # corpus.toml.example -- and read by nobody, so setting it had no
            # effect and every result silently used the hardcoded default.
            max_per_source_type=config.retriever.max_per_source_type,
        )
        if not result.chunks:
            print("(no results)")
            return 1
        for i, chunk in enumerate(result.chunks, 1):
            distance_str = f" d={chunk.distance:.4f}" if chunk.distance is not None else ""
            print(f"[{i}] {chunk.source_type}:{chunk.source_key}{distance_str}")
            if chunk.title:
                print(f"    {chunk.title}")
            if chunk.url:
                print(f"    {chunk.url}")
            wrapped = textwrap.fill(chunk.content, width=100, initial_indent="    ", subsequent_indent="    ")
            print(wrapped[:800] + ("..." if len(chunk.content) > 800 else ""))
            print()
        return 0
    finally:
        retriever.close()


if __name__ == "__main__":
    sys.exit(main())
