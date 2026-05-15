"""corpus-eval: run a known-answer query set against the corpus.

Eval queries live in a Python file you write (`tests/eval_queries.py`).
The shipped version is a commented placeholder template — replace its
queries with ones that target real content in YOUR corpus before relying
on the recall@K number. See the README "Eval" section for methodology.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from collections.abc import Sequence
from pathlib import Path

from dotenv import load_dotenv

from corpus.config import CorpusConfig
from corpus.db.sqlite import ChunkStore
from corpus.embedder.factory import make_embedder
from corpus.retriever import Retriever

load_dotenv()


def _load_queries(path: Path) -> list:
    spec = importlib.util.spec_from_file_location("eval_queries", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return list(module.EVAL_QUERIES)


def _print_result(
    query: str,
    expected: Sequence[str],
    found: Sequence[str],
    passed: bool,
    note: str,
    is_negative: bool,
    source_filter: Sequence[str] | None,
) -> None:
    status = "INFO" if is_negative else ("PASS" if passed else "FAIL")
    print(f"[{status}] {query}")
    if source_filter:
        print(f"       source_filter: {list(source_filter)}")
    if is_negative:
        print(f"       top-K (informational): {list(found)}")
    else:
        print(f"       expected ∈ {list(expected)}, got: {list(found)}")
    if note:
        print(f"       note: {note}")
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description="Run eval queries against the corpus")
    parser.add_argument(
        "--queries",
        default="tests/eval_queries.py",
        help="Path to a Python module exporting `EVAL_QUERIES = [...]`",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--rerank", action="store_true")
    parser.add_argument("--no-hybrid", action="store_true")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    queries_path = Path(args.queries)
    if not queries_path.is_file():
        print(f"Eval queries file not found: {queries_path}")
        print("See the README 'Eval' section + tests/eval_queries.py template.")
        return 2
    queries = _load_queries(queries_path)

    config = CorpusConfig.load(args.config)
    store = ChunkStore(config.db_path, embedding_dim=config.embedder.dim)
    embedder = make_embedder(
        provider=config.embedder.provider,
        model=config.embedder.model,
        dim=config.embedder.dim,
    )
    reranker = None
    if args.rerank:
        from corpus.reranker.local import BGEReranker
        reranker = BGEReranker()

    retriever = Retriever(
        store=store,
        embedder=embedder,
        reranker=reranker,
        reference_patterns=config.compiled_references(),
    )
    try:
        passes = 0
        scored = 0
        for q in queries:
            effective_filter = getattr(q, "source_filter", None)
            result = retriever.query(
                q.query,
                top_k=args.top_k,
                filter_sources=effective_filter,
                hybrid=not args.no_hybrid,
                rerank=args.rerank,
            )
            found = [c.source_key for c in result.chunks]
            is_negative = not q.expected_keys
            passed = any(k in found for k in q.expected_keys)
            if not is_negative:
                scored += 1
                if passed:
                    passes += 1
            _print_result(
                q.query, q.expected_keys, found, passed, getattr(q, "note", ""),
                is_negative, effective_filter,
            )
        if scored:
            print(f"Recall@{args.top_k}: {passes}/{scored} ({100 * passes / scored:.1f}%)")
        return 0 if passes == scored else 1
    finally:
        retriever.close()


if __name__ == "__main__":
    sys.exit(main())
