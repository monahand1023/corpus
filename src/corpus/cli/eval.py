"""corpus-eval: run a known-answer query set and report retrieval-quality metrics.

Reports recall@K, MRR, and nDCG@K per query plus an aggregate table and a
per-source-type breakdown. `--json` emits the whole result as JSON (parsed by the
CI regression gate). `--compare` runs the set under several retrieval configs
(hybrid on/off, and rerank if `--rerank`) and prints a metric x config table.

Eval queries live in a Python file exporting `EVAL_QUERIES` (default
`tests/eval_queries.py`). The shipped set targets `examples/sample_corpus`.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from corpus.cli._common import load_config_or_exit
from corpus.db.sqlite import ChunkStore
from corpus.embedder.factory import make_embedder
from corpus.eval.metrics import MetricSummary, QueryScore, aggregate, score_query
from corpus.retriever import Retriever

load_dotenv()


@dataclass(frozen=True)
class QueryRecord:
    query: str
    expected_keys: list[str]
    found_keys: list[str]
    source_type: str | None
    note: str
    is_negative: bool
    score: QueryScore | None


def _load_queries(path: Path) -> list[Any]:
    spec = importlib.util.spec_from_file_location("eval_queries", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    # Register in sys.modules before exec: dataclasses with `from __future__
    # import annotations` resolve string annotations via
    # sys.modules[cls.__module__], which raises AttributeError otherwise.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return list(module.EVAL_QUERIES)


def _run_query_set(
    retriever: Retriever,
    queries: Sequence[Any],
    *,
    top_k: int,
    hybrid: bool,
    rerank: bool,
) -> list[QueryRecord]:
    records: list[QueryRecord] = []
    for q in queries:
        effective_filter = getattr(q, "source_filter", None)
        result = retriever.query(
            q.query,
            top_k=top_k,
            filter_sources=effective_filter,
            hybrid=hybrid,
            rerank=rerank,
        )
        found = [c.source_key for c in result.chunks]
        expected = list(q.expected_keys)
        is_negative = not expected
        score = None if is_negative else score_query(expected, found, top_k)
        records.append(
            QueryRecord(
                query=q.query,
                expected_keys=expected,
                found_keys=found,
                source_type=getattr(q, "source_type", None),
                note=getattr(q, "note", ""),
                is_negative=is_negative,
                score=score,
            )
        )
    return records


def _scored(records: Sequence[QueryRecord]) -> list[QueryScore]:
    return [r.score for r in records if r.score is not None]


def _breakdown(records: Sequence[QueryRecord]) -> dict[str, MetricSummary]:
    groups: dict[str, list[QueryScore]] = {}
    for r in records:
        if r.score is None:
            continue
        key = r.source_type or "(untagged)"
        groups.setdefault(key, []).append(r.score)
    return {k: aggregate(v) for k, v in sorted(groups.items())}


def _summary_dict(s: MetricSummary) -> dict[str, float | int]:
    return {"recall_at_k": s.recall_at_k, "mrr": s.mrr, "ndcg_at_k": s.ndcg_at_k, "n": s.n}


def _print_human(records: Sequence[QueryRecord], top_k: int) -> None:
    for r in records:
        if r.is_negative:
            print(f"[INFO] {r.query}")
            print(f"       top-{top_k} (informational): {r.found_keys}")
        else:
            assert r.score is not None
            status = "PASS" if r.score.recall > 0 else "FAIL"
            print(f"[{status}] {r.query}")
            print(f"       expected in {r.expected_keys}, got: {r.found_keys}")
            print(
                f"       recall@{top_k}={r.score.recall:.0f} "
                f"rr={r.score.rr:.3f} ndcg@{top_k}={r.score.ndcg:.3f}"
            )
        if r.note:
            print(f"       note: {r.note}")
        print()

    overall = aggregate(_scored(records))
    print(f"=== Aggregate (n={overall.n}) ===")
    print(f"  recall@{top_k}: {overall.recall_at_k:.3f}")
    print(f"  MRR:       {overall.mrr:.3f}")
    print(f"  nDCG@{top_k}:   {overall.ndcg_at_k:.3f}")
    print()

    breakdown = _breakdown(records)
    if len(breakdown) > 1:
        print("=== By source_type ===")
        print(f"  {'source_type':<16} {'n':>4} {'recall':>8} {'mrr':>8} {'ndcg':>8}")
        for stype, s in breakdown.items():
            print(
                f"  {stype:<16} {s.n:>4} {s.recall_at_k:>8.3f} "
                f"{s.mrr:>8.3f} {s.ndcg_at_k:>8.3f}"
            )
        print()


def _build_json(
    records: Sequence[QueryRecord], top_k: int, hybrid: bool, rerank: bool
) -> dict[str, Any]:
    return {
        "config": {"top_k": top_k, "hybrid": hybrid, "rerank": rerank},
        "aggregate": _summary_dict(aggregate(_scored(records))),
        "by_source_type": {k: _summary_dict(v) for k, v in _breakdown(records).items()},
        "queries": [
            {
                "query": r.query,
                "expected_keys": r.expected_keys,
                "found_keys": r.found_keys,
                "source_type": r.source_type,
                "is_negative": r.is_negative,
                "recall": None if r.score is None else r.score.recall,
                "rr": None if r.score is None else r.score.rr,
                "ndcg": None if r.score is None else r.score.ndcg,
            }
            for r in records
        ],
    }


_CHECKABLE_METRICS = ("recall_at_k", "mrr", "ndcg_at_k")


def _load_thresholds(path: Path) -> dict[str, float]:
    """Load a `--check` thresholds JSON file.

    Raises FileNotFoundError if the path doesn't exist, ValueError for anything
    else invalid (bad JSON, non-object body, unknown/non-numeric metric).
    """
    if not path.is_file():
        raise FileNotFoundError(str(path))
    data: Any = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError("thresholds file must contain a JSON object")
    thresholds: dict[str, float] = {}
    for key, value in data.items():
        if key not in _CHECKABLE_METRICS:
            raise ValueError(f"unknown threshold metric: {key!r}")
        if not isinstance(value, int | float):
            raise ValueError(f"threshold for {key!r} must be a number")
        thresholds[key] = float(value)
    return thresholds


def _print_gate(overall: MetricSummary, thresholds: dict[str, float]) -> bool:
    """Print one `[gate] ...` line per checked metric to stderr.

    Returns True iff every checked metric meets (or exceeds) its floor.
    """
    passed = True
    for metric, floor in thresholds.items():
        value = getattr(overall, metric)
        ok = value >= floor
        if not ok:
            passed = False
        status = "PASS" if ok else "FAIL"
        print(f"[gate] {metric} = {value:.3f}  floor {floor:.3f}  {status}", file=sys.stderr)
    return passed


def _print_compare(
    retriever: Retriever, queries: Sequence[Any], *, top_k: int, have_reranker: bool
) -> None:
    configs: list[tuple[str, bool, bool]] = [
        ("hybrid", True, False),
        ("vector-only", False, False),
    ]
    if have_reranker:
        configs.append(("hybrid+rerank", True, True))

    print(f"=== Config comparison (top_k={top_k}) ===")
    print(f"  {'config':<16} {'recall':>8} {'mrr':>8} {'ndcg':>8}")
    for name, hybrid, rerank in configs:
        records = _run_query_set(retriever, queries, top_k=top_k, hybrid=hybrid, rerank=rerank)
        s = aggregate(_scored(records))
        print(f"  {name:<16} {s.recall_at_k:>8.3f} {s.mrr:>8.3f} {s.ndcg_at_k:>8.3f}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run eval queries and report retrieval metrics")
    parser.add_argument(
        "--queries",
        default="tests/eval_queries.py",
        help="Path to a Python module exporting EVAL_QUERIES",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--rerank", action="store_true")
    parser.add_argument("--no-hybrid", action="store_true")
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run several retrieval configs and print a metric x config table",
    )
    parser.add_argument(
        "--json", action="store_true", dest="as_json", help="Emit the result as JSON"
    )
    parser.add_argument("--config", default=None)
    parser.add_argument(
        "--check",
        default=None,
        metavar="PATH",
        help=(
            "Path to a JSON thresholds file (e.g. {\"recall_at_k\": 0.95}); "
            "gate the exit code on the aggregate meeting each floor (single-config only)"
        ),
    )
    args = parser.parse_args()

    queries_path = Path(args.queries)
    if not queries_path.is_file():
        print(f"Eval queries file not found: {queries_path}", file=sys.stderr)
        print("See docs/eval.md + tests/eval_queries.py.", file=sys.stderr)
        return 2
    queries = _load_queries(queries_path)

    config = load_config_or_exit(args.config)
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
        if args.compare:
            _print_compare(
                retriever, queries, top_k=args.top_k, have_reranker=reranker is not None
            )
            return 0
        records = _run_query_set(
            retriever, queries, top_k=args.top_k, hybrid=not args.no_hybrid, rerank=args.rerank
        )
        if args.as_json:
            print(json.dumps(_build_json(records, args.top_k, not args.no_hybrid, args.rerank), indent=2))
        else:
            _print_human(records, args.top_k)

        if args.check:
            check_path = Path(args.check)
            try:
                thresholds = _load_thresholds(check_path)
            except FileNotFoundError:
                print(f"Thresholds file not found: {check_path}", file=sys.stderr)
                return 2
            except ValueError as e:
                print(f"Invalid thresholds file {check_path}: {e}", file=sys.stderr)
                return 2
            overall = aggregate(_scored(records))
            return 0 if _print_gate(overall, thresholds) else 1

        return 0
    finally:
        retriever.close()


if __name__ == "__main__":
    sys.exit(main())
