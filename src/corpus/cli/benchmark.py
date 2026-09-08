"""corpus-benchmark: measure end-to-end and per-stage retrieval latency.

Reports p50/p95/p99 latency for each stage of the query pipeline (embed,
vector search, FTS search, post-processing = fusion + dedupe + diversity)
plus throughput. Optionally A/B compares two embedder providers against the
same corpus.

Stage timing always runs the REAL Retriever.query() -- it never re-derives
the candidate-pool fetch shape (source-type loop, per-source sizing) here.
That shape lives entirely in corpus.retriever.Retriever.query and has
already changed once; a hand-copied formula in this file would silently
drift out of sync with the real query path, timing a shape queries no
longer take. Per-stage numbers are recovered by timing the embedder's and
store's actual entry points for the duration of one real query() call, so
they reflect however many calls Retriever.query happens to make (one
embed, N vector_search/fts_search calls for N source types) rather than an
assumed count.

Usage:
  corpus-benchmark                                      # 50 queries, default config
  corpus-benchmark --runs 200                           # bigger sample
  corpus-benchmark --queries tests/eval_queries.py      # use a real eval set
  corpus-benchmark --compare voyage gemini              # A/B; requires both API keys
  corpus-benchmark --json out.json                      # machine-readable output
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from corpus.cli._common import load_config_or_exit
from corpus.config import CorpusConfig
from corpus.db.sqlite import ChunkStore
from corpus.embedder.factory import make_embedder
from corpus.retriever import Retriever

load_dotenv()

# Default queries if no --queries file is provided. Deliberately mixed:
# prose, identifier-shaped, short, long, multilingual.
_DEFAULT_QUERIES = [
    "the main architecture and design decisions",
    "how is the data ingested",
    "configuration options",
    "MCP tool definitions",
    "retrieval pipeline steps",
    "what does the embedder do",
    "how chunks are stored",
    "writing a new connector",
    "evaluation methodology",
    "sample corpus contents",
]


@dataclass
class StageTimings:
    embed: list[float] = field(default_factory=list)
    vector: list[float] = field(default_factory=list)
    fts: list[float] = field(default_factory=list)
    # Everything Retriever.query does after the store calls return: RRF
    # fusion, dedupe, and diversity capping. Not split further -- splitting
    # it would mean re-implementing those internals here, which is exactly
    # the drift risk this rewrite removes (see module docstring).
    post_processing: list[float] = field(default_factory=list)
    total: list[float] = field(default_factory=list)


def _percentile(values: list[float], pct: float) -> float:
    """Linear-interpolation percentile. `pct` in [0, 100]."""
    if not values:
        return 0.0
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * (pct / 100)
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)


def _summary(values: list[float]) -> dict[str, float]:
    return {
        "mean_ms": round(statistics.mean(values) * 1000, 2) if values else 0.0,
        "p50_ms": round(_percentile(values, 50) * 1000, 2),
        "p95_ms": round(_percentile(values, 95) * 1000, 2),
        "p99_ms": round(_percentile(values, 99) * 1000, 2),
        "min_ms": round(min(values) * 1000, 2) if values else 0.0,
        "max_ms": round(max(values) * 1000, 2) if values else 0.0,
    }


def _instrumented_query(
    retriever: Retriever,
    question: str,
    timings: StageTimings,
    top_k: int,
) -> None:
    """Run one query through the REAL Retriever.query() and record per-stage
    wall-clock times into `timings` by timing the embedder's and store's
    entry points for the duration of that one call -- never by re-deriving
    the candidate-pool fetch shape (see module docstring). This adapts
    automatically to however Retriever.query is actually implemented,
    including calling vector_search/fts_search once per source type."""
    embed_total = 0.0
    vector_total = 0.0
    fts_total = 0.0

    orig_embed = retriever._embedder.embed_query
    orig_vector_search = retriever._store.vector_search
    orig_fts_search = retriever._store.fts_search

    def timed_embed(*args: Any, **kwargs: Any) -> Any:
        nonlocal embed_total
        t0 = time.perf_counter()
        try:
            return orig_embed(*args, **kwargs)
        finally:
            embed_total += time.perf_counter() - t0

    def timed_vector_search(*args: Any, **kwargs: Any) -> Any:
        nonlocal vector_total
        t0 = time.perf_counter()
        try:
            return orig_vector_search(*args, **kwargs)
        finally:
            vector_total += time.perf_counter() - t0

    def timed_fts_search(*args: Any, **kwargs: Any) -> Any:
        nonlocal fts_total
        t0 = time.perf_counter()
        try:
            return orig_fts_search(*args, **kwargs)
        finally:
            fts_total += time.perf_counter() - t0

    retriever._embedder.embed_query = timed_embed  # type: ignore[method-assign]
    retriever._store.vector_search = timed_vector_search  # type: ignore[method-assign]
    retriever._store.fts_search = timed_fts_search  # type: ignore[method-assign]
    try:
        overall_start = time.perf_counter()
        retriever.query(question, top_k=top_k)
        total = time.perf_counter() - overall_start
    finally:
        retriever._embedder.embed_query = orig_embed  # type: ignore[method-assign]
        retriever._store.vector_search = orig_vector_search  # type: ignore[method-assign]
        retriever._store.fts_search = orig_fts_search  # type: ignore[method-assign]

    timings.embed.append(embed_total)
    timings.vector.append(vector_total)
    timings.fts.append(fts_total)
    timings.post_processing.append(max(total - embed_total - vector_total - fts_total, 0.0))
    timings.total.append(total)


def _run_benchmark(
    provider: str,
    model: str,
    dim: int,
    config: CorpusConfig,
    queries: list[str],
    runs_per_query: int,
    top_k: int,
) -> dict[str, Any]:
    store = ChunkStore(config.db_path, embedding_dim=dim)
    embedder = make_embedder(provider=provider, model=model, dim=dim)
    retriever = Retriever(
        store=store, embedder=embedder,
        reference_patterns=config.compiled_references(),
    )
    timings = StageTimings()
    try:
        run_start = time.monotonic()
        for q in queries:
            for _ in range(runs_per_query):
                _instrumented_query(retriever, q, timings, top_k)
        elapsed = time.monotonic() - run_start
    finally:
        retriever.close()

    n = len(timings.total)
    return {
        "provider": provider,
        "model": model,
        "dim": dim,
        "queries": len(queries),
        "runs_per_query": runs_per_query,
        "total_calls": n,
        "throughput_qps": round(n / elapsed, 2) if elapsed > 0 else 0.0,
        "wall_clock_seconds": round(elapsed, 2),
        "stages": {
            "embed": _summary(timings.embed),
            "vector_search": _summary(timings.vector),
            "fts_search": _summary(timings.fts),
            "post_processing": _summary(timings.post_processing),
            "total": _summary(timings.total),
        },
    }


def _benchmark_embed_only(
    provider: str,
    model: str,
    dim: int,
    queries: list[str],
    runs_per_query: int,
) -> dict[str, Any]:
    """Measure ONLY the embed-query step for a provider.

    The full retrieval pipeline can't be fairly A/B'd across providers against
    a single DB — the stored vectors were produced by one provider, so querying
    with another provider's embeddings searches against incompatible vectors.
    What IS comparable is the embedder API's own latency, which is what this
    measures."""
    embedder = make_embedder(provider=provider, model=model, dim=dim)
    latencies: list[float] = []
    run_start = time.monotonic()
    for q in queries:
        for _ in range(runs_per_query):
            t0 = time.perf_counter()
            embedder.embed_query(q)
            latencies.append(time.perf_counter() - t0)
    elapsed = time.monotonic() - run_start
    n = len(latencies)
    return {
        "provider": provider,
        "model": model,
        "dim": dim,
        "mode": "embed-only",
        "queries": len(queries),
        "runs_per_query": runs_per_query,
        "total_calls": n,
        "throughput_qps": round(n / elapsed, 2) if elapsed > 0 else 0.0,
        "wall_clock_seconds": round(elapsed, 2),
        "embed_latency": _summary(latencies),
    }


def _print_compare_report(report: dict[str, Any]) -> None:
    s = report["embed_latency"]
    print(f"  provider:   {report['provider']} ({report['model']}, {report['dim']}-dim)")
    print("  mode:       embed-only (see note below)")
    print(f"  calls:      {report['total_calls']}")
    print(f"  throughput: {report['throughput_qps']} qps")
    print(
        f"  embed latency: mean {s['mean_ms']}ms"
        f" | p50 {s['p50_ms']}ms"
        f" | p95 {s['p95_ms']}ms"
        f" | p99 {s['p99_ms']}ms"
    )


def _load_queries(path: Path) -> list[str]:
    spec = importlib.util.spec_from_file_location("eval_queries", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    # Register in sys.modules before exec: dataclasses with `from __future__
    # import annotations` resolve string annotations via
    # sys.modules[cls.__module__], which raises AttributeError otherwise.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return [q.query for q in module.EVAL_QUERIES]


def _print_report(report: dict[str, Any]) -> None:
    print(f"  provider:   {report['provider']} ({report['model']}, {report['dim']}-dim)")
    print(f"  queries:    {report['queries']} unique × {report['runs_per_query']} runs = {report['total_calls']} calls")
    print(f"  throughput: {report['throughput_qps']} qps")
    print(f"  wall clock: {report['wall_clock_seconds']}s")
    print()
    print(f"  {'Stage':<24} {'mean':>10} {'p50':>10} {'p95':>10} {'p99':>10} {'max':>10}")
    print(f"  {'-' * 24} {'-' * 10} {'-' * 10} {'-' * 10} {'-' * 10} {'-' * 10}")
    for name, s in report["stages"].items():
        print(
            f"  {name:<24} "
            f"{s['mean_ms']:>9.2f}ms"
            f" {s['p50_ms']:>9.2f}ms"
            f" {s['p95_ms']:>9.2f}ms"
            f" {s['p99_ms']:>9.2f}ms"
            f" {s['max_ms']:>9.2f}ms"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Latency + throughput benchmark for corpus retrieval")
    parser.add_argument("--runs", type=int, default=5, help="Repetitions per query (default: 5)")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--queries", default=None, help="Path to an EVAL_QUERIES module")
    parser.add_argument(
        "--compare",
        nargs=2,
        metavar=("PROVIDER1", "PROVIDER2"),
        default=None,
        help=(
            "A/B compare embed-step latency of two providers. Each must be "
            "installed + have an API key set. Measures embedder API latency "
            "only — NOT full-pipeline retrieval (that can't be fairly compared "
            "against a single DB)."
        ),
    )
    parser.add_argument("--json", default=None, help="Write report to FILE as JSON")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    config = load_config_or_exit(args.config)

    if args.queries:
        queries = _load_queries(Path(args.queries))
    else:
        queries = _DEFAULT_QUERIES
    if not queries:
        print("No queries to benchmark.")
        return 1

    print(f"Benchmarking {len(queries)} queries × {args.runs} runs against {config.db_path}")
    print()

    if args.compare:
        # Embed-only latency comparison. Each provider uses its own native
        # default dim — we're measuring API latency, not touching the DB, so
        # the dim guard doesn't apply and there's no cross-contamination.
        reports = []
        for provider in args.compare:
            print(f"=== {provider} ===")
            report = _benchmark_embed_only(
                provider=provider,
                model=_default_model_for(provider),
                dim=_default_dim_for(provider),
                queries=queries,
                runs_per_query=args.runs,
            )
            _print_compare_report(report)
            print()
            reports.append(report)
        print(
            "Note: this compares embedder API latency only. It does NOT "
            "compare retrieval quality — for that, ingest each provider into "
            "its own corpus and run `corpus-eval` against each."
        )
        if args.json:
            Path(args.json).write_text(json.dumps({"compare": reports}, indent=2))
        return 0

    report = _run_benchmark(
        provider=config.embedder.provider,
        model=config.embedder.model,
        dim=config.embedder.dim,
        config=config,
        queries=queries,
        runs_per_query=args.runs,
        top_k=args.top_k,
    )
    _print_report(report)
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2))
    return 0


def _default_model_for(provider: str) -> str:
    return {"voyage": "voyage-4-large", "gemini": "gemini-embedding-001"}.get(
        provider, "unknown-model"
    )


def _default_dim_for(provider: str) -> int:
    return {"voyage": 1024, "gemini": 1536}.get(provider, 1024)


if __name__ == "__main__":
    sys.exit(main())
