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

from corpus.cli._common import load_config_or_exit, open_store_read_only
from corpus.credentials import resolve_dotenv
from corpus.embedder.factory import make_embedder
from corpus.eval.goldset import (
    audit_queries,
    report_findings,
    sqlite_documents,
    sqlite_lookup,
)
from corpus.eval.metrics import MetricSummary, QueryScore, aggregate, score_query
from corpus.eval.noise import METRICS, MetricSpread, gate_verdict, spread_report
from corpus.retriever import Retriever


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
    rerank_pool_size: int = 30,
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
            rerank_pool_size=rerank_pool_size,
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


def _all_source_keys(db_path: str | Path) -> list[str]:
    """Every distinct source_key, or [] when the index cannot be read."""
    import sqlite3

    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    except sqlite3.Error:
        return []
    try:
        return [r[0] for r in conn.execute("SELECT DISTINCT source_key FROM chunks")]
    except sqlite3.Error:
        return []
    finally:
        conn.close()


def _document_count(db_path: str | Path) -> int:
    """Distinct documents in the archive, or 0 when it cannot be read.

    0 means "I could not judge this", and `triviality_report` says exactly
    that rather than reporting a clean gold set -- the distinction this
    codebase keeps having to make.
    """
    import sqlite3

    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    except sqlite3.Error:
        return 0
    try:
        row = conn.execute("SELECT count(DISTINCT source_key) FROM chunks").fetchone()
        return int(row[0]) if row else 0
    except sqlite3.Error:
        return 0
    finally:
        conn.close()


def _report_triviality(
    *, key_counts: dict[str, int], documents: int, top_k: int
) -> None:
    """Print the random-retriever baseline, but only when it matters.

    A gold-set query that accepts a large share of the archive scores well
    for a retriever that does nothing, and nothing else in this output would
    say so: recall@5 = 0.708 looks identical whether it was earned or fell
    out of the size of the answer sets.

    SILENT ON THE CLEAN PATH. A line printed on every run stops being read,
    and this one has to be read on the run where it fires.
    """
    from corpus.eval.triviality import triviality_report

    if not key_counts:
        return
    if documents <= 0:
        # NOT silent. Silence is what a SOUND gold set produces here, so
        # returning quietly made "I could not read the index" render
        # identically to "there is nothing wrong" -- the exact distinction
        # `_document_count`'s own docstring promises to preserve.
        print("=== Gold set ===")
        print(
            "  could not read the archive to judge whether any query is "
            "trivially\n  satisfiable, so this says nothing either way "
            "(not a clean result)."
        )
        print()
        return
    try:
        report = triviality_report(
            key_counts, documents=documents, top_k=top_k
        )
    except ValueError as exc:
        # A gold set naming more answers than the archive holds is a defect
        # in the gold set. Say so; do not let it stop the eval.
        print(f"=== Gold set ===\n  {exc}\n")
        return
    if report.is_clean:
        return
    print("=== Gold set ===")
    for line in report.describe().splitlines():
        print(f"  {line}")
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


# What a thresholds file may record about WHERE its numbers came from.
#
# A floor is a claim about a specific gold set, and nothing connected the two.
# One archive's floors were measured against an 8-query, transcripts-only set;
# the set grew to 24 and the floors stayed, leaving the recall gate six queries
# below its measurement -- a gate that could not fail. The drift is always in
# that direction: the set grows, the measurement rises, the floor stays. It
# never drifts toward failing, which is exactly why nobody notices.
_PROVENANCE_KEY = "measured"


def _load_threshold_provenance(path: Path) -> dict[str, Any]:
    """What gold set these floors were measured against, or {} if unrecorded."""
    try:
        data: Any = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    block = data.get(_PROVENANCE_KEY)
    return dict(block) if isinstance(block, dict) else {}


def _warn_on_stale_thresholds(
    provenance: dict[str, Any], *, n_queries: int | None
) -> None:
    """Say so when the gold set is not the one the floors were measured from.

    SILENT when the file records nothing, which is every thresholds file that
    existed before this was added. Nagging their owners on every run is how a
    warning stops being read.
    """
    recorded = provenance.get("n_queries")
    if not isinstance(recorded, int) or not n_queries or recorded == n_queries:
        return
    direction = "grew" if n_queries > recorded else "shrank"
    print(
        f"\n=== Thresholds ===\n"
        f"  These floors were measured against {recorded} queries; this gold "
        f"set has {n_queries}.\n"
        f"  The set {direction} since the floors were set, so they describe a "
        "different\n  measurement than the one above. Re-measure with "
        "--repeat and update both."
    )


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
        if key == _PROVENANCE_KEY:
            # Not a metric. Kept out of `thresholds` so it is never gated on.
            if not isinstance(value, dict):
                raise ValueError(
                    f"{_PROVENANCE_KEY!r} must be an object describing the gold "
                    "set these floors were measured against, e.g. "
                    '{"n_queries": 24, "date": "2026-09-16"}'
                )
            continue
        if key not in _CHECKABLE_METRICS:
            raise ValueError(f"unknown threshold metric: {key!r}")
        if not isinstance(value, int | float):
            raise ValueError(f"threshold for {key!r} must be a number")
        thresholds[key] = float(value)
    return thresholds


def _print_spread(spreads: dict[str, MetricSpread], top_k: int) -> None:
    print("\n=== Spread across runs ===")
    for metric in METRICS:
        label = metric.replace("_at_k", f"@{top_k}")
        print(f"  {label:<10} {spreads[metric].describe()}")
    widest = max(METRICS, key=lambda m: spreads[m].spread)
    widest_label = widest.replace("_at_k", f"@{top_k}")
    print(
        "\nA hosted embedder does not return a bit-identical vector for a fixed\n"
        "query, so result ORDER moves between identical runs. Order changes set\n"
        "MEMBERSHIP only when a hit sits on the k BOUNDARY -- and then it changes\n"
        "it completely, because rank k+1 scores that query 0 where rank k scored\n"
        "it 1."
    )
    if spreads[widest].spread > 0:
        print(
            f"\n{widest_label} moved most here ({spreads[widest].spread:.3f}). "
            "Do not assume that is\nthe rank-weighted metrics: recall@k is "
            "QUANTISED to 1/n, so one boundary\nflip moves it by a whole 1/n "
            "while MRR and nDCG absorb the same flip as a\nsmall continuous "
            "change. Measured on a real archive: recall@5 spread 0.042\n"
            "(= 1/24, one query) against MRR 0.005. Quantisation does not make a\n"
            "metric stable; it makes its noise arrive in one lump."
        )
    print("\nSet a gate below the worst run by at least twice this spread.")


def _print_gate(
    spreads: dict[str, MetricSpread],
    thresholds: dict[str, float],
    *,
    n_queries: int | None = None,
) -> bool:
    """Print one `[gate] ...` line per checked metric to stderr.

    Judged on the WORST run, not the average: a gate that passes on the mean
    and fails one run in three is a flaky gate, not a passing one. Returns
    True iff every checked metric clears its floor in every run.
    """
    passed = True
    for metric, floor in thresholds.items():
        verdict = gate_verdict(
            floor=floor, spread=spreads[metric], n_queries=n_queries
        )
        if not verdict.passed:
            passed = False
        print(verdict.describe(), file=sys.stderr)
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


def build_parser() -> argparse.ArgumentParser:
    """The CLI surface, separately constructible.

    Extracted so `corpus-doctor` can INSPECT it. Its "served vs
    evaluated" check used to print a hardcoded `[ ok ]` restating a
    config value; with a parser in hand it can read what `--top-k`
    actually defaults to and fail when that stops being derived from
    the config.
    """
    parser = argparse.ArgumentParser(description="Run eval queries and report retrieval metrics")
    parser.add_argument(
        "--queries",
        default="tests/eval_queries.py",
        help="Path to a Python module exporting EVAL_QUERIES",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help=(
            "Results per query. Defaults to [retriever] top_k from the config "
            "-- the number the MCP server actually serves. Measuring at a "
            "different k reports a figure no user experiences."
        ),
    )
    parser.add_argument("--rerank", action="store_true")
    parser.add_argument(
        "--rerank-pool-size",
        type=int,
        default=30,
        help=(
            "Candidates the cross-encoder re-scores. Latency is linear in "
            "this: measured on one archive, 0.35s with no re-ranking, 2.3s "
            "at 8, 4.5s at 15, 8.6s at 30. Worth sweeping before enabling "
            "re-ranking on a server people wait on."
        ),
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        metavar="N",
        help=(
            "Run the whole gold set N times and report each metric's spread. "
            "A hosted embedder is not bit-reproducible, so rank-weighted "
            "metrics move run to run even when nothing changed; with --check, "
            "the gate is then judged on the WORST run and flagged when its "
            "headroom is inside that noise. Costs N times the query "
            "embeddings."
        ),
    )
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
        "--allow-gold-issues",
        action="store_true",
        help="Run even when the gold set audit reports an error",
    )
    parser.add_argument(
        "--check",
        default=None,
        metavar="PATH",
        help=(
            "Path to a JSON thresholds file (e.g. {\"recall_at_k\": 0.95}); "
            "gate the exit code on the aggregate meeting each floor (single-config only)"
        ),
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    # Resolve credentials now that --config is known (env var > .env beside
    # --config > .env in cwd). See corpus.credentials for the precedence.
    resolve_dotenv(args.config)

    queries_path = Path(args.queries)
    if not queries_path.is_file():
        print(f"Eval queries file not found: {queries_path}", file=sys.stderr)
        print("See docs/eval.md + tests/eval_queries.py.", file=sys.stderr)
        return 2
    queries = _load_queries(queries_path)

    config = load_config_or_exit(args.config)
    store = open_store_read_only(config)
    embedder = make_embedder(
        provider=config.embedder.provider,
        model=config.embedder.model,
        dim=config.embedder.dim,
    )
    # The served path is the one worth measuring. Reporting recall@15 when
    # the server hands back 5 results describes a system nobody is running.
    if args.top_k is None:
        args.top_k = config.retriever.top_k
        print(
            f"top-k {args.top_k} (from [retriever] top_k; override with --top-k)",
            file=sys.stderr,
        )
    hybrid = not args.no_hybrid
    if hybrid != config.retriever.hybrid:
        print(
            f"warning: evaluating with hybrid={hybrid} but the config serves "
            f"hybrid={config.retriever.hybrid}; this measures a path nobody runs",
            file=sys.stderr,
        )

    findings = audit_queries(
        queries,
        lookup=sqlite_lookup(config.db_path),
        documents=sqlite_documents(config.db_path, queries),
        # BOTH of these, or two checks silently do not run. They are optional
        # on `audit_queries` so a caller with no index still gets the
        # structural checks -- which makes forgetting them here look exactly
        # like a gold set with nothing wrong. corpus-doctor passed them and
        # this command, the one with the authority to REFUSE to run, did not.
        documents_total=_document_count(config.db_path),
        all_keys=_all_source_keys(config.db_path),
    )
    if report_findings(findings, stream=sys.stderr) and not args.allow_gold_issues:
        print(
            "Refusing to run: the answer key is broken, so the metrics would "
            "describe the key rather than the retriever. Fix it, or pass "
            "--allow-gold-issues.",
            file=sys.stderr,
        )
        return 2

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
        if args.compare:
            _print_compare(
                retriever, queries, top_k=args.top_k, have_reranker=reranker is not None
            )
            return 0
        if args.repeat < 1:
            print("--repeat must be at least 1", file=sys.stderr)
            return 2
        runs: list[MetricSummary] = []
        for attempt in range(args.repeat):
            records = _run_query_set(
                retriever,
                queries,
                top_k=args.top_k,
                hybrid=not args.no_hybrid,
                rerank=args.rerank,
                rerank_pool_size=args.rerank_pool_size,
            )
            runs.append(aggregate(_scored(records)))
            if args.repeat > 1 and not args.as_json:
                r = runs[-1]
                print(
                    f"run {attempt + 1}/{args.repeat}: recall@{args.top_k}="
                    f"{r.recall_at_k:.3f} MRR={r.mrr:.3f} nDCG@{args.top_k}="
                    f"{r.ndcg_at_k:.3f}",
                    file=sys.stderr,
                )
        spreads = spread_report(runs)
        if args.repeat > 1 and not args.as_json:
            _print_spread(spreads, args.top_k)
        if args.as_json:
            print(json.dumps(_build_json(records, args.top_k, not args.no_hybrid, args.rerank), indent=2))
        else:
            _print_human(records, args.top_k)
            # Qualify the numbers just printed. A gold set whose queries a
            # random retriever satisfies produces a healthy-looking recall
            # that is mostly the size of its answer sets, and nothing else
            # in this output distinguishes the two.
            _report_triviality(
                key_counts={q.query: len(q.expected_keys) for q in queries},
                documents=_document_count(config.db_path),
                top_k=args.top_k,
            )

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
            n = runs[-1].n if runs else None
            # Before the verdict, not after: a floor measured against a
            # different gold set is not a floor about THIS measurement, and
            # the reader needs to know that while looking at the numbers.
            _warn_on_stale_thresholds(
                _load_threshold_provenance(check_path), n_queries=n
            )
            return 0 if _print_gate(spreads, thresholds, n_queries=n) else 1

        return 0
    finally:
        retriever.close()


if __name__ == "__main__":
    sys.exit(main())
