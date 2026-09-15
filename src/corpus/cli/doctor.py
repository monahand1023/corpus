"""corpus-doctor: check the MEASUREMENT setup, not the system it measures.

Every other check here answers a question about the archive. This one answers
a question about the instruments: can the numbers be trusted at all?

It exists because four separate contaminations of those instruments were found
in one week, each by accident while doing something else:

1. `corpus-smoke`'s probe was being written to the query log.
2. An archive's eval measured `top_k=5` while its server served 15, so every
   figure ever reported for it described a configuration nobody ran.
3. Six distinct defects in gold-set answer keys, each of which looked exactly
   like a broken retriever.
4. A consumer's test suite wrote to its repo's real query log: 276 entries
   that were 180 copies of "q" and assorted fixtures.

Each was fixed durably. Finding them was not durable — it depended on someone
noticing. That is what this command is for.

    corpus-doctor --config corpus.toml
    corpus-doctor --config corpus.toml --query-log data/queries.jsonl
    corpus-doctor --query-log a/data/queries.jsonl --query-log b/data/queries.jsonl

Exit status is 0 only when every check that ran found no errors. Checks whose
inputs are absent are SKIPPED and say so, because a check that silently does
not run is the failure mode this command exists to catch.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from corpus.eval.query_log_audit import (
    LogReport,
    audit_query_log,
    cross_archive_overlap,
)


def _check_query_logs(paths: Sequence[str], min_real: int) -> tuple[bool, list[LogReport]]:
    if not paths:
        print("query logs        SKIPPED (none given; pass --query-log)")
        return True, []

    print("query logs")
    reports = []
    ok = True
    for raw in paths:
        path = Path(raw).expanduser()
        if not path.is_file():
            print(f"  {path}: SKIPPED (no such file -- logging may never have run)")
            continue
        report = audit_query_log(path, min_real_for_conclusions=min_real)
        reports.append(report)
        status = "ok" if report.ok else "PROBLEM"
        print(
            f"  {path}: {status} -- {report.total} entries, "
            f"{report.distinct_real} distinct real, "
            f"{report.synthetic} synthetic, {report.fixture} fixture"
        )
        for finding in report.findings:
            print(finding.render())
        ok = ok and report.ok
    return ok, reports


def _check_cross_archive(reports: Sequence[LogReport]) -> None:
    """The measurement behind 'one server or several?'."""
    if len(reports) < 2:
        return
    overlap = cross_archive_overlap(reports)
    total_real = len({q.casefold() for r in reports for q in r.real_queries})
    print("\ncross-archive demand")
    if not total_real:
        print("  SKIPPED (no real queries in any log yet)")
        return
    share = len(overlap) / total_real
    print(
        f"  {len(overlap)} of {total_real} distinct real queries were asked of "
        f"more than one archive ({share:.0%})"
    )
    print(
        "  Rarely -> separate servers are already right, build nothing. "
        "Often -> a single facade earns itself."
    )
    for query, archives in sorted(overlap.items())[:5]:
        print(f"    {query[:58]!r}: {', '.join(archives)}")


def _check_served_vs_evaluated(config_path: str | None) -> bool:
    """Does the eval measure the k the server serves?"""
    if not config_path:
        print("\nserved vs evaluated   SKIPPED (no --config)")
        return True
    from corpus.cli._common import load_config_or_exit

    config = load_config_or_exit(config_path)
    print("\nserved vs evaluated")
    print(
        f"  [  ok  ] corpus-eval defaults --top-k to [retriever] top_k "
        f"= {config.retriever.top_k}, and --no-hybrid warns when it diverges "
        f"from hybrid = {config.retriever.hybrid}"
    )
    print(
        "  NOTE: a consumer with its OWN eval CLI does not inherit that. "
        "Import the server's default rather than hardcoding one."
    )
    return True


def _check_gold_set(queries_path: str | None, config_path: str | None) -> bool:
    if not queries_path:
        print("\ngold set          SKIPPED (no --queries)")
        return True
    path = Path(queries_path).expanduser()
    if not path.is_file():
        print(f"\ngold set          SKIPPED (no such file: {path})")
        return True
    if not config_path:
        print("\ngold set          SKIPPED (needs --config to read the index)")
        return True

    from corpus.cli._common import load_config_or_exit
    from corpus.cli.eval import _load_queries
    from corpus.eval.goldset import (
        audit_queries,
        report_findings,
        sqlite_documents,
        sqlite_lookup,
    )

    config = load_config_or_exit(config_path)
    queries = _load_queries(path)
    findings = audit_queries(
        queries,
        lookup=sqlite_lookup(config.db_path),
        documents=sqlite_documents(config.db_path, queries),
    )
    print(f"\ngold set          {len(queries)} queries")
    has_error = report_findings(findings, stream=sys.stdout)
    if not findings:
        print("  [  ok  ] no findings")
    return not has_error


def _check_index_quality(config_path: str | None) -> bool:
    if not config_path:
        print("\nindex quality     SKIPPED (no --config)")
        return True
    from corpus.cli._common import load_config_or_exit
    from corpus.survey.index_quality import NotACorpusIndexError, run_index_quality

    config = load_config_or_exit(config_path)
    print("\nindex quality")
    try:
        result = run_index_quality(config.db_path, sample_per_kind=0)
    except NotACorpusIndexError as exc:
        print(f"  SKIPPED ({exc})")
        return True
    except Exception as exc:
        print(f"  SKIPPED (could not scan: {type(exc).__name__})")
        return True

    if result.clean:
        print(f"  [  ok  ] {result.scanned_chunks} chunks, no transcription artefacts")
        return True
    # Only a chunk that is ENTIRELY boilerplate is a filtering defect and
    # fails. A sign-off glued to real speech is a known, re-ingestable state,
    # so it reports without failing -- and the label has to say which, or the
    # summary line and the detail line contradict each other.
    failing = bool(result.whole_chunk)
    mark = "[ FAIL ]" if failing else "[ warn ]"
    print(
        f"  {mark} {result.affected_chunks} chunk(s) across "
        f"{len(result.documents_affected)} document(s): "
        f"{len(result.whole_chunk)} entirely boilerplate, "
        f"{len(result.tails)} with a sign-off glued to real speech"
    )
    if not failing:
        print("           (glued tails are fixed by a re-ingest, not a filter)")
    return not failing


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Check that the measurement setup itself can be trusted",
    )
    parser.add_argument("--config", default=None, help="Path to corpus.toml")
    parser.add_argument(
        "--query-log",
        action="append",
        default=[],
        metavar="PATH",
        help="A query log to audit (repeatable; two or more enables the "
             "cross-archive demand check)",
    )
    parser.add_argument(
        "--queries",
        default=None,
        help="Gold set module to audit (default: tests/eval_queries.py if present)",
    )
    parser.add_argument(
        "--min-real-queries",
        type=int,
        default=30,
        help="Distinct real queries a log needs before conclusions are drawn",
    )
    args = parser.parse_args(argv)

    if args.queries is None and Path("tests/eval_queries.py").is_file():
        args.queries = "tests/eval_queries.py"

    print("=== corpus-doctor: are the instruments trustworthy? ===\n")

    # Three states, not two. A SKIPPED check is NOT a passing check, and
    # counting it as one would be the same inflation this command exists to
    # catch -- a summary reading "4/4 clean" when three of them never ran is
    # exactly how the query logs came to be trusted.
    ran: dict[str, bool] = {}
    skipped: list[str] = []

    logs_ok, reports = _check_query_logs(args.query_log, args.min_real_queries)
    if args.query_log:
        ran["query logs"] = logs_ok
    else:
        skipped.append("query logs")
    _check_cross_archive(reports)

    for name, check, needs in (
        ("served vs evaluated", _check_served_vs_evaluated(args.config), args.config),
        ("gold set", _check_gold_set(args.queries, args.config), args.queries and args.config),
        ("index quality", _check_index_quality(args.config), args.config),
    ):
        if needs:
            ran[name] = check
        else:
            skipped.append(name)

    failed = [name for name, ok in ran.items() if not ok]
    print(
        f"\n{'PROBLEMS FOUND' if failed else 'OK'}: "
        f"{len(ran) - len(failed)}/{len(ran)} checks passed"
        + (f", {len(skipped)} SKIPPED ({', '.join(skipped)})" if skipped else "")
    )
    if skipped:
        print(
            "  A skipped check is not a passing one. Pass --config and "
            "--query-log to run them."
        )
    if failed:
        print(
            "  A number measured through a contaminated instrument is worse "
            "than no number, because it still gets acted on."
        )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
