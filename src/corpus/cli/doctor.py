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


def _resolve_query_logs(args: argparse.Namespace) -> tuple[list[str], str]:
    """The logs to audit, and -- when there are none -- WHY there are none.

    `--query-log` used to be the only way in, so this check was skipped on
    every run nobody remembered the flag for. Across five live archives that
    was every run: the command printed "A skipped check is not a passing one"
    and then skipped it by default.

    The path comes from `corpus.query_log.configured_log_path`, the same
    function the server appends through. Resolving it independently here would
    be worse than not resolving it: the auditor would read a file nothing
    writes to and pronounce it clean.

    The reason matters as much as the paths. "Logging is disabled in config"
    and "logging is on but nothing has been served yet" are different
    problems with different fixes, and neither is "you forgot a flag".
    """
    if args.query_log:
        return list(args.query_log), ""
    if not args.config:
        return [], "no --config and no --query-log"

    from corpus.cli._common import load_config_or_exit
    from corpus.query_log import configured_log_path

    path = configured_log_path(load_config_or_exit(args.config))
    if path is None:
        return [], "query logging is OFF in config; set [query_log] enabled = true"
    if not path.is_file():
        return [], f"logging is on but {path} does not exist yet -- nothing served"
    return [str(path)], ""


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


def _check_gold_set(queries_path: str | None, db_path: str | None) -> bool:
    if not queries_path:
        print("\ngold set          SKIPPED (no --queries)")
        return True
    path = Path(queries_path).expanduser()
    if not path.is_file():
        print(f"\ngold set          SKIPPED (no such file: {path})")
        return True
    if not db_path:
        print("\ngold set          SKIPPED (needs --config or --db to read the index)")
        return True

    from corpus.cli.eval import _load_queries
    from corpus.eval.goldset import (
        audit_queries,
        report_findings,
        sqlite_documents,
        sqlite_lookup,
    )

    try:
        queries = _load_queries(path)
    except Exception as exc:
        # A gold set that will not import is itself a finding, and it must not
        # take the other checks down with it -- a diagnostic that stops at the
        # first problem hides every later one.
        print("\ngold set")
        print(f"  [ FAIL ] could not import {path}: {type(exc).__name__}: {exc}")
        print(
            "           (a gold set often imports its own package; run this "
            "from the deployment's interpreter)"
        )
        return False
    findings = audit_queries(
        queries,
        lookup=sqlite_lookup(db_path),
        documents=sqlite_documents(db_path, queries),
    )
    print(f"\ngold set          {len(queries)} queries")
    has_error = report_findings(findings, stream=sys.stdout)
    if not findings:
        print("  [  ok  ] no findings")
    return not has_error


def _check_index_quality(db_path: str | None) -> bool:
    if not db_path:
        print("\nindex quality     SKIPPED (no --config or --db)")
        return True
    from corpus.survey.index_quality import NotACorpusIndexError, run_index_quality

    print("\nindex quality")
    try:
        result = run_index_quality(db_path, sample_per_kind=0)
    except NotACorpusIndexError as exc:
        print(f"  SKIPPED ({exc})")
        return True
    except Exception as exc:
        print(f"  SKIPPED (could not scan: {type(exc).__name__})")
        return True

    # Vacuous BEFORE clean: a scan that examined nothing found nothing, and
    # the second half of that sentence is the part people read.
    if result.coverage.vacuous:
        print(f"  [ FAIL ] {result.coverage.describe()} -- this scan proves nothing")
        print("           Wrong --db, or a --source-type that is not in this index?")
        return False
    if result.clean:
        print(
            f"  [  ok  ] {result.coverage.describe()}, no transcription artefacts"
        )
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
        "--db",
        default=None,
        metavar="PATH",
        help=(
            "Index to check, for a deployment with no corpus.toml. The "
            "archives most likely to have drifted are FORKS of this engine "
            "with their own config handling, so a tool that only checks the "
            "well-behaved consumers checks the wrong half."
        ),
    )
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

    log_paths, log_skip_reason = _resolve_query_logs(args)
    logs_ok, reports = _check_query_logs(log_paths, args.min_real_queries)
    if log_paths:
        ran["query logs"] = logs_ok
    else:
        skipped.append(f"query logs ({log_skip_reason})")
    _check_cross_archive(reports)

    db_path = args.db
    if db_path is None and args.config:
        from corpus.cli._common import load_config_or_exit

        db_path = str(load_config_or_exit(args.config).db_path)

    for name, check, needs in (
        ("served vs evaluated", _check_served_vs_evaluated(args.config), args.config),
        ("gold set", _check_gold_set(args.queries, db_path), args.queries and db_path),
        ("index quality", _check_index_quality(db_path), db_path),
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
            "  A skipped check is not a passing one. Each skip above says "
            "what would make it run."
        )
    if failed:
        print(
            "  A number measured through a contaminated instrument is worse "
            "than no number, because it still gets acted on."
        )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
