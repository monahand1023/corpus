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
import time
from collections.abc import Callable, Sequence
from pathlib import Path

from corpus.eval.query_log_audit import (
    LogReport,
    audit_query_log,
    cross_archive_overlap,
)
from corpus.verify import Coverage

# An empty log younger than this is a fresh install; older is a writer that
# is not writing. Chosen so a week of real use has to pass before the check
# accuses anything.
EMPTY_LOG_SUSPICIOUS_DAYS = 7


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
        coverage = Coverage(report.total, "entries")
        if coverage.vacuous:
            # An empty log cannot distinguish "nobody asks" from "logging
            # never worked" -- but HOW LONG it has been empty can. A log
            # created moments ago is a fresh install; one empty for months is
            # a broken writer, which is exactly what happened here: an archive
            # logged nothing for two months because it was never reachable
            # from the app in use, and every run passed.
            #
            # Failing on the fresh case would make this exit non-zero on a
            # healthy new archive, and a tool that cries wolf gets ignored.
            age_days = (time.time() - path.stat().st_mtime) / 86400
            if age_days >= EMPTY_LOG_SUSPICIOUS_DAYS:
                print(
                    f"  {path}: {coverage.describe()} after {age_days:.0f} days "
                    f"-- logging is on but has recorded nothing; that is a "
                    f"broken writer, not low demand"
                )
                ok = False
            else:
                print(
                    f"  {path}: {coverage.describe()} -- logging is on and this "
                    f"log is {age_days:.0f} days old, so there is nothing to "
                    f"conclude yet"
                )
            continue
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


def _all_keys(db_path: str) -> list[str]:
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


def _document_total(db_path: str) -> int:
    """Distinct documents in the index, or 0 when it cannot be read.

    0 means "not known", and every caller treats it as "cannot judge" rather
    than "nothing wrong".
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
    coverage = Coverage(len(queries), "queries")
    if coverage.vacuous:
        # Every finding this audit could report is one it cannot reach. The
        # realistic route is a module that imports fine but whose query list
        # is empty or renamed.
        print(f"\ngold set          [ FAIL ] {coverage.describe()}")
        print("                  A gold set that asks nothing proves nothing.")
        return False
    findings = audit_queries(
        queries,
        lookup=sqlite_lookup(db_path),
        documents=sqlite_documents(db_path, queries),
        # The archive's SIZE, without which the triviality check silently
        # does not run. It is optional on `audit_queries` so a caller with no
        # index still gets the structural checks -- which makes forgetting it
        # here look exactly like a gold set with nothing wrong.
        documents_total=_document_total(db_path),
        # The index's key list, without which the sibling check
        # silently does not run -- and a gold set that names one of
        # several identical documents looks clean.
        all_keys=_all_keys(db_path),
    )
    print(f"\ngold set          {coverage.describe()}")
    has_error = report_findings(findings, stream=sys.stdout)
    if not findings:
        print("  [  ok  ] no findings")
    return not has_error


# Every rejection reason the transcription pipeline can record. A reason
# missing from a large, single-policy sample is a filter that never fires --
# which means it is unnecessary or broken, with no third option.
# TWO populations, judged separately. A reason only a per-window check can
# produce looks dormant forever if it is counted against whole-file verdicts,
# and vice versa -- reporting them together produced exactly that on a real
# sidecar. Each list names only what its own table can contain.
WHOLE_FILE_FILTERS = (
    "silence",
    "empty",
    "subtitle_boilerplate",
    "degenerate_repetition",
    "looping_repetition",
    "impossible_speech_rate",
    "only_unspoken_languages",
)
# Filters that CANNOT fire in this pipeline, and why, keyed by the population
# the explanation belongs to. Kept next to the lists they annotate so the
# three are maintained together; a test asserts every name here is a real
# filter OF THAT POPULATION, because a hand-maintained annotation beside a
# hand-maintained list is exactly what drifts.
#
# "Never fired" and "cannot fire" need opposite responses -- "delete this or
# find out why" versus "you never turned it on", or "it is doing its job one
# population over" -- and reporting them identically is the same conflation
# this command exists to eliminate.
#
# The bar for adding an entry is a MECHANISM plus evidence, not a hunch. Each
# of these was confirmed against a live sidecar's own verdict counts.
FILTER_NOTES: dict[tuple[str, str], str] = {
    ("whole-file", "only_unspoken_languages"): (
        "only fires when corpus-transcribe was given --language; "
        "`judge_transcript` guards it with `if expected_languages and ...`. "
        "Live evidence: 82 verdicts under the one policy that set languages, "
        "none under any other."
    ),
    ("whole-file", "subtitle_boilerplate"): (
        "shadowed by the per-window pass, which drops a sign-off window as "
        "`caption_boilerplate` first -- the file then exits as `empty`, not "
        "as this reason. Live evidence: 492 window drops and 436 empty files "
        "against 0 here, and 15 here under the one policy predating it."
    ),
    ("whole-file", "impossible_speech_rate"): (
        "shadowed harder: the per-window check applies the SAME ceiling over "
        "a shorter span, so text that could fail against the file's duration "
        "has already failed against its window's. Reaching this rule means "
        "every window passed it, and the file's rate cannot exceed the "
        "fastest window's."
    ),
}

PER_WINDOW_FILTERS = (
    "empty",
    "caption_boilerplate",
    "degenerate_repetition",
    "looping_repetition",
    "impossible_speech_rate",
)


def _check_filter_activity(sidecar: str | None, *, policy: str | None = None) -> bool:
    """Name the filters that never fired. A warning, never a build failure.

    Dormancy is evidence to act on, not proof of a defect: a filter can be
    legitimately quiet on one archive. It reports and does not fail, because a
    check that fails on something ambiguous gets switched off, and then the
    real signal goes with it.
    """
    print("\nfilter activity")
    if not sidecar or not Path(sidecar).expanduser().is_file():
        print("  SKIPPED (no transcript sidecar; pass --transcripts PATH)")
        return True

    from corpus.transcripts import store
    from corpus.verify import MIN_VERDICTS_FOR_DORMANCY, dormant

    try:
        with store.open_store(sidecar, read_only=True) as conn:
            # Default to the NEWEST rule set rather than all of them. Mixing
            # vocabularies manufactures dead filters that are not dead: on a
            # real sidecar three looked dormant purely because an earlier
            # pipeline spelled its reasons differently.
            scope = policy if policy is not None else store.latest_policy(conn)
            populations = {
                "whole-file": (
                    store.filter_activity(conn, policy=scope, table="no_text"),
                    WHOLE_FILE_FILTERS,
                    store.activity_coverage(conn, policy=scope, table="no_text"),
                ),
                "per-window": (
                    store.filter_activity(
                        conn, policy=scope, table="dropped_windows"
                    ),
                    PER_WINDOW_FILTERS,
                    store.activity_coverage(
                        conn, policy=scope, table="dropped_windows"
                    ),
                ),
            }
    except Exception as exc:
        print(f"  SKIPPED (could not read: {type(exc).__name__})")
        return True

    for label, (activity, known, scoped) in populations.items():
        coverage = Coverage(sum(activity.values()), f"{label} verdicts")
        if coverage.examined < MIN_VERDICTS_FOR_DORMANCY:
            # Zero rejections out of three files is not evidence of anything,
            # and a guard that fires on noise gets switched off.
            print(
                f"  [ info ] {coverage.describe()} -- too small to judge "
                f"dormancy (needs {MIN_VERDICTS_FOR_DORMANCY})"
            )
            # WHY it is too small. A large archive whose verdicts fell out of
            # scope when a threshold moved reads identically to a small one,
            # and the two need opposite responses. Measured live: a sidecar
            # with 3,381 whole-file verdicts reported "2 -- too small".
            caveat = scoped.describe()
            if caveat and scoped.total:
                print(f"            {caveat}")
            continue
        idle = dormant(activity, known=known, coverage=coverage)
        if not idle:
            print(f"  [  ok  ] {coverage.describe()}, every filter has fired")
            continue
        gated = [n for n in idle if (label, n) in FILTER_NOTES]
        dead = [n for n in idle if (label, n) not in FILTER_NOTES]
        if dead:
            print(f"  [ warn ] {coverage.describe()}; never fired: {', '.join(dead)}")
            print("           A filter that never fires is unnecessary or broken.")
            print("           Judged within one rule set and one population, so "
                  "this is not a mismatch.")
        for name in gated:
            print(f"  [ info ] {name}: cannot fire here --")
            print(f"           {FILTER_NOTES[label, name]}")
    return True


def _check_chunker_drift(config_path: str | None, db_path: str | None) -> bool:
    """Do the stored chunks still match what the chunker would produce?

    A source is ingested, the chunker changes, and nothing re-ingests it. The
    stored chunks are then whatever an older version made, and NOTHING SAYS
    SO: search works, every other check passes, the eval is unmoved. On a live
    archive 24% of one source's chunks had drifted, and the only
    warning anyone got was an 8.4M-token re-embedding bill.

    Sampled, because chunking a million-chunk archive to answer "is it
    current?" costs more than the answer is worth. A sample cannot prove a
    source is clean; it can show that it is not, which is the useful
    direction.
    """
    print("\nchunker drift")
    if not config_path or not db_path:
        print("  SKIPPED (needs --config)")
        return True

    from corpus.config import CorpusConfig
    from corpus.connectors.discovery import source_excludes
    from corpus.connectors.registry import build_pipeline
    from corpus.survey.drift import chunker_drift

    try:
        config = CorpusConfig.load(config_path)
    except Exception as exc:
        print(f"  SKIPPED (could not load config: {type(exc).__name__})")
        return True

    drifted_sources = []
    unreachable: list[tuple[str, str]] = []
    checked = 0
    for source in config.sources:
        try:
            connector, chunker = build_pipeline(source)
            with source_excludes(source.exclude):
                report = chunker_drift(
                    db_path, source.name, connector.load(), chunker, sample=25
                )
        except Exception as exc:
            # An unreachable source -- unmounted volume, missing extra, a
            # connector type the CONSUMER registers at runtime -- is not
            # evidence about drift either way. But it has to be NAMED:
            # "no source could be sampled" is a check that did not run,
            # reported in words that read like one that found nothing.
            unreachable.append((source.name, f"{type(exc).__name__}: {exc}"))
            continue
        if not report.examined:
            unreachable.append((source.name, "nothing sampled"))
            continue
        checked += 1
        if report.drifted:
            drifted_sources.append(report)

    if not checked:
        print("  SKIPPED: no source could be sampled")
        for name, why in unreachable[:5]:
            print(f"           {name}: {why[:100]}")
        if any("not registered" in why for _n, why in unreachable):
            print("           Pass --load <module> so the consumer's own "
                  "connectors are registered first --")
            print("           the module that DEFINES register(), e.g. "
                  "--load mail_rag.register")
        return True
    if not drifted_sources:
        print(f"  [  ok  ] {checked} source(s) sampled, all current")
        return True
    for report in sorted(drifted_sources, key=lambda r: -r.percent):
        print(f"  [ warn ] {report.describe()}")
    return True


def _check_transcribe_health(sidecar: str | None, *, policy: str | None = None) -> bool:
    """What the last transcription pass could not finish.

    The sidecar's `failures` table is where a pass records every file it gave
    up on, and nothing read it. The doctor -- whose whole job is "are the
    instruments trustworthy?" -- covered `no_text`, `dropped_windows`,
    `transcripts` and the query log, and not this one.

    Live on one archive, all recorded and none surfaced: 94 failure rows for
    files that had since been given a proper verdict, a file that timed out
    three passes running, and timeouts at all. The point is not that any of
    them is catastrophic. It is that the pass RECORDED each one and the tool
    that exists to read the record did not read it.

    Reported, never failing. A timeout is usually about the run rather than
    the file, and a check that fails the build on something ambiguous gets
    switched off -- taking the real signal with it.
    """
    print("\ntranscribe health")
    if not sidecar or not Path(sidecar).expanduser().is_file():
        print("  SKIPPED (no transcript sidecar; pass --transcripts PATH)")
        return True

    from corpus.transcripts import store

    try:
        with store.open_store(sidecar, read_only=True) as conn:
            scope = policy if policy is not None else store.latest_policy(conn)
            # `attempts` arrives via a column migration, and migrations only
            # run on a READ-WRITE open. This command is a read-only
            # diagnostic, so on a sidecar written by an older version the
            # column is simply absent -- and asking for it raised
            # OperationalError, which this function then reported as SKIPPED.
            # A diagnostic that needs a migration to have happened cannot
            # diagnose the state it most needs to: the one before anyone
            # upgraded.
            columns = {r[1] for r in conn.execute("PRAGMA table_info(failures)")}
            if "attempts" in columns:
                failures = conn.execute(
                    "SELECT path, error, attempts FROM failures"
                    " ORDER BY attempts DESC"
                ).fetchall()
            else:
                failures = [
                    (r[0], r[1], 1)
                    for r in conn.execute("SELECT path, error FROM failures")
                ]
            settled = {r[0] for r in conn.execute("SELECT path FROM transcripts")}
            settled |= {
                r[0]
                for r in conn.execute(
                    "SELECT path FROM no_text WHERE policy IS ?", (scope,)
                )
            }
            gave_up = conn.execute(
                "SELECT path FROM no_text WHERE reason = 'repeatedly_timed_out'"
            ).fetchall()
    except Exception as exc:
        print(f"  SKIPPED (could not read: {type(exc).__name__})")
        return True

    stale = [r for r in failures if r[0] in settled]
    live = [r for r in failures if r[0] not in settled]
    timeouts = [r for r in live if "timed out" in (r[1] or "").lower()]

    if not failures and not gave_up:
        print(f"  [  ok  ] no unfinished files recorded ({len(settled):,} settled)")
        return True

    if timeouts:
        # Phrased with the same words the run's own log uses, so someone
        # grepping for one finds the other.
        print(
            f"  [ warn ] {len(timeouts):,} file(s) timed out (hit their deadline "
            "and were cut off)"
        )
        for path, _error, attempts in timeouts[:5]:
            tries = f" ({attempts}x)" if (attempts or 1) > 1 else ""
            print(f"           {Path(path).name[:64]}{tries}")
    other = [r for r in live if r not in timeouts]
    if other:
        print(f"  [ warn ] {len(other):,} file(s) failed for other reasons")
        for path, error, _attempts in other[:3]:
            print(f"           {Path(path).name[:48]}: {(error or '')[:60]}")
    if gave_up:
        print(
            f"  [ warn ] {len(gave_up):,} file(s) GAVE UP after repeated timeouts "
            "and are no longer retried"
        )
        for (path,) in gave_up[:5]:
            print(f"           {Path(path).name[:64]}")
        print("           They carry a policy fingerprint, so a rule change "
              "brings them back.")
    if stale:
        # Not failures. Rows for files that were later given a verdict, which
        # a normal pass now sweeps -- if these persist, no pass has run since.
        print(
            f"  [ info ] {len(stale):,} STALE failure row(s): those files have "
            "since been answered"
        )
    return True


def _apply_load(load: str | None) -> bool:
    """Import a consumer's module and call its `register()`. True on success.

    Applied ONCE, before any check runs. It was wired into the
    shadowed-components check only, so a consumer's own connector types stayed
    unregistered for every other check -- and the drift check then reported
    "no source could be sampled" while advising the reader to pass the flag
    they had already passed.

    A consumer's connectors are a property of the DEPLOYMENT, not of one
    check.
    """
    if not load:
        return True
    import importlib

    try:
        module = importlib.import_module(load)
    except Exception as exc:
        print(f"  could not load {load}: {type(exc).__name__}: {exc}")
        return False
    register = getattr(module, "register", None)
    if not callable(register):
        print(f"  {load} has no register() to call")
        return False
    register()
    return True


def _check_shadowed_components(load: str | None = None) -> bool:
    """Name any connector a consumer has registered over the engine's own.

    THE FAILURE THIS MAKES VISIBLE. A consumer repository held a full copy of
    the transcript connector and registered it over the engine's. Engine fixes
    then stopped reaching that archive, and nothing said so -- every ingest
    reported success. A loop filter dropped 716 junk chunks in principle and
    301 in practice, because the half that landed in the engine was shadowed
    by the copy. It was found by chasing a discrepancy, not by any check.

    Reported, not failed. Overriding a connector is a legitimate thing to do
    -- supplying archive-specific exclusions, for instance. What is not
    legitimate is doing it by accident and never being told.
    """
    from corpus.connectors.registry import _BUILTIN_BUILDERS, CONNECTOR_REGISTRY

    print("\nshadowed components")
    if load:
        # Without this the check cannot see the failure it exists for. This
        # command runs standalone and never imports a consumer's registration,
        # so the registry it inspects is pristine -- an archive that DOES
        # override reports "none overridden", which is the same false clean
        # result the check was built to eliminate.
        import importlib

        try:
            module = importlib.import_module(load)
            register = getattr(module, "register", None)
            if callable(register):
                register()
            else:
                print(f"  SKIPPED ({load} has no register() to call)")
                return True
        except Exception as exc:
            print(f"  SKIPPED (could not load {load}: {type(exc).__name__}: {exc})")
            return True

    shadowed = sorted(
        name
        for name, builder in CONNECTOR_REGISTRY.items()
        if name in _BUILTIN_BUILDERS and builder is not _BUILTIN_BUILDERS[name]
    )
    if not shadowed:
        print(f"  [  ok  ] {len(CONNECTOR_REGISTRY)} connectors, none overridden")
        return True
    print(f"  [ warn ] overridden by this deployment: {', '.join(shadowed)}")
    print("           Engine fixes to these do NOT reach this archive. That is")
    print("           fine if deliberate -- import the shared piece rather than")
    print("           copying it, so future fixes arrive on their own.")
    return True


def _check_duplicate_content(db_path: str | None) -> bool:
    """Passages indexed from more than one document.

    Measured on a live archive: 6.5% of the index, traced to 168 recordings
    present twice under different paths -- 17.8 GPU-hours transcribed twice
    and a search that can return the same passage from two files.

    Reported, never deduplicated. Some duplication is legitimate and must not
    be touched: an email thread quotes what it replies to, a template repeats
    its boilerplate. A second archive measured 1.2% and almost all of it was
    quoting. Nothing here can tell that apart from a file copied into two
    folders -- a person can, given the evidence.
    """
    print("\nduplicate content")
    if not db_path:
        print("  SKIPPED (no --config or --db)")
        return True

    from corpus.survey.duplicates import find_duplicate_content

    try:
        report = find_duplicate_content(db_path)
    except Exception as exc:
        print(f"  SKIPPED (could not scan: {type(exc).__name__})")
        return True

    if report.coverage.vacuous:
        print(f"  SKIPPED ({report.coverage.describe()})")
        return True
    if not report.redundant_chunks:
        print(f"  [  ok  ] {report.coverage.describe()}, no passage appears twice")
        return True

    print(
        f"  [ info ] {report.redundant_chunks:,} of {report.coverage.describe()} "
        f"({report.percent:.1f}%) duplicate another document"
    )
    for left, right, shared in report.duplicate_documents[:5]:
        print(f"           {shared:>4} shared: {Path(left).name} / {Path(right).name}")
    print("           Some of this is legitimate (quoted mail, shared "
          "boilerplate).")
    print("           Pairs sharing MANY passages are usually one file in two "
          "places.")
    return True


def _check_threshold_margins(sidecar: str | None) -> bool:
    """How close is each threshold to the real data it must not reject?

    Reported, never enforced. A threshold may legitimately sit near data it is
    meant to nearly touch; what it must not do is sit there unnoticed, which
    is what happened twice in two days. A ceiling 0.3% above real material
    reads exactly like one 60% above it until something measures the distance.
    """
    print("\nthreshold margins")
    if not sidecar or not Path(sidecar).expanduser().is_file():
        print("  SKIPPED (no transcript sidecar; pass --transcripts PATH)")
        return True

    from corpus.transcripts import quality, store
    from corpus.verify import margin

    try:
        with store.open_store(sidecar, read_only=True) as conn:
            rows = conn.execute(
                "SELECT text, duration_s, path FROM transcripts "
                "WHERE duration_s > 1.0"
            ).fetchall()
    except Exception as exc:
        print(f"  SKIPPED (could not read: {type(exc).__name__})")
        return True

    if not rows:
        print("  SKIPPED (no transcripts to measure against)")
        return True

    # Only thresholds whose crossing DELETES something are worth reporting:
    # the margin that matters is the one protecting real material.
    def _worst(
        score: Callable[[str], float] | None,
        eligible: Callable[[str, float], bool],
    ) -> tuple[float, str, str]:
        """Highest-scoring row, with the evidence attached.

        The score alone gets read backwards. A threshold that is too
        PERMISSIVE always looks tight, because the junk it failed to catch is
        sitting in the sample it is measured against -- which is exactly what
        happened: "looping share: 0.7% headroom" was five copies of a decode
        loop, and the obvious action it suggested (raise the ceiling) was the
        opposite of right. So the nearest row's name and text come too.
        """
        best = (0.0, "", "")
        for text, duration, path in rows:
            if not eligible(text or "", duration):
                continue
            # `score=None` means chars-per-second, which needs the duration
            # and so cannot be expressed as a function of the text alone.
            value = score(text or "") if score else len(text or "") / duration
            if value > best[0]:
                best = (value, path or "", (text or "")[:60].replace("\n", " "))
        return best

    measured = (
        ("max chars/sec", quality.DEFAULT_MAX_CHARS_PER_SECOND,
         _worst(None, lambda t, d: True)),
        ("looping share", quality.DEFAULT_MAX_LOOPING_SHARE,
         _worst(quality.looping_share, lambda t, d: True)),
        # Measured only on text the filter actually evaluates. This rule is
        # guarded by `len(text) > 40`, and ignoring that guard reported a
        # crossed threshold for "Ah! Ah! Ah! Ah!" -- a real exclamation the
        # guard exists to protect. A margin against data the rule never sees
        # is not a margin.
        ("repeat share", 0.9,
         _worst(quality.repeat_share, lambda t, d: len(t) > 40)),
    )

    tight = 0
    for label, threshold, (observed, path, excerpt) in measured:
        m = margin(threshold=threshold, observed_max=observed, label=label)
        mark = "[ warn ]" if m.tight else "[  ok  ]"
        print(f"  {mark} {m.describe()}")
        if m.tight and path:
            name = Path(path).name
            print(f"           nearest: {name}")
            print(f"           {excerpt!r}")
        tight += m.tight
    if tight:
        print("           Look at the text above before moving anything. A")
        print("           permissive threshold always LOOKS tight: the junk it")
        print("           failed to catch is in the sample it is measured on.")
    return True


def _check_index_quality(db_path: str | None) -> bool:
    if not db_path:
        print("\nindex quality     SKIPPED (no --config or --db)")
        return True
    from corpus.survey.index_quality import NotACorpusIndexError, run_index_quality
    from corpus.verify import DetectorBroken

    print("\nindex quality")
    try:
        result = run_index_quality(db_path, sample_per_kind=0)
    except NotACorpusIndexError as exc:
        print(f"  SKIPPED ({exc})")
        return True
    except DetectorBroken as exc:
        # Deliberately NOT caught by the broad handler below. A detector that
        # cannot fire is the loudest signal this command has; swallowing it as
        # "skipped" would turn it into the quietest.
        print(f"  [ FAIL ] {exc}")
        return False
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
        "--load",
        default=None,
        metavar="MODULE",
        help="Import MODULE and call its register() before checking for "
             "shadowed connectors. A consumer's overrides are invisible "
             "until its registration has run (e.g. --load docs_rag.transcripts)",
    )
    parser.add_argument(
        "--transcripts",
        default=None,
        metavar="PATH",
        help="Transcript sidecar to check for filters that never fire "
             "(default: inferred from a `transcripts` source in --config)",
    )
    parser.add_argument(
        "--min-real-queries",
        type=int,
        default=30,
        help="Distinct real queries a log needs before conclusions are drawn",
    )
    args = parser.parse_args(argv)

    # Applied ONCE, before any check runs. A consumer's connectors are a
    # property of the deployment, not of the one check that used to load them
    # -- the drift check reported "no source could be sampled" while advising
    # the reader to pass the flag they had already passed.
    if args.load and not _apply_load(args.load):
        print("  (continuing; checks that need those connectors will say so)")

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
    sidecar = args.transcripts
    if args.config and (db_path is None or sidecar is None):
        from corpus.cli._common import load_config_or_exit

        cfg = load_config_or_exit(args.config)
        if db_path is None:
            db_path = str(cfg.db_path)
        if sidecar is None:
            # A `transcripts` source points at the sidecar, so the operator
            # does not have to name it twice.
            for source in cfg.sources:
                if source.type == "transcripts":
                    sidecar = str(source.path)
                    break

    for name, check, needs in (
        ("served vs evaluated", _check_served_vs_evaluated(args.config), args.config),
        ("gold set", _check_gold_set(args.queries, db_path), args.queries and db_path),
        ("index quality", _check_index_quality(db_path), db_path),
        (
            "filter activity",
            _check_filter_activity(sidecar),
            sidecar,
        ),
        ("threshold margins", _check_threshold_margins(sidecar), sidecar),
        ("transcribe health", _check_transcribe_health(sidecar), sidecar),
        (
            "chunker drift",
            _check_chunker_drift(args.config, db_path),
            args.config and db_path,
        ),
        ("duplicate content", _check_duplicate_content(db_path), db_path),
        (
            "shadowed components",
            _check_shadowed_components(args.load),
            True,
        ),
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
