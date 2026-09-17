"""A query a coin flip passes is a query that measures nothing.

recall@k asks whether any acceptable answer landed in the top k. When a
query accepts 14% of an archive's 196,575 documents, five random results
contain one of them 53% of the time -- so the query scores ~0.5 for a
retriever that does not work at all, and its contribution to the gate is
mostly noise wearing the shape of a result.

This is the same defect as every other one in this codebase: a check that
cannot fail is indistinguishable from a check that passes. It just arrives
through the gold set instead of through the code.

Measured on a real archive, which is why the threshold is not stricter:
of 24 queries, 23 sat at or below 0.094 and one sat at 0.535. The set as a
whole scored 0.050 for a random retriever against a floor of 0.708 -- the
gate was overwhelmingly earned, and exactly one query was not pulling its
weight. A check that flagged the other 23 would be wrong and would get
switched off.
"""

from __future__ import annotations

import pytest

from corpus.eval.triviality import (
    DEFAULT_TRIVIAL_ABOVE,
    random_hit_rate,
    triviality_report,
)


def test_a_query_accepting_most_of_the_archive_is_flagged():
    report = triviality_report(
        {"everything anyone ever photographed": 14_000, "one specific bridge": 7},
        documents=100_000,
        top_k=5,
    )
    assert report.trivial == ["everything anyone ever photographed"]
    assert not report.is_clean
    assert "14,000" in report.describe()


def test_a_normal_gold_set_is_not_flagged():
    """23 of 24 real queries, verbatim. If this trips, the check is useless:
    every archive with broad topical queries would fail it and it would be
    turned off, taking the real signal with it."""
    sizes = {
        f"q{i}": n for i, n in enumerate(
            [3855, 3710, 2360, 1919, 1458, 1455, 1301, 1202, 1187, 1016,
             941, 929, 923, 908, 654, 610, 607, 597, 368, 231, 200, 38, 7]
        )
    }
    report = triviality_report(sizes, documents=100_000, top_k=5)
    assert report.is_clean, report.describe()


def test_the_set_wide_floor_is_what_a_random_retriever_would_score():
    """The number that says whether a gate is earned at all. A floor of
    0.708 over a set a coin flip scores 0.700 on is not a gate."""
    report = triviality_report(
        {"a": 14_000, "b": 14_000}, documents=100_000, top_k=5
    )
    assert report.random_recall == pytest.approx(0.530, abs=0.01)


def test_an_empty_archive_does_not_divide_by_zero():
    """Zero documents means "I cannot judge this", not "everything is fine"."""
    report = triviality_report({"a": 5}, documents=0, top_k=5)
    assert report.random_recall == 0.0
    assert report.is_clean
    assert "cannot" in report.describe().lower()


def test_a_negative_query_is_not_counted_as_trivial():
    """A query with no acceptable answer is a NEGATIVE control -- it exists
    to check that nothing is returned. Scoring it for triviality would flag
    the one query shape that is deliberately unanswerable."""
    report = triviality_report({"nothing matches this": 0}, documents=1000, top_k=5)
    assert report.is_clean
    assert report.random_recall == 0.0


def test_more_keys_than_documents_is_reported_not_silently_clamped():
    """A gold set naming more answers than the archive holds is a bug in the
    gold set. Clamping to 1.0 would hide it behind a plausible number."""
    with pytest.raises(ValueError, match="more acceptable answers"):
        triviality_report({"a": 2000}, documents=1000, top_k=5)


def test_the_hit_rate_matches_the_closed_form():
    # 1 - (1 - K/N)^k
    assert random_hit_rate(keys=1, documents=2, top_k=1) == pytest.approx(0.5)
    assert random_hit_rate(keys=0, documents=100, top_k=5) == 0.0
    assert random_hit_rate(keys=100, documents=100, top_k=1) == pytest.approx(1.0)


def test_the_default_threshold_is_a_coin_flip():
    """Stated once, here, because a threshold nobody can justify is the thing
    this project keeps finding at the bottom of a broken check."""
    assert DEFAULT_TRIVIAL_ABOVE == 0.5


def test_the_eval_reports_the_random_baseline_beside_the_score(tmp_path, capsys):
    """A number with nothing to compare it to is the reason a 0.708 floor
    over a set chance scores 0.700 on looks like a passing gate.

    This is the part that makes the check real: it runs on every eval,
    beside the metric it qualifies, rather than being a library nobody
    calls.
    """
    from corpus.cli.eval import _report_triviality

    _report_triviality(
        key_counts={"broad": 14_000, "narrow": 7},
        documents=100_000,
        top_k=5,
    )
    out = capsys.readouterr().out

    assert "random retriever" in out
    assert "0.265" in out or "0.264" in out, out  # (0.530 + 0.000) / 2
    assert "broad" in out, "the offending query was not named"


def test_the_eval_says_nothing_when_the_gold_set_is_sound(tmp_path, capsys):
    """Silence on the clean path. A line printed every run stops being read,
    and this one has to be read on the run where it matters."""
    from corpus.cli.eval import _report_triviality

    _report_triviality(key_counts={"a": 7, "b": 38}, documents=100_000, top_k=5)
    out = capsys.readouterr().out

    assert out.strip() == "", f"printed on a clean gold set:\n{out}"


# --- as a gold-set finding, so the doctor and the forks get it too -----------


class _Q:
    def __init__(self, query, expected_keys):
        self.query = query
        self.expected_keys = expected_keys


def test_the_audit_flags_a_trivially_satisfiable_query():
    """`corpus-eval` is not the only thing that reads a gold set, and it is
    the expensive one -- it spends embedding calls to find out.

    `audit_queries` is what `corpus-doctor` runs and what the forked private
    eval CLIs import, so a check that lives only in the eval CLI is a check
    half the archives never run. This one needs no retrieval at all: it is
    arithmetic over the answer-set sizes.
    """
    from corpus.eval.goldset import audit_queries

    findings = audit_queries(
        [_Q("broad", [f"k{i}" for i in range(600)]), _Q("narrow", ["a"])],
        documents_total=1000,
        top_k=5,
    )
    trivial = [f for f in findings if f.kind == "trivially_satisfiable"]
    assert len(trivial) == 1, [f.kind for f in findings]
    assert trivial[0].query == "broad"
    assert trivial[0].severity == "warning"
    assert "600" in trivial[0].detail


def test_the_audit_is_unchanged_when_it_is_not_told_the_archive_size():
    """Optional, like `lookup` and `documents`: the structural checks have to
    keep working before an index exists. Absent a size, this check cannot run
    -- and must not invent a verdict."""
    from corpus.eval.goldset import audit_queries

    findings = audit_queries([_Q("broad", [f"k{i}" for i in range(600)])])
    assert not [f for f in findings if f.kind == "trivially_satisfiable"]


def test_the_doctor_passes_the_archive_size_so_the_check_can_run(tmp_path, capsys):
    """A check wired up but never given its input is a check that reports
    clean forever -- the defect this codebase has shipped in four places.

    `audit_queries` takes `documents_total` as an OPTIONAL argument and
    silently skips triviality without it. That is right for a caller with no
    index, and lethal for the doctor, which has one.
    """
    import sqlite3

    from corpus.cli.doctor import _check_gold_set

    db = tmp_path / "index.db"
    conn = sqlite3.connect(db)
    conn.execute(
        "CREATE TABLE chunks (id TEXT, source_type TEXT, source_key TEXT,"
        " content TEXT, content_hash TEXT, metadata TEXT)"
    )
    for i in range(100):
        conn.execute(
            "INSERT INTO chunks VALUES (?,'notes',?,'body','h','{}')",
            (f"c{i}", f"doc{i}.md"),
        )
    conn.commit()
    conn.close()

    queries = tmp_path / "eval_queries.py"
    keys = ", ".join(f'"doc{i}.md"' for i in range(60))
    # A dataclass, not a dict: dicts are what `unusable-query-shape` rejects,
    # and corpus-eval could not run them either.
    queries.write_text(
        "from dataclasses import dataclass, field\n"
        "@dataclass\n"
        "class Q:\n"
        "    query: str\n"
        "    expected_keys: list\n"
        f'EVAL_QUERIES = [Q("everything", [{keys}])]\n'
    )

    _check_gold_set(str(queries), str(db))
    out = capsys.readouterr().out

    assert "trivially_satisfiable" in out, (
        f"the doctor never gave the audit the archive size:\n{out}"
    )


def test_a_gold_set_the_eval_cannot_run_is_named_as_such():
    """The doctor and the eval disagreed about what a query IS.

    `corpus-eval` reads `q.query` and `q.expected_keys` as attributes, so a
    gold set written as dicts raises AttributeError and never runs. The
    audit reads the same fields with `getattr(..., default)`, so the same
    file passed through it as a list of empty NEGATIVE controls -- and
    reported "no expected keys ... if the keys come from a labelling
    function, it matched nothing", which sends someone to debug a labelling
    function that is not there.

    Two readings of the same file, and the diagnostic was the forgiving one.
    """
    from corpus.eval.goldset import audit_queries

    findings = audit_queries([{"query": "ducks", "expected_keys": ["a.md"]}])
    kinds = [f.kind for f in findings]

    assert "unusable-query-shape" in kinds, kinds
    bad = next(f for f in findings if f.kind == "unusable-query-shape")
    assert bad.severity == "error"
    assert "dict" in bad.detail
    assert "empty-answer-set" not in kinds, (
        "it is still being read as a negative control as well"
    )


# --- siblings the label left out ---------------------------------------------


def test_an_unlisted_sibling_of_an_expected_key_is_flagged():
    """Four labels in one gold set named one document of several identical ones.

    An archive keeps siblings: a doorbell clip in the iMovie library AND on
    the drive it came from; five ReleaseNotes.txt in one driver package; a
    Spanish handbook for each of two schools; 31 localisation files. A label
    naming one of them asserts the others do NOT answer the question, so the
    retriever returns a correct document and is scored wrong -- and the query
    looks like a retrieval failure worth chasing.

    Measured before shipping: 3 of 24 queries on the archive where the
    mistakes were made, 0 of 24 on each of two others. It fires where the
    mistake is and stays quiet elsewhere, which is the only reason it is
    worth having.
    """
    from corpus.eval.goldset import audit_queries

    findings = audit_queries(
        [_Q("release notes", ["pkg/audio/HDABus/ReleaseNotes.txt"])],
        all_keys=[
            "pkg/audio/HDABus/ReleaseNotes.txt",
            "pkg/audio/HDMI/ReleaseNotes.txt",
            "pkg/audio/SAFD/ReleaseNotes.txt",
            "unrelated/notes.md",
        ],
    )
    sib = [f for f in findings if f.kind == "unlisted-sibling"]
    assert len(sib) == 1, [f.kind for f in findings]
    assert sib[0].severity == "warning"
    assert "2" in sib[0].detail
    assert "unrelated/notes.md" not in sib[0].detail


def test_a_complete_label_is_not_flagged():
    from corpus.eval.goldset import audit_queries

    findings = audit_queries(
        [_Q("release notes", ["pkg/a/ReleaseNotes.txt", "pkg/b/ReleaseNotes.txt"])],
        all_keys=["pkg/a/ReleaseNotes.txt", "pkg/b/ReleaseNotes.txt"],
    )
    assert not [f for f in findings if f.kind == "unlisted-sibling"]


def test_without_the_key_list_the_sibling_check_does_not_guess():
    from corpus.eval.goldset import audit_queries

    findings = audit_queries([_Q("a", ["pkg/a/ReleaseNotes.txt"])])
    assert not [f for f in findings if f.kind == "unlisted-sibling"]


def test_the_doctor_passes_the_key_list_so_the_sibling_check_can_run(tmp_path, capsys):
    """`all_keys` is optional, so forgetting it looks exactly like a gold set
    with nothing wrong. Same trap as `documents_total`, same test."""
    import sqlite3

    from corpus.cli.doctor import _check_gold_set

    db = tmp_path / "index.db"
    conn = sqlite3.connect(db)
    conn.execute(
        "CREATE TABLE chunks (id TEXT, source_type TEXT, source_key TEXT,"
        " content TEXT, content_hash TEXT, metadata TEXT)"
    )
    for key in ("pkg/a/ReleaseNotes.txt", "pkg/b/ReleaseNotes.txt"):
        conn.execute(
            "INSERT INTO chunks VALUES (?,'notes',?,'body','h','{}')", (key, key)
        )
    conn.commit()
    conn.close()

    queries = tmp_path / "eval_queries.py"
    queries.write_text(
        "from dataclasses import dataclass\n"
        "@dataclass\n"
        "class Q:\n"
        "    query: str\n"
        "    expected_keys: list\n"
        'EVAL_QUERIES = [Q("release notes", ["pkg/a/ReleaseNotes.txt"])]\n'
    )

    _check_gold_set(str(queries), str(db))
    out = capsys.readouterr().out

    assert "unlisted-sibling" in out, (
        f"the doctor never gave the audit the key list:\n{out}"
    )


def test_the_EVAL_runs_the_same_gold_set_audit_the_doctor_does():
    """Two commands read the same gold set and applied different audits.

    `corpus-doctor` passes `all_keys=` and `documents_total=`, with inline
    comments warning that omitting them silently disables two checks.
    `corpus-eval` -- the command that prints "Refusing to run: the answer key
    is broken" -- omitted both, so the command with the authority to REFUSE
    was the one running the weaker audit.

    Checked by reading the call site: the audit's optional arguments are
    exactly the shape that makes forgetting them invisible, so the test has
    to assert they are passed rather than assert on behaviour that looks
    identical either way.
    """
    import ast
    from pathlib import Path

    tree = ast.parse(Path("src/corpus/cli/eval.py").read_text())
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and getattr(node.func, "id", getattr(node.func, "attr", "")) == "audit_queries"
    ]
    assert calls, "audit_queries is no longer called from corpus-eval at all"
    for call in calls:
        passed = {kw.arg for kw in call.keywords}
        missing = {"all_keys", "documents_total"} - passed
        assert not missing, (
            f"corpus-eval calls audit_queries without {sorted(missing)}, which "
            "silently disables the sibling and triviality checks"
        )
