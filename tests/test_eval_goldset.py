"""Tests for the gold-set audit.

Every case here is a defect that actually shipped in a gold set and was
mistaken for a retrieval failure. That is the expensive mistake: the response
to a broken answer key that looks like a broken retriever is to go and tune
the retriever, which was working.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from corpus.eval.goldset import audit_queries


@dataclass(frozen=True)
class Q:
    query: str
    expected_keys: list[str] = field(default_factory=list)


def _kinds(findings) -> set[str]:
    return {f.kind for f in findings}


def _indexed(*keys: str):
    known = set(keys)
    return lambda candidates: {k for k in candidates if k in known}


# --- an answer set that matched nothing ------------------------------------
# A labelling function that returns no keys makes the query a "negative", which
# is silently EXCLUDED from the metrics rather than failing. A mail gold set
# shipped with a thread whose subject pattern matched zero messages, and the
# aggregate simply did not mention it.


def test_an_empty_answer_set_is_reported() -> None:
    findings = audit_queries([Q("a quote from a moving company", [])])
    assert "empty-answer-set" in _kinds(findings)


def test_an_empty_answer_set_is_a_warning_not_an_error() -> None:
    # It is indistinguishable from a deliberate negative query, so it must not
    # block a run -- only be impossible to miss.
    findings = audit_queries([Q("nothing should match this", [])])
    assert all(f.severity == "warning" for f in findings)


# --- keys that are not in the index ----------------------------------------


def test_an_answer_set_absent_from_the_index_is_an_error() -> None:
    findings = audit_queries(
        [Q("stale", ["gone-1", "gone-2"])], lookup=_indexed("other")
    )
    assert "answer-not-in-index" in _kinds(findings)
    assert any(f.severity == "error" for f in findings)


def test_a_partly_absent_answer_set_is_only_a_warning() -> None:
    # One stale key among many still leaves a query that can pass honestly.
    findings = audit_queries(
        [Q("mostly fine", ["a", "b", "gone"])], lookup=_indexed("a", "b")
    )
    assert "answer-partly-not-in-index" in _kinds(findings)
    assert all(f.severity == "warning" for f in findings)


def test_no_finding_when_every_key_is_present() -> None:
    findings = audit_queries([Q("fine", ["a", "b"])], lookup=_indexed("a", "b"))
    assert findings == []


# --- a cap in the labelling code -------------------------------------------
# A photo gold set capped answer sets at 400 keys while the largest correct
# answer had 1,456 members. The retriever returned a perfectly good photo from
# the right trip, the key did not list it, and recall read 0.500 instead of
# 0.625.


def test_several_answer_sets_at_the_same_round_size_are_flagged() -> None:
    queries = [
        Q("trip one", [f"a{i}" for i in range(400)]),
        Q("trip two", [f"b{i}" for i in range(400)]),
    ]
    assert "possible-cap" in _kinds(audit_queries(queries))


def test_one_set_at_a_round_size_is_not_enough_to_flag() -> None:
    # A single set of exactly 100 is ordinary; a cap shows up as a REPEATED
    # size, and flagging every round number would make the audit noise.
    queries = [
        Q("trip one", [f"a{i}" for i in range(100)]),
        Q("trip two", [f"b{i}" for i in range(37)]),
    ]
    assert "possible-cap" not in _kinds(audit_queries(queries))


def test_sets_sharing_a_non_round_size_are_not_flagged() -> None:
    queries = [
        Q("trip one", [f"a{i}" for i in range(37)]),
        Q("trip two", [f"b{i}" for i in range(37)]),
    ]
    assert "possible-cap" not in _kinds(audit_queries(queries))


# --- a query copied out of its own answer ----------------------------------
# The point of paraphrasing a gold query away from the target's wording is that
# a query sharing the answer's phrasing is answered by BM25 whatever the
# embedder does. This check found a real one in a shipped mail gold set.


def test_a_query_sharing_a_phrase_with_its_answer_is_flagged() -> None:
    findings = audit_queries(
        [Q("the weekly bulletin from our congregation", ["doc"])],
        documents={"doc": "This week's news from our congregation follows."},
    )
    assert "query-echoes-answer" in _kinds(findings)


def test_a_paraphrased_query_is_not_flagged() -> None:
    findings = audit_queries(
        [Q("the newsletter we get from where we worship", ["doc"])],
        documents={"doc": "This week's news from our congregation follows."},
    )
    assert "query-echoes-answer" not in _kinds(findings)


def test_one_or_two_shared_words_are_not_an_echo() -> None:
    # Ordinary shared vocabulary must not trip it, or every query about a
    # subject would be flagged for mentioning that subject.
    findings = audit_queries(
        [Q("photos of the children at dinner", ["doc"])],
        documents={"doc": "The children ran outside after we finished dinner."},
    )
    assert "query-echoes-answer" not in _kinds(findings)


# --- duplicates ------------------------------------------------------------


def test_a_repeated_query_is_reported_once() -> None:
    queries = [Q("same", ["a"]), Q("same", ["a"]), Q("different", ["a"])]
    findings = [f for f in audit_queries(queries) if f.kind == "duplicate-query"]
    assert len(findings) == 1


def test_duplicate_detection_ignores_case_and_padding() -> None:
    queries = [Q("Same Query", ["a"]), Q("  same query  ", ["a"])]
    assert "duplicate-query" in _kinds(audit_queries(queries))


# --- usable before an index exists -----------------------------------------


def test_structural_checks_run_without_a_store() -> None:
    queries = [Q("empty", []), Q("dup", ["a"]), Q("dup", ["a"])]
    kinds = _kinds(audit_queries(queries))
    assert {"empty-answer-set", "duplicate-query"} <= kinds


def test_errors_sort_before_warnings() -> None:
    queries = [Q("empty", []), Q("stale", ["gone"])]
    findings = audit_queries(queries, lookup=_indexed("other"))
    assert findings[0].severity == "error"


# --- the index-reading helpers ---------------------------------------------
# These live in the library, not in a CLI, because the private archives run
# their own forked eval CLIs. A check only half the deployments run is a check
# only half the deployments get.


def _index(tmp_path, rows):
    import sqlite3

    path = tmp_path / "index.db"
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE chunks (id TEXT PRIMARY KEY, source_key TEXT, content TEXT)"
    )
    conn.executemany(
        "INSERT INTO chunks (id, source_key, content) VALUES (?,?,?)",
        [(str(i), k, c) for i, (k, c) in enumerate(rows)],
    )
    conn.commit()
    conn.close()
    return path


def test_sqlite_lookup_reports_only_keys_that_exist(tmp_path) -> None:
    from corpus.eval.goldset import sqlite_lookup

    db = _index(tmp_path, [("a", "text a"), ("b", "text b")])
    assert sqlite_lookup(db)(["a", "b", "missing"]) == {"a", "b"}


def test_sqlite_lookup_batches_past_the_sqlite_variable_limit(tmp_path) -> None:
    # SQLite caps host parameters per statement, so a 1,456-key answer set has
    # to be chunked. Getting this wrong fails only on the biggest sets, which
    # are exactly the ones a truncation bug already makes suspicious.
    from corpus.eval.goldset import sqlite_lookup

    keys = [f"k{i}" for i in range(1500)]
    db = _index(tmp_path, [(k, "x") for k in keys])
    assert sqlite_lookup(db)(keys) == set(keys)


def test_sqlite_lookup_treats_an_unreadable_index_as_all_present(tmp_path) -> None:
    # An audit that cannot read the index must not manufacture findings about
    # it -- reporting every key as missing would fail every query loudly.
    from corpus.eval.goldset import sqlite_lookup

    assert sqlite_lookup(tmp_path / "nope.db")(["a"]) == {"a"}


def test_sqlite_documents_samples_rather_than_reading_whole_answer_sets(tmp_path) -> None:
    from corpus.eval.goldset import sqlite_documents

    db = _index(tmp_path, [(f"k{i}", f"body {i}") for i in range(100)])
    q = Q("anything", [f"k{i}" for i in range(100)])
    assert len(sqlite_documents(db, [q], sample_per_query=5)) == 5


def test_report_findings_signals_whether_anything_was_an_error() -> None:
    import io

    from corpus.eval.goldset import GoldFinding, report_findings

    warn = GoldFinding("q", "kind", "detail", "warning")
    err = GoldFinding("q", "kind", "detail", "error")
    assert report_findings([warn], stream=io.StringIO()) is False
    assert report_findings([warn, err], stream=io.StringIO()) is True
    assert report_findings([], stream=io.StringIO()) is False
