"""A filter that never fires is either unnecessary or broken.

There is no third option, and this codebase has already had one: a repetition
check thresholded at 0.9 whose highest score across 74 real transcripts was
0.250. It had never once fired on real data, and nothing said so -- a dead
knob reads exactly like a knob that has nothing to reject.

Making that visible needs two things: the reason each rejection fired has to
be PERSISTED, and the set of reasons that never fired has to be reported.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from corpus.transcripts import store
from corpus.transcripts.store import Transcript, Window
from corpus.verify import Coverage, dormant


def _store(tmp_path: Path) -> sqlite3.Connection:
    return sqlite3.connect(tmp_path / "t.db")


# --- the reason must survive the write ---------------------------------------


def test_a_discarded_window_records_WHICH_rule_rejected_it(tmp_path: Path) -> None:
    """It recorded the text and the model's scores, but not the verdict.

    The docstring on save_dropped_windows says "without it the only record of
    thousands of destructive decisions is a count" -- and it dropped the one
    field that says which rule fired, which is what makes the filter auditable
    rather than merely visible.
    """
    with store.open_store(tmp_path / "t.db") as conn:
        store.save_dropped_windows(
            conn, "/w/a.m4a",
            [{"window_start": 0.0, "no_speech": 0.8, "avg_logprob": -0.2,
              "text": "Thank you.", "reason": "caption_boilerplate"}],
            policy="p",
        )
        row = conn.execute(
            "SELECT reason FROM dropped_windows WHERE path = ?", ("/w/a.m4a",)
        ).fetchone()
    assert row[0] == "caption_boilerplate"


def test_an_older_sidecar_gains_the_column_by_migration(tmp_path: Path) -> None:
    db = tmp_path / "old.db"
    conn = sqlite3.connect(db)
    conn.executescript(
        "CREATE TABLE dropped_windows (path TEXT, window_start REAL,"
        " no_speech REAL, avg_logprob REAL, text TEXT, policy TEXT);"
    )
    conn.commit()
    conn.close()

    with store.open_store(db) as conn:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(dropped_windows)")}
    assert "reason" in cols


# --- reporting which filters never fired -------------------------------------


def test_filter_activity_counts_every_reason_recorded(tmp_path: Path) -> None:
    with store.open_store(tmp_path / "t.db") as conn:
        store.save_no_text(conn, "/w/a.m4a", duration_s=1.0, policy="p", reason="silence")
        store.save_no_text(conn, "/w/b.m4a", duration_s=1.0, policy="p", reason="silence")
        store.save_dropped_windows(
            conn, "/w/c.m4a",
            [{"window_start": 0.0, "text": "x", "reason": "looping_repetition"}],
            policy="p",
        )
        counts = store.filter_activity(conn)

    assert counts["silence"] == 2
    assert counts["looping_repetition"] == 1


def test_dormant_names_the_filters_that_never_fired() -> None:
    known = ["silence", "looping_repetition", "impossible_speech_rate"]
    counts = {"silence": 40, "looping_repetition": 6}
    assert dormant(counts, known=known, coverage=Coverage(500, "verdicts")) == [
        "impossible_speech_rate"
    ]


def test_dormant_says_nothing_on_a_sample_too_small_to_mean_anything() -> None:
    """Zero rejections out of three files is not evidence of a dead filter.

    Reporting it there would be the cry-wolf failure: a guard that fires on
    noise gets switched off, which leaves you worse off than no guard.
    """
    counts = {"silence": 1}
    assert dormant(counts, known=["silence", "looping_repetition"],
                   coverage=Coverage(3, "verdicts")) == []


def test_dormant_reports_nothing_when_every_filter_has_fired() -> None:
    counts = {"a": 1, "b": 2}
    assert dormant(counts, known=["a", "b"], coverage=Coverage(900, "verdicts")) == []


def test_activity_can_be_scoped_to_one_policy(tmp_path: Path) -> None:
    """Dormancy across policy versions is a false positive, not a finding.

    Measured on a real 7,445-transcript sidecar: three filters looked dormant
    only because that archive was written by an earlier pipeline whose reason
    vocabulary differed ("no_speech_detected" where the current one records
    "silence"). Comparing a current filter list against verdicts recorded
    under older rules manufactures dead filters that are not dead -- the
    cry-wolf failure that gets a guard switched off.

    The policy fingerprint already marks which rules produced each verdict, so
    scoping to it is the fix that was already lying there.
    """
    with store.open_store(tmp_path / "t.db") as conn:
        store.save_no_text(conn, "/w/a.m4a", duration_s=1.0, policy="old",
                           reason="no_speech_detected")
        store.save_no_text(conn, "/w/b.m4a", duration_s=1.0, policy="new",
                           reason="silence")

        assert store.filter_activity(conn) == {
            "no_speech_detected": 1, "silence": 1
        }
        assert store.filter_activity(conn, policy="new") == {"silence": 1}
        assert store.filter_activity(conn, policy="old") == {"no_speech_detected": 1}


def test_the_latest_policy_can_be_identified(tmp_path: Path) -> None:
    """So dormancy defaults to ONE rule set instead of mixing vocabularies.

    Run across all policies on a real sidecar, three filters looked dormant
    purely because an earlier pipeline spelled its reasons differently. The
    newest policy is the one whose filter list the current code actually has.
    """
    with store.open_store(tmp_path / "t.db") as conn:
        store.save_no_text(conn, "/w/a.m4a", duration_s=1.0, policy="old",
                           reason="no_speech_detected")
        store.save_no_text(conn, "/w/b.m4a", duration_s=1.0, policy="new",
                           reason="silence")
        assert store.latest_policy(conn) == "new"


def test_latest_policy_is_none_on_an_empty_sidecar(tmp_path: Path) -> None:
    with store.open_store(tmp_path / "t.db") as conn:
        assert store.latest_policy(conn) is None


def test_activity_can_be_scoped_to_one_population(tmp_path: Path) -> None:
    """Per-window and whole-file filters are different populations.

    A reason that only a per-window check can produce will always look dormant
    if it is judged against whole-file verdicts, and vice versa. Reporting
    them together guarantees false positives -- which is exactly what a real
    sidecar produced: filters flagged dormant that simply cannot appear in the
    table being counted.
    """
    with store.open_store(tmp_path / "t.db") as conn:
        store.save_no_text(conn, "/w/a.m4a", duration_s=1.0, policy="p",
                           reason="looping_repetition")
        store.save_dropped_windows(
            conn, "/w/b.m4a",
            [{"window_start": 0.0, "text": "x", "reason": "caption_boilerplate"}],
            policy="p",
        )

        assert store.filter_activity(conn, table="no_text") == {
            "looping_repetition": 1
        }
        assert store.filter_activity(conn, table="dropped_windows") == {
            "caption_boilerplate": 1
        }


def test_latest_policy_prefers_transcripts_which_are_restamped(tmp_path: Path) -> None:
    """After a re-judge the sidecar legitimately holds MIXED policies.

    `rejudge_stored` restamps transcripts to the current rules, but leaves
    no_text rows under the old ones on purpose: a loosened threshold means
    those files should be RETRIED, not silently marked current. So reading the
    policy from no_text reports the superseded rule set, and dormancy then
    judges the current filter list against an old vocabulary -- the exact
    false positive the policy scoping was added to prevent.

    Transcripts are restamped by every run, so they carry the current rules.
    """
    with store.open_store(tmp_path / "t.db") as conn:
        store.save_no_text(conn, "/w/a.m4a", duration_s=1.0, policy="old",
                           reason="silence")
        t = Transcript(path="/w/b.m4a", text="real speech here",
                       windows=[Window(0.0, 30.0, "real speech here", "en")],
                       duration_s=30.0, model="m")
        t.policy = "new"
        store.save_transcript(conn, t)

        assert store.latest_policy(conn) == "new"


def test_latest_policy_falls_back_to_no_text_when_nothing_was_kept(
    tmp_path: Path,
) -> None:
    # An archive that produced only rejections still has a newest rule set.
    with store.open_store(tmp_path / "t.db") as conn:
        store.save_no_text(conn, "/w/a.m4a", duration_s=1.0, policy="only",
                           reason="silence")
        assert store.latest_policy(conn) == "only"


# --- "too small to judge" must not describe a large archive -------------------


def _sidecar_with(tmp_path, rows):
    """rows: (table, reason, policy) tuples."""
    import sqlite3

    db = tmp_path / "t.db"
    conn = sqlite3.connect(db)
    for table in ("no_text", "dropped_windows"):
        conn.execute(
            f"CREATE TABLE {table} (path TEXT, reason TEXT, policy TEXT)"
        )
    for table, reason, policy in rows:
        conn.execute(
            f"INSERT INTO {table} (path, reason, policy) VALUES (?, ?, ?)",
            ("p", reason, policy),
        )
    conn.commit()
    return conn


def test_verdicts_under_an_older_policy_are_counted_separately(tmp_path):
    """Live shape: a sidecar holding 3,381 whole-file verdicts reported "2
    whole-file verdicts -- too small to judge dormancy". Both numbers were
    right and the sentence was misleading: the archive is large, and the
    check had scoped itself out of almost all of it after a threshold
    changed. "Small archive" and "instrument scoped out of a big one" need
    different answers, and they read identically."""
    from corpus.transcripts.store import activity_coverage

    conn = _sidecar_with(
        tmp_path,
        [("no_text", "no_speech_detected", "old")] * 300
        + [("no_text", "looping_repetition", "current")] * 2,
    )
    cov = activity_coverage(conn, policy="current", table="no_text")
    assert cov.in_scope == 2
    assert cov.other_policy == 300
    assert "policy" in cov.describe().lower()
    assert "300" in cov.describe()


def test_rows_recorded_before_reasons_were_stored_are_named_not_ignored(tmp_path):
    """`dropped_windows.reason` was added by a migration, so every row written
    before it is NULL. A live sidecar has 2,548 such rows and the check
    reported "examined nothing (per-window verdicts)" -- true of the reasons,
    false of the drops, and the difference is whether the filter is dormant or
    merely unreadable."""
    from corpus.transcripts.store import activity_coverage

    conn = _sidecar_with(
        tmp_path, [("dropped_windows", None, "current")] * 2548
    )
    cov = activity_coverage(conn, policy="current", table="dropped_windows")
    assert cov.in_scope == 0
    assert cov.unattributed == 2548
    assert "2,548" in cov.describe()
    assert "dormant" not in cov.describe().lower(), "an unreadable rule is not a dead one"


def test_a_genuinely_empty_sidecar_says_so(tmp_path):
    from corpus.transcripts.store import activity_coverage

    conn = _sidecar_with(tmp_path, [])
    cov = activity_coverage(conn, policy="current", table="no_text")
    assert cov.in_scope == 0 and cov.other_policy == 0 and cov.unattributed == 0
    assert "no verdicts" in cov.describe().lower()


def test_a_well_covered_sidecar_reports_no_caveat(tmp_path):
    from corpus.transcripts.store import activity_coverage

    conn = _sidecar_with(
        tmp_path, [("no_text", "no_speech_detected", "current")] * 250
    )
    cov = activity_coverage(conn, policy="current", table="no_text")
    assert cov.in_scope == 250
    assert cov.describe() == ""
