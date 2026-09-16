"""The sidecar database a transcription run writes, and an index reads.

Transcribing is slow, expensive and separate from indexing: hours of GPU on
one side, an embedding API on the other, and no reason for either to wait on
the other. A sidecar database is the seam. A run fills it; `corpus-ingest`
reads it whenever it likes; re-running either is cheap because both know what
is already done.

The schema is shaped by what a long run actually needs, and every table here
earned its place on a real real archive:

* `transcripts` — what was said. `policy` records the rules in force when the
  text was ACCEPTED. Rejections carried that from the start and acceptances
  did not, and the asymmetry showed the moment a rule tightened: finding the
  transcripts the new rules would never have produced meant a one-off script.
  With the column it is a query.

* `no_text` — files that transcribed cleanly and produced nothing usable.
  These are NOT failures; the run did exactly what it should and the audio has
  no speech in it. Recording them is what makes a restart cheap: without the
  row the file is simply absent from `transcripts`, so the next run re-decodes
  and re-transcribes it in full to reach the same answer. One interrupted pass
  re-paid that for 889 already-examined silent clips before this table existed.

* `dropped_windows` — every window the filter discarded, with the evidence it
  judged on. Without these rows the filter is unauditable: it makes thousands
  of destructive decisions and keeping only a COUNT of them leaves the
  question that matters unanswerable — how much of what it threw away was real
  speech. Recall can be measured from surviving text; precision cannot be
  measured at all. Keyed on path rather than on a transcript row, because a
  file whose windows were ALL discarded produces no transcript, and those are
  both the most interesting cases to audit and the ones that would vanish.

* `failures` — the run broke. Distinct from `no_text` on purpose, because
  conflating "this file has no speech" with "this file crashed the decoder"
  makes both unactionable.

POLICY FINGERPRINTS. `no_text` and `dropped_windows` both carry one, and it
must cover EVERY setting that can turn audio into "nothing usable" — including
the ones that reject without the model running at all. A stored verdict
outliving the rule that produced it is the failure this exists to prevent.
`policy_fingerprint` builds it; the caller passes everything, and passing too
much is harmless where passing too little is silent.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS transcripts (
  path TEXT PRIMARY KEY,
  duration_s REAL,
  dropped_windows INTEGER NOT NULL DEFAULT 0,
  text TEXT NOT NULL,
  languages TEXT NOT NULL,      -- JSON: per-window detected language
  segments TEXT NOT NULL,       -- JSON: [{start, end, lang, text}]
  model TEXT NOT NULL,
  policy TEXT NOT NULL DEFAULT '',
  transcribed_at TEXT NOT NULL,
  elapsed_s REAL
);

CREATE TABLE IF NOT EXISTS failures (
  path TEXT PRIMARY KEY,
  error TEXT NOT NULL,
  failed_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS no_text (
  path TEXT PRIMARY KEY,
  duration_s REAL NOT NULL,
  policy TEXT NOT NULL,
  reason TEXT NOT NULL DEFAULT '',
  rejected_text TEXT NOT NULL DEFAULT '',
  checked_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS dropped_windows (
  path TEXT NOT NULL,
  window_start REAL NOT NULL,
  no_speech REAL NOT NULL,
  avg_logprob REAL NOT NULL,
  text TEXT NOT NULL,
  reason TEXT NOT NULL DEFAULT '',
  policy TEXT NOT NULL,
  PRIMARY KEY (path, window_start)
);
"""

# Columns added after the first databases were created. SQLite's
# CREATE TABLE IF NOT EXISTS leaves an existing table alone, so a database made
# by an earlier version keeps the old shape unless it is migrated.
_ADDED_COLUMNS: dict[str, tuple[tuple[str, str], ...]] = {
    "no_text": (
        ("reason", "TEXT NOT NULL DEFAULT ''"),
        ("rejected_text", "TEXT NOT NULL DEFAULT ''"),
    ),
    "transcripts": (("policy", "TEXT NOT NULL DEFAULT ''"),),
    "dropped_windows": (("reason", "TEXT NOT NULL DEFAULT ''"),),
}


@dataclass(frozen=True)
class Window:
    """One transcribed window: what was said, when, and in what language."""

    start: float
    end: float
    text: str
    lang: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {"start": self.start, "end": self.end, "lang": self.lang,
                "text": self.text}


@dataclass
class Transcript:
    path: str
    text: str
    windows: list[Window] = field(default_factory=list)
    duration_s: float | None = None
    dropped_windows: int = 0
    model: str = ""
    policy: str = ""
    elapsed_s: float | None = None

    @property
    def languages(self) -> list[str | None]:
        return [w.lang for w in self.windows]


def policy_fingerprint(settings: Mapping[str, Any]) -> str:
    """A stable hash of every setting that can reject audio.

    Pass EVERYTHING that participates in a rejection, including settings that
    reject before the model runs. That completeness was once false in a real
    pipeline: the voice-activity thresholds were omitted even though they
    produce the single most destructive verdict available -- "no speech",
    reached without transcribing at all -- so tightening them left stale
    verdicts in place and the files were never retried.

    Passing a setting that turns out not to matter only causes a needless
    retry. Omitting one that does causes a wrong answer that never expires.
    """
    payload = json.dumps(settings, sort_keys=True, default=_stable)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _stable(value: Any) -> Any:
    """Make sets and paths hashable in a stable order."""
    if isinstance(value, (set, frozenset)):
        return sorted(str(v) for v in value)
    if isinstance(value, Path):
        return str(value)
    return str(value)


def connect(path: Path | str, *, read_only: bool = False) -> sqlite3.Connection:
    """Open the sidecar, creating and migrating it unless read-only."""
    if read_only:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        return conn
    Path(path).expanduser().parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    migrate(conn)
    return conn


def migrate(conn: sqlite3.Connection) -> list[str]:
    """Add any missing columns. Returns what it changed, for logging."""
    applied: list[str] = []
    for table, columns in _ADDED_COLUMNS.items():
        have = {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}
        if not have:
            continue  # table absent: SCHEMA will create it whole
        for name, ddl in columns:
            if name not in have:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {ddl}")
                applied.append(f"{table}.{name}")
    conn.commit()
    return applied


@contextmanager
def open_store(
    path: Path | str, *, read_only: bool = False
) -> Iterator[sqlite3.Connection]:
    conn = connect(path, read_only=read_only)
    try:
        yield conn
    finally:
        conn.close()


def _now() -> str:
    return datetime.now(UTC).isoformat()


def save_transcript(conn: sqlite3.Connection, transcript: Transcript) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO transcripts (path, duration_s, dropped_windows,"
        " text, languages, segments, model, policy, transcribed_at, elapsed_s)"
        " VALUES (?,?,?,?,?,?,?,?,?,?)",
        (
            transcript.path,
            transcript.duration_s,
            transcript.dropped_windows,
            transcript.text,
            json.dumps(transcript.languages),
            json.dumps([w.as_dict() for w in transcript.windows],
                       ensure_ascii=False),
            transcript.model,
            transcript.policy,
            _now(),
            transcript.elapsed_s,
        ),
    )
    conn.commit()


def clear_failure(conn: sqlite3.Connection, path: str) -> None:
    """Drop a `failures` row now that this file has an answer.

    `failures` is deliberately excluded from `already_done` so a failed file
    is retried -- and nothing removed the row when the retry worked. A live
    sidecar held 94 failure rows for files that had since been given a proper
    verdict: the table read as 94 broken files and there were none. A record
    that says something which stopped being true is worse than no record,
    because someone acts on it.
    """
    conn.execute("DELETE FROM failures WHERE path = ?", (path,))


def save_no_text(
    conn: sqlite3.Connection,
    path: str,
    *,
    duration_s: float,
    policy: str,
    reason: str = "",
    rejected_text: str = "",
) -> None:
    """Record that a file produced nothing usable, under THESE rules.

    The policy is what makes the row safe to trust later: when the rules
    change the fingerprint changes, the row stops matching, and the file is
    retried rather than inheriting a verdict made under different rules.
    """
    conn.execute(
        "INSERT OR REPLACE INTO no_text"
        " (path, duration_s, policy, reason, rejected_text, checked_at)"
        " VALUES (?,?,?,?,?,?)",
        (path, duration_s, policy, reason, rejected_text, _now()),
    )
    conn.commit()


def save_failure(conn: sqlite3.Connection, path: str, error: str) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO failures (path, error, failed_at) VALUES (?,?,?)",
        (path, error, _now()),
    )
    conn.commit()


def save_dropped_windows(
    conn: sqlite3.Connection,
    path: str,
    dropped: Sequence[Mapping[str, Any]],
    *,
    policy: str,
) -> None:
    """Keep the evidence behind each discarded window.

    This is what makes the filter auditable at all. Without it the only record
    of thousands of destructive decisions is a count.
    """
    conn.executemany(
        "INSERT OR REPLACE INTO dropped_windows"
        " (path, window_start, no_speech, avg_logprob, text, reason, policy)"
        " VALUES (?,?,?,?,?,?,?)",
        [
            (
                path,
                float(d.get("window_start", 0.0)),
                float(d.get("no_speech", 0.0)),
                float(d.get("avg_logprob", 0.0)),
                str(d.get("text", "")),
                # WHICH rule fired. Without it the evidence this table exists
                # to keep cannot answer the question that matters: is any of
                # these filters dead? A count of discards cannot say.
                str(d.get("reason", "")),
                policy,
            )
            for d in dropped
        ],
    )
    conn.commit()


def already_done(conn: sqlite3.Connection, *, policy: str) -> set[str]:
    """Paths this run can skip: transcribed, or judged empty under THIS policy.

    Failures are deliberately NOT included -- a failure is worth retrying,
    since it usually means a broken decode or a transient resource problem
    rather than a settled verdict about the audio.
    """
    done = {r[0] for r in conn.execute("SELECT path FROM transcripts")}
    done |= {
        r[0]
        for r in conn.execute("SELECT path FROM no_text WHERE policy = ?", (policy,))
    }
    return done


def stale_transcripts(
    conn: sqlite3.Connection, *, policy: str
) -> list[tuple[str, str, float, list[str]]]:
    """Transcripts KEPT under rules other than these.

    The other half of what `already_done` is for. A rule change invalidates a
    stored "no speech" verdict, but a stored transcript was equally a verdict
    -- "this text is real" -- and it was being inherited forever. The six
    looping transcripts that motivated `looping_share` would have survived the
    fix that was written to catch them.
    """
    rows = conn.execute(
        "SELECT path, text, duration_s, languages FROM transcripts"
        " WHERE policy IS NOT ? ",
        (policy,),
    ).fetchall()
    out = []
    for path, text, duration_s, languages in rows:
        try:
            langs = [lg for lg in json.loads(languages or "[]") if lg]
        except (ValueError, TypeError):
            langs = []
        out.append((path, text, duration_s or 0.0, langs))
    return out


def restamp_transcript(conn: sqlite3.Connection, path: str, *, policy: str) -> None:
    """Record that this transcript still passes, under these rules."""
    conn.execute("UPDATE transcripts SET policy = ? WHERE path = ?", (policy, path))
    conn.commit()


def demote_transcript(
    conn: sqlite3.Connection,
    path: str,
    *,
    duration_s: float,
    policy: str,
    reason: str,
    rejected_text: str,
) -> None:
    """Move a transcript the current rules reject into `no_text`.

    The text is kept as `rejected_text` rather than dropped, so a rule that
    turns out to be too aggressive can be audited and reversed against real
    evidence instead of a re-run.
    """
    conn.execute("DELETE FROM transcripts WHERE path = ?", (path,))
    conn.execute(
        "INSERT OR REPLACE INTO no_text"
        " (path, duration_s, policy, reason, rejected_text, checked_at)"
        " VALUES (?,?,?,?,?,?)",
        (path, duration_s, policy, reason, rejected_text, _now()),
    )
    conn.commit()


def latest_policy(conn: sqlite3.Connection) -> str | None:
    """The most recently recorded rule set, or None when nothing is recorded.

    Dormancy has to be judged within ONE rule set. Counting verdicts written
    under older rules against the current filter list manufactures dead
    filters that are not dead -- on a real sidecar, three looked dormant only
    because an earlier pipeline spelled its reasons differently. The newest
    policy is the one whose filter names the running code actually knows.
    """
    # TRANSCRIPTS FIRST, and the order matters. After a re-judge the sidecar
    # legitimately holds mixed policies: `rejudge_stored` restamps transcripts
    # to the current rules but deliberately leaves no_text rows under the old
    # ones, because a loosened threshold means those files should be RETRIED
    # rather than silently marked current. Reading no_text therefore reports
    # the superseded rule set, and anything judging the current filter list
    # against it manufactures the false positives policy scoping exists to
    # prevent. Transcripts are restamped by every run, so they carry the
    # newest rules; no_text is the fallback for an archive that kept nothing.
    for table, stamp in (("transcripts", "transcribed_at"), ("no_text", "checked_at")):
        try:
            row = conn.execute(
                f"SELECT policy FROM {table}"
                " WHERE policy IS NOT NULL AND policy != ''"
                f" ORDER BY {stamp} DESC LIMIT 1"
            ).fetchone()
        except sqlite3.Error:
            continue
        if row:
            return str(row[0])
    return None


def filter_activity(
    conn: sqlite3.Connection,
    *,
    policy: str | None = None,
    table: str | None = None,
) -> dict[str, int]:
    """How many times each rejection reason actually fired.

    The input to dormancy detection: a filter that never appears here, across
    a corpus large enough to mean it, is either unnecessary or broken. This
    project shipped a repetition check that had never once fired on real data
    and nothing said so, because a count of total discards cannot distinguish
    a dead rule from a rule with nothing to reject.
    """
    out: Counter[str] = Counter()
    where = " WHERE reason IS NOT NULL AND reason != ''"
    params: tuple[str, ...] = ()
    if policy is not None:
        # Scope to ONE rule set. Comparing a current filter list against
        # verdicts recorded under older rules manufactures dead filters that
        # are not dead: measured on a real sidecar, three looked dormant only
        # because an earlier pipeline spelled the reasons differently.
        where += " AND policy = ?"
        params = (policy,)
    # Per-window and whole-file rejections are different POPULATIONS. A
    # reason only a per-window check can produce looks dormant forever if it
    # is judged against whole-file verdicts, so a caller comparing a filter
    # list must scope to the table that list belongs to.
    tables = (table,) if table else ("no_text", "dropped_windows")
    for name in tables:
        if name not in ("no_text", "dropped_windows"):
            raise ValueError(f"unknown table: {name}")
        try:
            rows = conn.execute(
                f"SELECT reason, count(*) FROM {name}{where} GROUP BY reason",
                params,
            )
        except sqlite3.Error:
            continue
        for reason, n in rows:
            out[reason] += n
    return dict(out)


@dataclass(frozen=True)
class ActivityCoverage:
    """Why a dormancy check has the verdicts it has -- or has none.

    "This archive is small" and "this instrument scoped itself out of a large
    archive" produce the same empty result and need opposite responses. A live
    sidecar holding 3,381 whole-file verdicts reported "2 whole-file verdicts
    -- too small to judge dormancy": both numbers correct, the sentence
    misleading, because a threshold change had moved the policy fingerprint
    and left 3,379 verdicts out of scope.

    The third case is its own: `dropped_windows.reason` arrived in a
    migration, so every row written before it is NULL. 2,548 such rows read as
    "examined nothing" -- true of the reasons, false of the drops. An
    unreadable rule is not a dead one.
    """

    in_scope: int = 0
    other_policy: int = 0
    unattributed: int = 0

    @property
    def total(self) -> int:
        return self.in_scope + self.other_policy + self.unattributed

    def describe(self) -> str:
        """The caveat to print, or "" when there is nothing to explain."""
        if not self.total:
            return "no verdicts recorded at all"
        parts = []
        if self.other_policy:
            parts.append(
                f"{self.other_policy:,} more were recorded under a different "
                "policy (a threshold changed since); re-run to bring them "
                "into scope"
            )
        if self.unattributed:
            parts.append(
                f"{self.unattributed:,} carry no reason -- they predate the "
                "column, so the rule behind them cannot be read"
            )
        return "; ".join(parts)


def activity_coverage(
    conn: sqlite3.Connection, *, policy: str | None = None, table: str = "no_text"
) -> ActivityCoverage:
    """Split a table's verdicts into in-scope, other-policy and unattributed."""
    if table not in ("no_text", "dropped_windows"):
        raise ValueError(f"unknown table: {table}")
    has_reason = "reason IS NOT NULL AND reason != ''"
    try:
        unattributed = conn.execute(
            f"SELECT count(*) FROM {table} WHERE NOT ({has_reason})"
        ).fetchone()[0]
        if policy is None:
            in_scope = conn.execute(
                f"SELECT count(*) FROM {table} WHERE {has_reason}"
            ).fetchone()[0]
            other = 0
        else:
            in_scope = conn.execute(
                f"SELECT count(*) FROM {table} WHERE {has_reason} AND policy = ?",
                (policy,),
            ).fetchone()[0]
            other = conn.execute(
                f"SELECT count(*) FROM {table} WHERE {has_reason} AND "
                "(policy IS NULL OR policy != ?)",
                (policy,),
            ).fetchone()[0]
    except sqlite3.Error:
        return ActivityCoverage()
    return ActivityCoverage(
        in_scope=in_scope, other_policy=other, unattributed=unattributed
    )


def counts(conn: sqlite3.Connection) -> dict[str, int]:
    """Row counts per table, for progress reporting and sanity checks."""
    out: dict[str, int] = {}
    for table in ("transcripts", "no_text", "failures", "dropped_windows"):
        try:
            out[table] = conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
        except sqlite3.Error:
            out[table] = 0
    return out
