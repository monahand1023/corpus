"""corpus-ingest's exit codes are its whole interface to an unattended run.

Cron and CI see an exit code, not stderr, so the difference between 0, 1
and EXIT_ANOMALY (3) is the entire signal that a nightly ingest quietly
stopped yielding documents or refused a destructive prune. None of it was
tested: the command is 206 lines of exit-code and refusal logic and had no
test file at all.

`Ingester` is faked here on purpose. What is under test is the command's
contract -- which conditions refuse, which continue, which escalate -- not
whether ingestion works, which `tests/test_ingester.py` covers against a
real store.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from corpus.cli.ingest import EXIT_ANOMALY
from corpus.config import CorpusConfig, EmbedderConfig, SourceConfig
from corpus.ingester import IngestResult

CLEAN = IngestResult(
    source_name="notes",
    documents=10,
    chunks_seen=40,
    chunks_upserted=40,
    chunks_skipped=0,
    orphans_deleted=0,
    tokens_used=100,
    elapsed_seconds=1.0,
)


class FakeIngester:
    """Returns a scripted result (or raises) per source name."""

    def __init__(self, results):
        self._results = results
        self.calls: list[tuple[str, bool]] = []
        self.closed = False

    def ingest(self, name, prune_anyway=False):
        self.calls.append((name, prune_anyway))
        outcome = self._results[name]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    def close(self):
        self.closed = True


def _config(*names: str) -> CorpusConfig:
    return CorpusConfig(
        db_path="c.db",
        embedder=EmbedderConfig(provider="hash", dim=64),
        sources=[SourceConfig(name=n, type="markdown", path=".") for n in names],
    )


def _run(monkeypatch, argv, *, results=None, cfg=None):
    import corpus.cli.ingest as mod

    fake = FakeIngester(results or {})
    monkeypatch.setattr(mod, "load_config_or_exit", lambda _p: cfg or _config("notes"))
    monkeypatch.setattr(mod, "Ingester", lambda _c: fake)
    monkeypatch.setattr("sys.argv", argv)
    return mod.main(), fake


# --- refusals: nothing should be ingested at all ------------------------------


def test_prune_anyway_with_all_is_refused_before_anything_runs(monkeypatch):
    """--prune-anyway forces a destructive sweep past the blast-radius guard.
    Across every source at once, nobody has reviewed what it is about to
    delete."""
    code, fake = _run(monkeypatch, ["corpus-ingest", "--all", "--prune-anyway"])
    assert code == 2
    assert fake.calls == [], "a refused invocation still ingested"


def test_path_cannot_be_combined_with_source(monkeypatch):
    code, fake = _run(monkeypatch, ["corpus-ingest", "--path", ".", "--source", "notes"])
    assert code == 2
    assert fake.calls == []


# --- exit codes ---------------------------------------------------------------


def test_a_clean_run_exits_zero(monkeypatch):
    code, fake = _run(
        monkeypatch, ["corpus-ingest", "--source", "notes"], results={"notes": CLEAN}
    )
    assert code == 0
    assert fake.closed, "the ingester was left open"


@pytest.mark.parametrize(
    "field",
    ["yield_drop_detail", "path_change_detail", "skip_rise_detail", "vanished_detail"],
)
def test_every_anomaly_escalates_the_exit_code(monkeypatch, field, capsys):
    """A run that reports an anomaly and exits 0 is indistinguishable from a
    healthy one to anything that only sees the exit code -- which is every
    unattended caller."""
    result = replace(CLEAN, **{field: "documents went from 900 to 12"})
    code, _ = _run(
        monkeypatch, ["corpus-ingest", "--source", "notes"], results={"notes": result}
    )
    assert code == EXIT_ANOMALY
    assert "WARNING" in capsys.readouterr().out


def test_prune_anyway_is_the_acknowledgement_that_clears_an_anomaly(monkeypatch):
    """Re-running with --prune-anyway IS the operator saying they looked. If
    that still exited 3 there would be no way to clear the signal, which
    trains people to ignore it."""
    result = replace(CLEAN, yield_drop_detail="documents went from 900 to 12")
    code, fake = _run(
        monkeypatch,
        ["corpus-ingest", "--source", "notes", "--prune-anyway"],
        results={"notes": result},
    )
    assert code == 0
    assert fake.calls == [("notes", True)], "--prune-anyway was not passed through"


def test_a_refused_prune_is_a_failure_not_a_warning(monkeypatch, capsys):
    result = replace(
        CLEAN,
        prune_refused=True,
        prune_refused_detail="would delete 9,000 of 10,000 chunks",
        pruning_performed=False,
    )
    code, _ = _run(
        monkeypatch, ["corpus-ingest", "--source", "notes"], results={"notes": result}
    )
    assert code == 1
    out = capsys.readouterr().out
    assert "REFUSED" in out
    assert "9,000 of 10,000" in out, "the operator was told a refusal but not why"


def test_one_unreadable_source_does_not_stop_the_others(monkeypatch, capsys):
    """--all across an archive where one volume is unmounted. That source's
    index is left intact -- pruning never ran for it -- and the rest proceed."""
    code, fake = _run(
        monkeypatch,
        ["corpus-ingest", "--all"],
        cfg=_config("gone", "notes"),
        results={
            "gone": FileNotFoundError("/Volumes/Archive: no such directory"),
            "notes": CLEAN,
        },
    )
    assert code == 1
    assert [c[0] for c in fake.calls] == ["gone", "notes"]
    assert "index left intact" in capsys.readouterr().out


def test_a_hard_failure_outranks_an_anomaly(monkeypatch):
    """"this did not work" must not be reported as "this worked, take a
    look" just because another source happened to be anomalous first."""
    code, _ = _run(
        monkeypatch,
        ["corpus-ingest", "--all"],
        cfg=_config("odd", "gone"),
        results={
            "odd": replace(CLEAN, yield_drop_detail="900 -> 12"),
            "gone": OSError("unmounted"),
        },
    )
    assert code == 1


def test_the_ingester_is_closed_even_when_a_source_explodes(monkeypatch):
    """A leaked store holds SQLite's writer lock for the life of the process."""
    import corpus.cli.ingest as mod

    fake = FakeIngester({"notes": RuntimeError("boom")})
    monkeypatch.setattr(mod, "load_config_or_exit", lambda _p: _config("notes"))
    monkeypatch.setattr(mod, "Ingester", lambda _c: fake)
    monkeypatch.setattr("sys.argv", ["corpus-ingest", "--source", "notes"])

    with pytest.raises(RuntimeError):
        mod.main()
    assert fake.closed


# --- --path -------------------------------------------------------------------


def test_path_with_nothing_ingestable_says_what_is_supported(monkeypatch, capsys):
    import corpus.cli.ingest as mod

    monkeypatch.setattr(mod, "detect_sources", lambda _p: [])
    code, fake = _run(monkeypatch, ["corpus-ingest", "--path", "/tmp/empty"])
    assert code == 1
    assert fake.calls == []
    assert "Supported types" in capsys.readouterr().out


def test_path_ingests_what_it_detected_not_what_is_configured(monkeypatch, capsys):
    import corpus.cli.ingest as mod

    detected = [SourceConfig(name="empty_pdf", type="pdf", path="/tmp/x")]
    monkeypatch.setattr(mod, "detect_sources", lambda _p: detected)
    code, fake = _run(
        monkeypatch,
        ["corpus-ingest", "--path", "/tmp/x"],
        cfg=_config("notes", "papers"),
        results={"empty_pdf": replace(CLEAN, source_name="empty_pdf")},
    )
    assert code == 0
    assert [c[0] for c in fake.calls] == ["empty_pdf"]
    assert "Detected 1 source(s)" in capsys.readouterr().out
