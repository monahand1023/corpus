"""An errored check was counted in the numerator of "N/N checks passed".

Eight checks in `corpus-doctor` swallow an exception, print
`SKIPPED (could not read: OperationalError)` and `return True`. The summary
builds `failed` from the return values, so a check that blew up internally
lands among the passes:

    OK: 12/12 checks passed

while three of them never examined anything. The banner underneath -- "A
skipped check is not a passing one" -- is true of the skips listed beside
it (those come from the `needs` gate, which is about missing arguments) and
false of exactly these.

This is the defect the command exists to eliminate, in the command itself.

A THIRD STATE, not a bool. "Passed", "failed" and "could not run" are three
facts, and collapsing the third into either of the others is the collapse
this whole file is about. `None` is that state, so a check that forgets to
return anything on an error path is counted correctly by accident rather
than wrongly by accident.
"""

from __future__ import annotations


def test_a_check_that_could_not_run_is_not_counted_as_passed():
    from corpus.cli.doctor import summarise_checks

    summary = summarise_checks(
        {"a": True, "b": True, "c": None}, skipped=[]
    )
    assert summary.passed == 2
    assert summary.unran == ["c"]
    assert "2/3" in summary.describe()
    assert "could not run" in summary.describe().lower()


def test_a_failure_still_reads_as_a_failure():
    from corpus.cli.doctor import summarise_checks

    summary = summarise_checks({"a": True, "b": False}, skipped=[])
    assert summary.failed == ["b"]
    assert summary.ok is False
    assert "PROBLEMS FOUND" in summary.describe()


def test_a_check_that_could_not_run_makes_the_whole_run_not_ok():
    """The exit code has to move. A diagnostic that exits 0 while blind is
    the same silent pass one layer up, and CI reads the exit code."""
    from corpus.cli.doctor import summarise_checks

    summary = summarise_checks({"a": True, "b": None}, skipped=[])
    assert summary.ok is False, "a blind run reported success"


def test_an_all_clear_is_still_an_all_clear():
    from corpus.cli.doctor import summarise_checks

    summary = summarise_checks({"a": True, "b": True}, skipped=[])
    assert summary.ok is True
    assert summary.unran == []
    assert "OK: 2/2" in summary.describe()


def test_argument_skips_stay_separate_from_errored_checks():
    """`needs`-gated skips mean "you did not give me a path"; an unran check
    means "I had everything and still could not look". Different remedies,
    so they must not merge into one number."""
    from corpus.cli.doctor import summarise_checks

    summary = summarise_checks({"a": True, "b": None}, skipped=["c", "d"])
    text = summary.describe()
    assert "2 SKIPPED" in text
    assert summary.unran == ["b"]
    assert "b" not in text.split("SKIPPED")[1]


# --- a check that compared nothing --------------------------------------------


def test_served_vs_evaluated_actually_compares_the_two_numbers(tmp_path, capsys):
    """It printed a hardcoded `[ ok ]` and returned True on every path.

        print(f"  [  ok  ] corpus-eval defaults --top-k to [retriever] top_k "
              f"= {config.retriever.top_k}, ...")

    That is a RESTATEMENT of a config value, not a comparison. There is no
    False anywhere in the function, and it counted toward "N/N checks
    passed" -- so the answer to "does the eval measure the k the server
    serves?" was yes, unconditionally, including when it was no.

    There IS something to verify: `corpus-eval --top-k` defaults to None and
    falls back to `config.retriever.top_k`. A refactor that hardcodes a
    literal there breaks the guarantee silently, and this is what notices.
    """
    from corpus.cli.doctor import _check_served_vs_evaluated

    cfg = tmp_path / "corpus.toml"
    cfg.write_text(
        'db_path = "./x.db"\n[retriever]\ntop_k = 9\n[embedder]\nprovider = "hash"\ndim = 8\n'
    )
    ok = _check_served_vs_evaluated(str(cfg))
    out = capsys.readouterr().out

    assert "9" in out, "the served k was not reported"
    assert ok is True, out
    assert "resolves" in out.lower() or "matches" in out.lower(), (
        f"still restating the config instead of comparing:\n{out}"
    )


def test_it_fails_when_the_eval_stops_deriving_k_from_the_config(
    tmp_path, capsys, monkeypatch
):
    """The regression it exists to catch: someone gives --top-k a literal
    default, and the eval quietly stops measuring what the server serves."""
    import corpus.cli.eval as eval_cli
    from corpus.cli.doctor import _check_served_vs_evaluated

    cfg = tmp_path / "corpus.toml"
    cfg.write_text(
        'db_path = "./x.db"\n[retriever]\ntop_k = 9\n[embedder]\nprovider = "hash"\ndim = 8\n'
    )

    def _hardcoded_parser():
        import argparse

        p = argparse.ArgumentParser()
        p.add_argument("--top-k", type=int, default=5)
        return p

    monkeypatch.setattr(eval_cli, "build_parser", _hardcoded_parser, raising=False)
    ok = _check_served_vs_evaluated(str(cfg))
    out = capsys.readouterr().out

    assert ok is False, f"a hardcoded eval default was reported as fine:\n{out}"
    assert "5" in out and "9" in out, out


# --- a margin that reads backwards -------------------------------------------


def test_a_margin_says_when_the_nearest_row_only_survived_the_fallback(
    tmp_path, capsys
):
    """"0.0% headroom" reads as "this threshold is too tight". It can mean
    the exact opposite, and on a live archive it did.

    The two rows nearest the looping ceiling were both decode loops -- a
    repeated Japanese sentence and one word twelve times. Neither is material
    at risk; both are junk the ceiling FAILED to catch. Acting on the obvious
    reading, raising the ceiling, would have kept more of them.

    The check already warns about this in prose and prints the offending
    text, which asks the reader to notice. The distinguishing fact is
    computable instead: a single-window transcript whose window scores ABOVE
    the strict per-window ceiling was dropped and then REINSTATED by the
    fallback, and is only in the sample at all because of it. Real material
    near the ceiling does not have that property.
    """
    import sqlite3

    from corpus.cli.doctor import _check_threshold_margins

    db = tmp_path / "t.db"
    conn = sqlite3.connect(db)
    conn.execute(
        "CREATE TABLE transcripts (path TEXT, text TEXT, duration_s REAL,"
        " segments TEXT, policy TEXT DEFAULT '', model TEXT DEFAULT '')"
    )
    # One window, looping 0.83 -- above the 0.6 window ceiling, below the
    # 0.85 transcript one. Exactly the shape that only survives reinstatement.
    loop = "私たちの家に来てくれた。" * 12
    conn.execute(
        "INSERT INTO transcripts (path, text, duration_s, segments) VALUES (?,?,?,?)",
        ("/w/loop.mov", loop, 30.0,
         '[{"start": 0.0, "end": 30.0, "text": ' + repr(loop).replace("'", '"') + "}]"),
    )
    conn.commit()
    conn.close()

    _check_threshold_margins(str(db))
    out = capsys.readouterr().out

    assert "looping share" in out
    assert "fallback" in out.lower(), (
        "the margin did not say the nearest row only survived reinstatement, "
        f"so it still reads as a threshold that is too tight:\n{out}"
    )
