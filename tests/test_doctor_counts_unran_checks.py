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
