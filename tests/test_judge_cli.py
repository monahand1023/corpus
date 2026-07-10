from __future__ import annotations

from types import SimpleNamespace

from corpus.cli import judge as judge_cli
from corpus.cli._common import load_python_export


def test_load_python_export_with_future_annotations(tmp_path):
    mod = tmp_path / "cases_mod.py"
    mod.write_text(
        "from __future__ import annotations\n"
        "from dataclasses import dataclass\n"
        "@dataclass(frozen=True)\n"
        "class C:\n"
        "    x: int\n"
        "ITEMS = [C(1), C(2)]\n"
    )
    items = load_python_export(mod, "ITEMS")
    assert [c.x for c in items] == [1, 2]


def _stub_validate(monkeypatch, report):
    monkeypatch.setattr(judge_cli, "run_validation_study", lambda cases, *, model, client=None: report)
    monkeypatch.setattr(judge_cli, "make_client", lambda: SimpleNamespace())
    monkeypatch.setattr(judge_cli, "load_python_export", lambda p, a: [object()])


def _fixture_file(tmp_path):
    fx = tmp_path / "fx.py"
    fx.write_text("JUDGE_CASES = []\n")  # only needs to exist; load is stubbed
    return fx


def test_validate_mode_gate_pass(tmp_path, monkeypatch, capsys):
    from corpus.eval.validation import ValidationReport

    _stub_validate(
        monkeypatch,
        ValidationReport(kappa=0.9, raw_agreement=0.95, n=6, adversarial_total=4, adversarial_caught=4),
    )
    thresholds = tmp_path / "th.json"
    thresholds.write_text('{"kappa": 0.6}')
    rc = judge_cli.main_argv(
        ["--validate", "--fixture", str(_fixture_file(tmp_path)), "--check", str(thresholds)]
    )
    assert rc == 0
    assert "kappa" in capsys.readouterr().err.lower()


def test_validate_mode_gate_fail_on_low_kappa(tmp_path, monkeypatch):
    from corpus.eval.validation import ValidationReport

    _stub_validate(
        monkeypatch,
        ValidationReport(kappa=0.3, raw_agreement=0.6, n=6, adversarial_total=4, adversarial_caught=4),
    )
    thresholds = tmp_path / "th.json"
    thresholds.write_text('{"kappa": 0.6}')
    rc = judge_cli.main_argv(
        ["--validate", "--fixture", str(_fixture_file(tmp_path)), "--check", str(thresholds)]
    )
    assert rc == 1


def test_validate_mode_gate_fail_on_missed_adversarial(tmp_path, monkeypatch):
    from corpus.eval.validation import ValidationReport

    _stub_validate(
        monkeypatch,
        ValidationReport(kappa=0.9, raw_agreement=0.9, n=6, adversarial_total=4, adversarial_caught=3),
    )
    rc = judge_cli.main_argv(["--validate", "--fixture", str(_fixture_file(tmp_path))])
    assert rc == 1
