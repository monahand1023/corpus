from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from corpus.eval.validation import (
    JudgeCase,
    ValidationReport,
    cohens_kappa,
    run_validation_study,
)


def test_kappa_perfect_agreement():
    a = [True, False, True, False]
    assert cohens_kappa(a, a) == pytest.approx(1.0)


def test_kappa_total_disagreement_is_negative():
    a = [True, True, False, False]
    b = [False, False, True, True]
    assert cohens_kappa(a, b) < 0


def test_kappa_known_2x2_value():
    # 10 items. Both raters "True" on 5, both "False" on 3, split on 2.
    # po = 8/10 = 0.8. a_true = 6/10, b_true = 6/10.
    # pe = 0.6*0.6 + 0.4*0.4 = 0.52. kappa = (0.8-0.52)/(1-0.52) = 0.5833...
    a = [True] * 6 + [False] * 4
    b = [True] * 5 + [False] * 3 + [True, False]
    assert cohens_kappa(a, b) == pytest.approx(0.5833333, abs=1e-6)


def test_kappa_all_same_category_both_raters_is_one():
    a = [True, True, True]
    b = [True, True, True]
    assert cohens_kappa(a, b) == pytest.approx(1.0)


def test_kappa_length_mismatch_raises():
    with pytest.raises(ValueError, match="length"):
        cohens_kappa([True, False], [True])


def test_kappa_empty_raises():
    with pytest.raises(ValueError, match="zero"):
        cohens_kappa([], [])


def _verdict_resp(faithful: bool):
    inp = {
        "faithfulness_rationale": "r",
        "faithfulness_passed": faithful,
        "answer_relevance_rationale": "r",
        "answer_relevance_passed": True,
        "citation_correctness_rationale": "r",
        "citation_correctness_passed": True,
    }
    block = SimpleNamespace(type="tool_use", name="record_verdict", input=inp)
    return SimpleNamespace(content=[block], usage=SimpleNamespace(input_tokens=1, output_tokens=1))


def test_run_validation_study_perfect_agreement():
    cases = [
        JudgeCase("q1", "a1", ["k"], [("k", "c")], human_faithful=True),
        JudgeCase("q2", "a2", ["k"], [("k", "c")], human_faithful=False, adversarial=True),
    ]
    # Judge returns faithful=True for q1, faithful=False for q2 → matches humans.
    create = MagicMock(side_effect=[_verdict_resp(True), _verdict_resp(False)])
    client = SimpleNamespace(messages=SimpleNamespace(create=create))
    report = run_validation_study(cases, client=client)
    assert isinstance(report, ValidationReport)
    assert report.n == 2
    assert report.kappa == 1.0
    assert report.raw_agreement == 1.0
    assert report.adversarial_total == 1
    assert report.adversarial_caught == 1
    assert report.adversarial_ok is True


def test_run_validation_study_skips_unlabeled():
    cases = [
        JudgeCase("q1", "a1", ["k"], [("k", "c")], human_faithful=True),
        JudgeCase("q2", "a2", ["k"], [("k", "c")], human_faithful=None),  # unlabeled → skipped
    ]
    create = MagicMock(side_effect=[_verdict_resp(True)])
    client = SimpleNamespace(messages=SimpleNamespace(create=create))
    report = run_validation_study(cases, client=client)
    assert report.n == 1
    assert create.call_count == 1


def test_run_validation_study_no_labels_raises():
    cases = [JudgeCase("q1", "a1", ["k"], [("k", "c")], human_faithful=None)]
    client = SimpleNamespace(messages=SimpleNamespace(create=MagicMock()))
    with pytest.raises(ValueError, match="labeled"):
        run_validation_study(cases, client=client)
