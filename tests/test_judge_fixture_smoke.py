from __future__ import annotations

from corpus.eval.validation import JudgeCase


def test_fixture_imports_and_shape():
    from tests.judge_fixture import JUDGE_CASES

    assert isinstance(JUDGE_CASES, list)
    assert len(JUDGE_CASES) >= 6
    assert all(isinstance(c, JudgeCase) for c in JUDGE_CASES)


def test_every_case_is_labeled():
    from tests.judge_fixture import JUDGE_CASES

    assert all(c.human_faithful is not None for c in JUDGE_CASES)


def test_adversarial_cases_are_labeled_unfaithful():
    from tests.judge_fixture import JUDGE_CASES

    adversarial = [c for c in JUDGE_CASES if c.adversarial]
    assert len(adversarial) >= 4
    assert all(c.human_faithful is False for c in adversarial)


def test_context_is_self_contained():
    from tests.judge_fixture import JUDGE_CASES

    for c in JUDGE_CASES:
        assert c.context, f"case {c.query!r} has empty context"
        assert all(isinstance(k, str) and isinstance(t, str) for k, t in c.context)
