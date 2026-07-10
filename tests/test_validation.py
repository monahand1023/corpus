from __future__ import annotations

import pytest

from corpus.eval.validation import cohens_kappa


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
