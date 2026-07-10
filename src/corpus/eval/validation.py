"""Judge validation: Cohen's κ plus a study runner that certifies the judge
against human faithfulness labels.

`cohens_kappa` is a pure function (no corpus/anthropic internals). The study
runner (added in Task 6) lives here too but only reaches for the judge — and
thus the Anthropic SDK — lazily via `corpus.eval.judge`.
"""

from __future__ import annotations

from collections.abc import Sequence


def cohens_kappa(a: Sequence[bool], b: Sequence[bool]) -> float:
    """Cohen's κ for two binary raters over the same items.

    κ = (po - pe) / (1 - pe), where po is observed agreement and pe is
    chance agreement from the raters' marginals. Returns 1.0 when both raters
    place every item in the same single category (pe == 1: no chance-adjustable
    disagreement is possible, so perfect agreement by convention).
    """
    if len(a) != len(b):
        raise ValueError("rater sequences must be the same length")
    n = len(a)
    if n == 0:
        raise ValueError("cannot compute kappa over zero items")
    po = sum(1 for x, y in zip(a, b, strict=True) if x == y) / n
    pa_true = sum(1 for x in a if x) / n
    pb_true = sum(1 for y in b if y) / n
    pe = pa_true * pb_true + (1 - pa_true) * (1 - pb_true)
    if pe == 1.0:
        return 1.0
    return (po - pe) / (1 - pe)
