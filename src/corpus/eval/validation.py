"""Judge validation: Cohen's κ plus a study runner that certifies the judge
against human faithfulness labels.

`cohens_kappa` is a pure function (no corpus/anthropic internals). The study
runner (added in Task 6) lives here too but only reaches for the judge — and
thus the Anthropic SDK — lazily via `corpus.eval.judge`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .judge import JUDGE_DEFAULT_MODEL, judge_answer

if TYPE_CHECKING:
    from anthropic import Anthropic


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


@dataclass(frozen=True)
class JudgeCase:
    """One frozen validation case. `context` is stored so `--validate` needs no
    DB — the judge sees exactly these (source_key, text) pairs. All content is
    self-authored public text (no private-corpus data)."""

    query: str
    answer: str
    cited_keys: list[str]
    context: list[tuple[str, str]]
    human_faithful: bool | None = None
    adversarial: bool = False
    note: str = ""


@dataclass(frozen=True)
class ValidationReport:
    kappa: float
    raw_agreement: float
    n: int
    adversarial_total: int
    adversarial_caught: int

    @property
    def adversarial_ok(self) -> bool:
        return self.adversarial_caught == self.adversarial_total


def run_validation_study(
    cases: Sequence[JudgeCase],
    *,
    model: str = JUDGE_DEFAULT_MODEL,
    client: Anthropic | None = None,
) -> ValidationReport:
    """Run the judge over the labeled cases and report κ (judge faithfulness vs
    human) + raw agreement + the adversarial-catch check."""
    labeled = [c for c in cases if c.human_faithful is not None]
    if not labeled:
        raise ValueError("no labeled cases (human_faithful is None for all)")
    human: list[bool] = []
    judge: list[bool] = []
    adversarial_total = 0
    adversarial_caught = 0
    for case in labeled:
        verdict = judge_answer(
            case.query, case.answer, case.cited_keys, case.context, model=model, client=client
        )
        judged_faithful = verdict.faithfulness.passed
        assert case.human_faithful is not None  # narrowed by the `labeled` filter
        human.append(case.human_faithful)
        judge.append(judged_faithful)
        if case.adversarial:
            adversarial_total += 1
            if not judged_faithful:
                adversarial_caught += 1
    raw = sum(1 for h, j in zip(human, judge, strict=True) if h == j) / len(labeled)
    return ValidationReport(
        kappa=cohens_kappa(human, judge),
        raw_agreement=raw,
        n=len(labeled),
        adversarial_total=adversarial_total,
        adversarial_caught=adversarial_caught,
    )
