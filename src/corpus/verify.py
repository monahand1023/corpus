"""Making a check say what it examined, so "nothing" cannot read as "fine".

THE DEFECT THIS EXISTS FOR. Every verification failure found while hardening
this project had one shape: a check that did not run, or ran against the wrong
surface, is indistinguishable from a check that passed.

  * a repetition filter sat at a 0.9 threshold whose highest score across 74
    real transcripts was 0.250 -- on real data it had never once fired
  * `corpus-doctor` printed "a skipped check is not a passing one" and then
    skipped its own query-log check by default, on every run of five archives
  * `corpus-smoke` scoped itself to one launcher and reported "1/2 servers
    healthy" on a machine with four archives configured
  * an index-quality scan of an EMPTY index reported "no artefacts found"
  * a privacy grep exited 2 (error, because of a shell alias mangling its
    arguments) and the exit status was read as "no matches" -- three times

None of those were carelessness. Each was a correct answer to the wrong
question, and each got MORE convincing the more times it was repeated, because
every repetition shared the same blind spot.

THE RULE. A check reports what it examined. Examining zero is never a pass; it
is an unknown, and an unknown must be louder than a pass, not quieter.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class Coverage:
    """What a check actually looked at.

    `unit` is a plural noun -- "chunks", "servers", "queries" -- because the
    describing string is read by a person skimming output, and "0 chunks"
    scans as a pass where "examined nothing" does not.
    """

    examined: int
    unit: str

    def __post_init__(self) -> None:
        if self.examined < 0:
            raise ValueError(f"examined cannot be negative: {self.examined}")

    @property
    def vacuous(self) -> bool:
        """True when this check proved nothing, having looked at nothing."""
        return self.examined == 0

    def __bool__(self) -> bool:
        """Falsy when vacuous, so `if not coverage:` guards read naturally.

        The comparison people forget is `count == 0`; making the object itself
        falsy removes the need to remember it.
        """
        return not self.vacuous

    def describe(self) -> str:
        if self.vacuous:
            return f"examined nothing ({self.unit})"
        return f"{self.examined:,} {self.unit}"


class DetectorBroken(RuntimeError):
    """A detector could not find a positive it is known to match.

    Raised instead of returning "nothing found", because those are different
    facts and only one of them is reassuring.
    """


def self_check(
    detector: Callable[[str], bool], *, positive: str, label: str
) -> None:
    """Prove `detector` fires on a string it must match, before trusting it.

    A scanner built on a detector that cannot fire reports a clean result
    forever. That is not hypothetical here: a review of an early caption
    phrase list found 22 of 45 entries could never match anything, and the
    tests passed because they asserted against the list rather than against
    strings a model really produces.

    `positive` is deliberately a caller-supplied sample rather than something
    generated here -- the planted needle has to be the shape the real detector
    is meant to catch, and only the caller knows that.
    """
    try:
        fired = detector(positive)
    except Exception as exc:  # a detector that throws has not "found nothing"
        raise DetectorBroken(
            f"{label}: detector raised {type(exc).__name__} on a known positive, "
            f"so this scan cannot be trusted"
        ) from exc
    if not fired:
        raise DetectorBroken(
            f"{label}: detector did not fire on a known positive, so it cannot "
            f"find anything -- this scan proves nothing"
        )


# Below this many verdicts, "never fired" is noise rather than evidence. A
# guard that fires on a three-file sample gets switched off, which leaves you
# worse off than no guard -- the same reason the caption tail matcher dropped
# its over-broad heuristics.
MIN_VERDICTS_FOR_DORMANCY = 200


def dormant(
    counts: Mapping[str, int],
    *,
    known: Iterable[str],
    coverage: Coverage,
    minimum: int = MIN_VERDICTS_FOR_DORMANCY,
) -> list[str]:
    """Known filters that never fired, once the sample is big enough to mean it.

    A filter that never fires across a large corpus is either unnecessary or
    broken, and there is no third option. This project shipped one: a
    repetition check thresholded at 0.9 whose highest score across 74 real
    transcripts was 0.250. Nothing reported it, because a dead knob reads
    exactly like a knob with nothing to reject.

    Returns [] below `minimum` rather than guessing: zero rejections out of
    three files is not evidence of anything.

    `counts` must come from ONE rule set. Counting verdicts recorded under
    older rules against a current filter list manufactures dead filters that
    are not dead -- measured on a real sidecar, three looked dormant purely
    because an earlier pipeline spelled its reasons differently. See
    `corpus.transcripts.store.filter_activity(policy=...)`.
    """
    if coverage.examined < minimum:
        return []
    return sorted(name for name in known if counts.get(name, 0) == 0)


# Below this much headroom, a threshold is close enough to real data that
# ordinary variation can cross it. Not a failure -- a number to look at.
TIGHT_MARGIN_PERCENT = 25.0


@dataclass(frozen=True)
class Margin:
    """How much room a threshold leaves above the data it must not reject.

    `percent` is None when nothing was observed, because a margin against no
    data is not a small margin -- it is no measurement.
    """

    label: str
    threshold: float
    observed_max: float | None
    percent: float | None

    @property
    def tight(self) -> bool:
        return self.percent is not None and self.percent < TIGHT_MARGIN_PERCENT

    def describe(self) -> str:
        """Headroom, stated against what was actually measured.

        It says "kept material", never "real data". The distinction is not
        pedantry -- it was a live misreading. corpus-doctor reported
        "looping share: 0.7% headroom ... real data reaches 0.844444", and
        that 0.844444 was a Whisper decode loop: the same Japanese sentence
        thirteen times over a video of a child in a playroom. Junk the
        ceiling was too permissive to catch, described as the material the
        ceiling exists to protect.

        The action that invites -- raise the ceiling, there is no room -- is
        exactly backwards. A threshold that is too permissive ALWAYS looks
        tight, because its own failures are in the sample it is measured
        against. The check cannot tell real from junk, so it must not claim
        to; the person reading it can, given the evidence.
        """
        if self.percent is None:
            return f"{self.label}: observed nothing, so there is no margin to report"
        return (
            f"{self.label}: {self.percent:.1f}% headroom "
            f"(threshold {self.threshold:g}, kept material reaches "
            f"{self.observed_max:g})"
        )


def margin(*, threshold: float, observed_max: float | None, label: str) -> Margin:
    """Headroom between a threshold and the highest real value beneath it.

    THE DEFECT THIS EXISTS FOR, twice in two days. A speech-rate ceiling sat
    0.6% above the fastest real speech, having been set when the archive's
    maximum was 41% lower -- the threshold never moved, the data did. A loop
    ceiling sat 0.3% above the highest real score and was deleting family
    recordings: one birthday song survived by 0.012 while another was deleted.

    Both were found by hand, by accident, and there are eleven tunable
    thresholds. A margin is REPORTED rather than enforced, because a threshold
    may legitimately sit near data it is meant to nearly touch -- what it must
    not do is sit there unnoticed.

    A negative percent means real data already crosses the threshold, so it is
    actively rejecting real material.
    """
    if observed_max is None or observed_max <= 0:
        return Margin(label, threshold, observed_max, None)
    return Margin(label, threshold, observed_max, (threshold / observed_max - 1) * 100)


__all__ = [
    "MIN_VERDICTS_FOR_DORMANCY",
    "TIGHT_MARGIN_PERCENT",
    "Coverage",
    "DetectorBroken",
    "Margin",
    "dormant",
    "margin",
    "self_check",
]
