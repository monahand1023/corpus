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


__all__ = [
    "MIN_VERDICTS_FOR_DORMANCY",
    "Coverage",
    "DetectorBroken",
    "dormant",
    "self_check",
]
