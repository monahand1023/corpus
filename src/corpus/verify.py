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


__all__ = ["Coverage"]
