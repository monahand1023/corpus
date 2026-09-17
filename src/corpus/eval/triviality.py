"""Would a retriever that does not work score well on this gold set?

recall@k asks whether ANY acceptable answer landed in the top k. That is the
right question, and it has a failure mode: when a query accepts a large
fraction of the archive, five arbitrary results contain an acceptable answer
by chance, and the query scores well for a retriever that is doing nothing.

Measured on a real real archive: one query accepted 27,946
documents, which five random results hit 53% of the time. Its contribution
to the gate was a coin flip wearing the shape of a measurement. The other
23 queries sat at or below 0.094, and the set as a whole scored 0.050
against a floor of 0.708 -- so the gate WAS earned, and exactly one query
was not pulling its weight.

That ratio is why the default threshold is a coin flip rather than
something stricter. A check that flagged broad-but-discriminating queries
would fire on every archive with topical questions in its gold set, and a
guard that cries wolf gets switched off -- taking the real signal with it.

THE CLOSED FORM. Drawing k results uniformly from N documents of which K
are acceptable, the chance that at least one is acceptable is

    1 - (1 - K/N)^k

Sampling without replacement is very slightly different; the difference is
immaterial at these sizes and the approximation errs toward reporting a
HIGHER random score, which is the safe direction for a check that is
looking for queries that are too easy.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

# A coin flip. Above this, a query tells you more about the size of its
# answer set than about the retriever.
DEFAULT_TRIVIAL_ABOVE = 0.5


def random_hit_rate(*, keys: int, documents: int, top_k: int) -> float:
    """Chance that `top_k` uniformly drawn documents include an acceptable one."""
    if documents <= 0 or keys <= 0 or top_k <= 0:
        return 0.0
    return 1.0 - (1.0 - min(keys, documents) / documents) ** top_k


@dataclass(frozen=True)
class TrivialityReport:
    documents: int
    top_k: int
    rates: dict[str, float]
    trivial: list[str]
    threshold: float
    key_counts: Mapping[str, int]

    @property
    def random_recall(self) -> float:
        """What a random retriever would score on the WHOLE set.

        The number that says whether a floor is a gate at all: a floor of
        0.708 over a set chance scores 0.700 on is not measuring retrieval.
        """
        return sum(self.rates.values()) / len(self.rates) if self.rates else 0.0

    @property
    def is_clean(self) -> bool:
        return not self.trivial

    def describe(self) -> str:
        if self.documents <= 0:
            return (
                "cannot judge triviality: the archive reports 0 documents, so "
                "no query's difficulty is known (this is not a clean result)"
            )
        head = (
            f"a random retriever would score recall@{self.top_k} "
            f"{self.random_recall:.3f} on this gold set "
            f"({self.documents:,} documents)"
        )
        if not self.trivial:
            return head + "; no query is trivially satisfiable"
        lines = [head + f"; {len(self.trivial)} query(s) above {self.threshold:.2f}:"]
        for name in self.trivial:
            lines.append(
                f"  {self.rates[name]:.3f}  {self.key_counts[name]:,} acceptable "
                f"answer(s)  {name}"
            )
        lines.append(
            "  A query chance passes this often measures the size of its "
            "answer set, not the retriever. Narrow it or drop it."
        )
        return "\n".join(lines)


def triviality_report(
    key_counts: Mapping[str, int],
    *,
    documents: int,
    top_k: int,
    threshold: float = DEFAULT_TRIVIAL_ABOVE,
) -> TrivialityReport:
    """Flag gold-set queries a random retriever would satisfy.

    `key_counts` maps a query to how many documents it accepts. A count of 0
    is a NEGATIVE control -- a query that exists to check nothing comes back
    -- and is never trivial; scoring it would flag the one query shape that
    is deliberately unanswerable.
    """
    for name, count in key_counts.items():
        if documents > 0 and count > documents:
            raise ValueError(
                f"gold-set query {name!r} names {count:,} more acceptable "
                f"answers than the archive holds ({documents:,} documents). "
                "That is a defect in the gold set, not a hard query -- "
                "clamping it would hide it behind a plausible number."
            )
    rates = {
        name: random_hit_rate(keys=count, documents=documents, top_k=top_k)
        for name, count in key_counts.items()
    }
    trivial = sorted(
        (n for n, r in rates.items() if r > threshold),
        key=lambda n: -rates[n],
    )
    return TrivialityReport(
        documents=documents,
        top_k=top_k,
        rates=rates,
        trivial=trivial,
        threshold=threshold,
        key_counts=dict(key_counts),
    )


__all__ = [
    "DEFAULT_TRIVIAL_ABOVE",
    "TrivialityReport",
    "random_hit_rate",
    "triviality_report",
]
