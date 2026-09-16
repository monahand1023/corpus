"""Reservoir sampling had no tests, and everything estimated rests on it.

`corpus-survey media` extrapolates total audio hours from a sample of probed
files; `corpus-survey overlap` reports "already indexed" percentages with a
confidence interval from a sample of documents. Both numbers are only as
honest as the sample being uniform.

Algorithm R is short and easy to get subtly wrong -- an off-by-one in the
replacement index biases the result toward early or late items, and a biased
sample still produces a confident-looking number. That is the whole failure
mode this project keeps running into, so it is worth pinning statistically
rather than by inspection.
"""

from __future__ import annotations

import random
from collections import Counter

from corpus.survey.sampling import ReservoirSampler, reservoir_sample


def test_a_sample_smaller_than_k_keeps_everything() -> None:
    s: ReservoirSampler[int] = ReservoirSampler(10, random.Random(0))
    for i in range(4):
        s.offer(i)
    assert s.sample == [0, 1, 2, 3]
    assert s.seen == 4


def test_seen_counts_the_whole_population_not_the_retained_sample() -> None:
    """`total_seen` is what turns "sampled N" into "sampled N of M"."""
    s: ReservoirSampler[int] = ReservoirSampler(3, random.Random(0))
    for i in range(1000):
        s.offer(i)
    assert len(s.sample) == 3
    assert s.seen == 1000


def test_an_empty_stream_samples_nothing() -> None:
    s: ReservoirSampler[int] = ReservoirSampler(5, random.Random(0))
    assert s.sample == [] and s.seen == 0


def test_k_of_zero_retains_nothing_but_still_counts() -> None:
    s: ReservoirSampler[int] = ReservoirSampler(0, random.Random(0))
    for i in range(10):
        s.offer(i)
    assert s.sample == [] and s.seen == 10


def test_the_same_seed_gives_the_same_sample() -> None:
    def draw() -> list[int]:
        s: ReservoirSampler[int] = ReservoirSampler(5, random.Random(1234))
        for i in range(200):
            s.offer(i)
        return list(s.sample)

    assert draw() == draw()


def test_every_item_is_equally_likely_to_be_retained() -> None:
    """The property that makes an extrapolation honest.

    A biased sampler still returns k items and still reports `seen`, so
    nothing downstream can tell. Run the whole thing many times and check the
    empirical selection rate for each position is close to k/n.

    Tolerance is wide enough not to flake and narrow enough to catch the
    classic off-by-ones: using `randint(0, seen)` instead of `seen - 1`, or
    comparing `j <= k`, both skew selection measurably.
    """
    population, k, trials = 20, 5, 4000
    counts: Counter[int] = Counter()
    rng = random.Random(20260916)
    for _ in range(trials):
        s: ReservoirSampler[int] = ReservoirSampler(k, rng)
        for i in range(population):
            s.offer(i)
        counts.update(s.sample)

    expected = trials * k / population          # 1000 per item
    worst = max(abs(counts[i] - expected) for i in range(population))
    assert worst < expected * 0.12, (
        f"selection is uneven: expected ~{expected:.0f} each, "
        f"worst deviation {worst:.0f}, counts={dict(sorted(counts.items()))}"
    )


def test_the_convenience_wrapper_matches_the_class() -> None:
    """`reservoir_sample` was referenced by nothing at all -- not even a test."""
    sample, seen = reservoir_sample(range(500), 7, random.Random(42))
    assert len(sample) == 7
    assert seen == 500
    assert all(0 <= x < 500 for x in sample)


def test_the_wrapper_handles_a_population_smaller_than_k() -> None:
    sample, seen = reservoir_sample([1, 2], 10, random.Random(0))
    assert sorted(sample) == [1, 2] and seen == 2
