"""Reservoir sampling (Algorithm R): draw a uniform random sample of fixed
size from a stream of unknown length in O(sample_size) memory.

Both `media.py` (which audio/video files to hand to `ffprobe`) and
`overlap.py` (which documents to extract phrases from) need this: the
population can be hundreds of thousands of files, and accumulating it all
just to call `random.sample` on it afterward would defeat the whole
"stream, don't accumulate" requirement this tool is built around.
"""

from __future__ import annotations

import random
from collections.abc import Iterable
from typing import TypeVar

T = TypeVar("T")


class ReservoirSampler[T]:
    """Incremental Algorithm R, fed one item at a time via `offer`.

    Needed (rather than the simpler "sample from a finite iterable" form)
    whenever several independent samples are drawn from ONE interleaved
    stream in a single pass — `media.py` reservoir-samples per file
    extension while walking a tree exactly once.
    """

    def __init__(self, k: int, rng: random.Random | None = None):
        self.k = k
        self._rng = rng if rng is not None else random.Random()
        self.sample: list[T] = []
        self.seen = 0

    def offer(self, item: T) -> None:
        self.seen += 1
        if len(self.sample) < self.k:
            self.sample.append(item)
            return
        j = self._rng.randint(0, self.seen - 1)
        if j < self.k:
            self.sample[j] = item


def reservoir_sample(
    items: Iterable[T], k: int, rng: random.Random | None = None
) -> tuple[list[T], int]:
    """Sample up to `k` items uniformly from `items` in one pass.

    Returns `(sample, total_seen)` — `total_seen` is the full population
    count even though only `k` items are retained, so a caller can report
    "sampled N of M".
    """
    sampler: ReservoirSampler[T] = ReservoirSampler(k, rng)
    for item in items:
        sampler.offer(item)
    return sampler.sample, sampler.seen
