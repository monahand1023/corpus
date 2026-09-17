"""Pricing a run must not take longer than the run's first minute.

`corpus-transcribe --dry-run` is, by its own docstring, "the documented first
step" and "the only warning before hours of compute". It reads each file's
duration with an `ffprobe` subprocess, and it did so one file at a time.

The cost is not the disk. Measured on a real archive: a plain `find` over the
same library reached 50,000 files in 0.035 seconds, while the dry run was
still going after nine minutes -- because it was spawning and awaiting one
subprocess per file, serially. Every one of those probes is independent and
spends nearly all its time waiting, which is the definition of work that
should overlap.

A safety step slow enough to be skipped stops being a safety step.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

from corpus.survey.media import probe_all


def test_probe_all_returns_exactly_what_a_serial_map_would() -> None:
    paths = [Path(f"/tmp/clip-{i}.mov") for i in range(25)]

    def probe(path: Path) -> float | None:
        return float(len(path.name))

    assert probe_all(paths, probe) == {p: probe(p) for p in paths}


def test_probe_all_keeps_a_none_without_dropping_the_entry() -> None:
    """A duration that cannot be read is meaningful: callers treat it as
    "long enough to keep". Losing the key would silently change that."""
    paths = [Path("/tmp/a.mov"), Path("/tmp/b.mov")]

    result = probe_all(paths, lambda p: None if p.name == "a.mov" else 3.0)

    assert result == {Path("/tmp/a.mov"): None, Path("/tmp/b.mov"): 3.0}


def test_probe_all_overlaps_its_probes() -> None:
    """The point of the change. A serial implementation passes every test
    above; only concurrency distinguishes it."""
    lock = threading.Lock()
    live = 0
    high_water = 0

    def probe(path: Path) -> float | None:
        nonlocal live, high_water
        with lock:
            live += 1
            high_water = max(high_water, live)
        time.sleep(0.05)  # stands in for an ffprobe subprocess
        with lock:
            live -= 1
        return 1.0

    probe_all([Path(f"/tmp/{i}.mov") for i in range(16)], probe, workers=8)

    assert high_water > 1, (
        f"probes never overlapped (high water mark {high_water}); "
        "pricing a run is still serial"
    )


def test_probe_all_handles_an_empty_list() -> None:
    assert probe_all([], lambda p: 1.0) == {}
