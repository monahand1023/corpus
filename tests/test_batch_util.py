"""Batch packing and retry — both guard a money-spending run, neither tested.

`pack_batches` must respect BOTH caps the API enforces. Over-packing means the
request is rejected after the caller believes it was sent; under-packing means
more batches than necessary, which is merely wasteful.

`retry_with_backoff` is what stops a transient network blip from killing a
long paid run. A retry that re-raises too early wastes the run; one that
retries forever hangs it.
"""

from __future__ import annotations

import json

import pytest

from corpus.contextual.batch_util import pack_batches, retry_with_backoff

# --- pack_batches ------------------------------------------------------------


def test_requests_within_both_caps_stay_in_one_batch() -> None:
    reqs = [{"id": i} for i in range(5)]
    assert pack_batches(reqs) == [reqs]


def test_the_request_count_cap_is_respected() -> None:
    reqs = [{"id": i} for i in range(10)]
    batches = pack_batches(reqs, max_requests=3)
    assert [len(b) for b in batches] == [3, 3, 3, 1]


def test_the_byte_cap_is_respected() -> None:
    reqs = [{"id": i, "pad": "x" * 100} for i in range(10)]
    size = len(json.dumps(reqs[0]))
    batches = pack_batches(reqs, max_bytes=size * 3)
    assert all(sum(len(json.dumps(r)) for r in b) <= size * 3 for b in batches)
    assert sum(len(b) for b in batches) == 10, "no request may be dropped"


def test_no_request_is_lost_or_duplicated() -> None:
    """The property that matters: packing is a partition."""
    reqs = [{"id": i, "pad": "y" * (i * 7)} for i in range(40)]
    batches = pack_batches(reqs, max_requests=6, max_bytes=300)
    flat = [r for b in batches for r in b]
    assert flat == reqs


def test_an_empty_input_produces_no_batches() -> None:
    assert pack_batches([]) == []


def test_a_single_oversized_request_is_still_emitted() -> None:
    """It cannot be split, so dropping it silently would be the worse failure;
    the API will reject it and say so."""
    big = {"id": 1, "pad": "z" * 10_000}
    assert pack_batches([big], max_bytes=10) == [[big]]


# --- retry_with_backoff ------------------------------------------------------


def test_a_successful_call_is_not_retried() -> None:
    calls = []

    def fn():
        calls.append(1)
        return "ok"

    assert retry_with_backoff(fn, "thing") == "ok"
    assert len(calls) == 1


def test_a_transient_failure_is_retried_then_succeeds(monkeypatch) -> None:
    monkeypatch.setattr("corpus.contextual.batch_util.time.sleep", lambda _s: None)
    calls = []

    def fn():
        calls.append(1)
        if len(calls) < 3:
            raise ConnectionError("blip")
        return "ok"

    assert retry_with_backoff(fn, "thing", attempts=5) == "ok"
    assert len(calls) == 3


def test_the_last_failure_is_raised_rather_than_swallowed(monkeypatch) -> None:
    """A run that quietly returns None after exhausting retries would corrupt
    whatever consumed the result."""
    monkeypatch.setattr("corpus.contextual.batch_util.time.sleep", lambda _s: None)

    def fn():
        raise ConnectionError("always down")

    with pytest.raises(ConnectionError, match="always down"):
        retry_with_backoff(fn, "thing", attempts=3)


def test_backoff_grows_and_is_capped(monkeypatch) -> None:
    """2s doubling to a 60s ceiling: uncapped growth would stall a long run."""
    slept: list[float] = []
    monkeypatch.setattr("corpus.contextual.batch_util.time.sleep", slept.append)

    def fn():
        raise ConnectionError("down")

    with pytest.raises(ConnectionError):
        retry_with_backoff(fn, "thing", attempts=10)

    assert slept[0] == 2.0
    assert slept == sorted(slept), "delays must be non-decreasing"
    assert max(slept) <= 60.0, f"backoff exceeded its cap: {slept}"
    assert len(slept) == 9, "one sleep per failure except the last"
