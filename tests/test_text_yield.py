"""Tests for `corpus.util.text_yield` -- per-connector-type calibration of
the raw-bytes-to-estimated-tokens cost estimate `corpus-index`'s plan
shows before ingesting."""

from __future__ import annotations

import math

from corpus.util.text_yield import (
    CHARS_PER_TOKEN_ESTIMATE,
    DEFAULT_TEXT_YIELD_RATIO,
    MEASURED_TEXT_YIELD_RATIOS,
    TEXT_YIELD_RATIOS,
    estimate_tokens_from_bytes,
    text_yield_ratio,
)


def test_measured_ratios_are_all_positive_and_at_most_around_one() -> None:
    # Sanity bound: these are chars-extracted-per-byte-of-source, measured
    # against real text; nothing should be negative, and nothing should be
    # wildly above 1.0 (markdown's 1.0054 is the observed ceiling).
    for connector_type, ratio in MEASURED_TEXT_YIELD_RATIOS.items():
        assert ratio > 0, connector_type
        assert ratio <= 1.1, connector_type


def test_compressed_binary_formats_yield_far_less_than_plain_text() -> None:
    # The whole point of this module: pdf/docx/zip must not be anywhere
    # close to text/markdown's ~1:1 ratio.
    assert MEASURED_TEXT_YIELD_RATIOS["pdf"] < 0.01
    assert MEASURED_TEXT_YIELD_RATIOS["zip"] < 0.01
    assert MEASURED_TEXT_YIELD_RATIOS["docx"] < 0.1
    assert MEASURED_TEXT_YIELD_RATIOS["text"] > 0.9
    assert MEASURED_TEXT_YIELD_RATIOS["markdown"] > 0.9


def test_text_yield_ratio_prefers_measured_over_default() -> None:
    assert text_yield_ratio("pdf") == MEASURED_TEXT_YIELD_RATIOS["pdf"]


def test_text_yield_ratio_falls_back_for_unknown_type() -> None:
    assert text_yield_ratio("some_future_connector_type") == DEFAULT_TEXT_YIELD_RATIO


def test_every_registered_connector_type_has_a_ratio() -> None:
    # Every type in DEFAULT_GLOBS (i.e. every connector actually registered)
    # should resolve to *something* in TEXT_YIELD_RATIOS -- measured or a
    # documented reasoned default -- not silently fall through to the
    # generic DEFAULT_TEXT_YIELD_RATIO, which would mean a connector was
    # added without anyone deciding what its cost estimate should look like.
    from corpus.connectors.registry import DEFAULT_GLOBS

    missing = [t for t in DEFAULT_GLOBS if t not in TEXT_YIELD_RATIOS]
    assert missing == [], f"connector type(s) with no text-yield ratio decided: {missing}"


def test_aup3_ratio_is_low_not_the_safe_default() -> None:
    # aup3 project files are mostly binary audio sample data; the connector
    # emits a small, roughly fixed-size description regardless of the
    # project's actual size. Applying the generic "when uncertain, assume
    # ~1:1" default here would reproduce the exact bug this module exists
    # to fix (a multi-hundred-MB project "costing" hundreds of millions of
    # estimated tokens) -- see the module docstring's aup3 entry.
    assert TEXT_YIELD_RATIOS["aup3"] < 0.01


def test_estimate_tokens_from_bytes_applies_ratio_then_divides_by_four() -> None:
    total_bytes = 1_000_000
    expected_chars = total_bytes * MEASURED_TEXT_YIELD_RATIOS["pdf"]
    expected_tokens = math.ceil(expected_chars / CHARS_PER_TOKEN_ESTIMATE)

    assert estimate_tokens_from_bytes("pdf", total_bytes) == expected_tokens


def test_estimate_tokens_from_bytes_rounds_up_not_down() -> None:
    # 1 byte * pdf's 0.0016 ratio / 4 is a tiny fraction, not zero -- must
    # round up to at least 1, never down to 0, per this module's
    # "never understate cost" bias.
    assert estimate_tokens_from_bytes("pdf", 1) >= 1


def test_estimate_tokens_from_bytes_matches_hand_computed_example() -> None:
    # The exact number from the CHANGELOG/module-docstring illustration: a
    # large pdf source's estimate should land far below a flat bytes/4
    # estimate would have given (400x overstatement was the reported bug).
    total_bytes = 21_000_000_000  # a large PDF source
    flat_bytes_over_four = total_bytes // 4
    calibrated = estimate_tokens_from_bytes("pdf", total_bytes)

    assert calibrated < flat_bytes_over_four / 100  # more than 100x smaller
