"""The chars/4 token heuristic, which every chunk boundary and cost estimate
is built on.

Deliberately not billing-accurate -- the docstring says so and the real count
comes from the embedder -- but it must never return 0 or a negative, because
callers divide by it and use it to size windows.
"""

from __future__ import annotations

import pytest

from corpus.util.tokens import MAX_CHUNK_TOKENS, OVERLAP_TOKENS, estimate_tokens


@pytest.mark.parametrize("text", ["", " ", "a", "ab", "abc"])
def test_a_short_or_empty_string_never_estimates_zero(text: str) -> None:
    """Zero would make a caller divide by zero or size a window at nothing."""
    assert estimate_tokens(text) >= 1


def test_the_estimate_grows_with_length() -> None:
    assert estimate_tokens("x" * 400) > estimate_tokens("x" * 40)


def test_it_follows_the_documented_chars_over_four_heuristic() -> None:
    assert estimate_tokens("x" * 400) == 100


def test_the_overlap_fits_inside_a_chunk() -> None:
    """An overlap at or beyond the chunk size would make windows fail to
    advance, so a chunker built on these could loop."""
    assert 0 < OVERLAP_TOKENS < MAX_CHUNK_TOKENS


def test_multibyte_text_is_counted_by_characters_not_bytes() -> None:
    """CJK packs more meaning per character, so counting bytes would
    over-estimate it threefold and shrink those chunks for no reason."""
    assert estimate_tokens("あ" * 400) == estimate_tokens("x" * 400)
