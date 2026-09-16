"""Escaping for hand-rendered TOML, which writes the user's corpus.toml.

`corpus-init` and `corpus-index` both render `[[sources]]` blocks as text.
A path that escapes badly produces a corpus.toml that will not parse -- and
the user discovers it on the NEXT run, having already lost the working config
that was overwritten.

Both characters this guards against are legal in APFS filenames, so they are
reachable by anyone who names a folder carelessly rather than maliciously.
"""

from __future__ import annotations

import tomllib

import pytest

from corpus.util.toml_write import toml_str


@pytest.mark.parametrize(
    "raw",
    [
        "plain",
        "/Users/x/Documents",
        'has "quotes" inside',
        r"has\backslashes",
        r'both "quoted" and \back\slashed',
        "trailing backslash\\",
        "",
        "unicode → 日本語 → emoji 🎵",
        "newline\nin\npath",
        "tab\tseparated",
    ],
)
def test_an_escaped_value_round_trips_through_a_real_toml_parser(raw: str) -> None:
    """The only assertion that matters: stdlib tomllib reads back what went in.

    Asserting the escaped TEXT would test the escaping's spelling; parsing it
    tests the property the caller actually depends on.
    """
    document = f"path = {toml_str(raw)}\n"
    assert tomllib.loads(document)["path"] == raw


def test_a_windows_style_path_survives() -> None:
    raw = r"C:\Users\dan\Documents\notes"
    assert tomllib.loads(f"p = {toml_str(raw)}\n")["p"] == raw


def test_the_result_is_a_quoted_scalar_not_a_bare_word() -> None:
    assert toml_str("plain").startswith('"') and toml_str("plain").endswith('"')


def test_a_value_that_would_otherwise_close_the_string_early() -> None:
    """The failure mode: an unescaped quote ends the string and the rest
    becomes syntax, so the file parses as something else entirely or not at
    all."""
    raw = '" ; injected = "oops'
    assert tomllib.loads(f"v = {toml_str(raw)}\n") == {"v": raw}


@pytest.mark.parametrize(
    "raw",
    [
        "newline\nhere",
        "carriage\rreturn",
        "form\ffeed",
        "back\bspace",
        "null\x00byte",
        "bell\x07",
        "delete\x7f",
        "tab\there",          # the one control character TOML allows literally
    ],
)
def test_control_characters_are_escaped(raw: str) -> None:
    """Every byte except `/` and NUL is a legal filename character on APFS and
    ext4, so these are reachable by anyone who names a folder carelessly."""
    assert tomllib.loads(f"p = {toml_str(raw)}\n")["p"] == raw
