"""Tiny formatting helpers shared by every survey subcommand's human-readable
output. Kept dependency-free (no `humanize`, no rich) on purpose — corpus's
base install stays minimal, and a survey CLI is exactly the kind of thing
that should work in a bare checkout with nothing extra installed."""

from __future__ import annotations

_SIZE_UNITS = ("B", "KB", "MB", "GB", "TB", "PB")


def human_size(num_bytes: int | float) -> str:
    """`1536` -> `"1.5 KB"`. Binary (1024) units, one decimal place, `0 B`
    for zero rather than `0.0 B`."""
    if num_bytes == 0:
        return "0 B"
    n = float(num_bytes)
    for unit in _SIZE_UNITS:
        if abs(n) < 1024.0 or unit == _SIZE_UNITS[-1]:
            return f"{n:.1f} {unit}" if unit != "B" else f"{int(n)} {unit}"
        n /= 1024.0
    return f"{n:.1f} PB"  # unreachable, satisfies mypy's control-flow check


def human_count(n: int) -> str:
    return f"{n:,}"
