"""Turning a recording into windows, and windows back into one transcript.

None of this touches a model or an audio library. It is the arithmetic of
where to cut and how to rejoin, which is where the subtle mistakes live and
which is therefore the part worth testing exhaustively.

WHY WINDOW AT ALL. A speech-to-text model detects language once, usually on
the first thirty seconds, and applies that verdict to the whole file. On a
recording that starts in English and continues in Japanese, everything after
the switch is transcribed as though it were English. Windowing forces a fresh
decision per window, which is also what lets a search hit say WHERE in a
two-hour recording it came from.

WHY THE OVERLAP, AND WHY IT IS CONDITIONAL. Cutting a fixed thirty seconds
lands mid-word, so consecutive windows overlap slightly and the duplicated
text is trimmed on rejoin. But a window boundary that falls on real silence
is NOT an artefact, and trimming across one deletes speech that was genuinely
said twice -- "Happy birthday to you" following "Happy birthday to you" is a
duplicate to a matcher and a birthday to everyone else. So each window records
whether it continues the previous one, and only those seams are trimmed.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass

# Thirty seconds is the window most speech models are trained on, and the unit
# their language detection operates over.
WINDOW_S = 30.0
# Enough to span a word cut in half, small enough that the duplicated text is
# cheap to find and trim.
OVERLAP_S = 2.0

# The longest suffix/prefix pair the rejoin will consider, in words. Beyond
# this a "match" is more likely to be a repeated phrase than a seam artefact.
_MAX_OVERLAP_WORDS = 12
# Below this, a match is too short to be evidence of anything -- two or three
# common words coincide constantly.
_MIN_OVERLAP_WORDS = 3


@dataclass(frozen=True)
class Window:
    """Where to transcribe, and whether it abuts the previous window.

    `continues_previous` is load-bearing on rejoin: it distinguishes a cut
    made mid-speech (whose duplicated text is an artefact) from a boundary
    that fell on real silence (whose repetition may be real).
    """

    start: float
    end: float
    continues_previous: bool = False

    @property
    def duration(self) -> float:
        return self.end - self.start


def fixed_windows(
    total_seconds: float,
    *,
    window_s: float = WINDOW_S,
    overlap_s: float = OVERLAP_S,
) -> Iterator[Window]:
    """Overlapping windows covering the whole recording.

    The fallback for audio that voice-activity detection could not segment.
    When a detector finds no speech region but the audio is not silent, the
    file is still transcribed in full and the text is judged afterwards --
    rather than being discarded on the word of a detector that demonstrably
    misses quiet, distant and reverberant recordings. On one archive, four of
    nine audited detector rejections were real family recordings.
    """
    if total_seconds <= 0:
        return
    start, first = 0.0, True
    step = window_s - overlap_s
    while start < total_seconds:
        yield Window(start, min(total_seconds, start + window_s), not first)
        first = False
        start += step


def pack_regions(
    regions: Sequence[tuple[float, float]], *, window_s: float = WINDOW_S
) -> list[tuple[float, float]]:
    """Merge consecutive speech regions while the merged span fits one window.

    A detector's region is often a single phrase, and one window per region
    sent the model one- and two-second clips: too little audio to tell the
    language from, and where it invents text. Measured on a real archive, 41%
    of windows under 2s came back in a language the rest of the recording was
    not in, against ~7% for windows of 10s or more. Packing restores the
    context while still skipping long silences: a gap that would push the
    span past `window_s` starts a new window. A region already longer than
    `window_s` is passed through for `windows_for_regions` to cut.
    """
    packed: list[tuple[float, float]] = []
    for start, end in regions:
        if packed and end - packed[-1][0] <= window_s:
            packed[-1] = (packed[-1][0], end)
        else:
            packed.append((start, end))
    return packed


def windows_for_regions(
    regions: Sequence[tuple[float, float]],
    *,
    window_s: float = WINDOW_S,
    overlap_s: float = OVERLAP_S,
) -> Iterator[Window]:
    """Windows covering detected speech regions, skipping the silence between.

    A speech region IS a natural window -- a detector found where talking
    starts and stops, which is what an amplitude-based silence pass was only
    approximating, and badly: music and wind are loud and are not speech.

    Regions longer than `window_s` are still cut, and ONLY those cuts overlap,
    because only they can land mid-word. Consecutive regions are separated by
    real silence, so their seam is not an artefact.
    """
    for start_s, end_s in regions:
        if end_s - start_s <= window_s:
            yield Window(start_s, end_s, False)
            continue
        start, first = start_s, True
        step = window_s - overlap_s
        while start < end_s:
            yield Window(start, min(end_s, start + window_s), not first)
            first = False
            start += step


def _key(word: str) -> str:
    return word.lower().strip(".,!?;:、。")


def join_trimming_overlap(previous: str, nxt: str) -> str:
    """Append `nxt` to `previous`, dropping what the overlap duplicated.

    Matches the longest suffix of `previous` that starts `nxt`, on word
    boundaries and ignoring case and trailing punctuation. Falls back to a
    plain join, and that direction is deliberate: a duplicated phrase is a far
    smaller error than a dropped one, so this never deletes on a guess.
    """
    if not previous:
        return nxt
    if not nxt:
        return previous
    previous_words, next_words = previous.split(), nxt.split()
    longest = min(_MAX_OVERLAP_WORDS, len(previous_words), len(next_words))
    for size in range(longest, _MIN_OVERLAP_WORDS - 1, -1):
        tail = [_key(w) for w in previous_words[-size:]]
        head = [_key(w) for w in next_words[:size]]
        if tail == head:
            return " ".join(previous_words + next_words[size:])
    return f"{previous} {nxt}".strip()


def join_windows(texts: Sequence[tuple[str, bool]]) -> str:
    """Rejoin window texts into one transcript.

    Each entry is `(text, continues_previous)`. A window that continues the
    previous one has its seam trimmed; one that follows real silence is joined
    with a plain space, because repetition across it may be genuine.
    """
    out = ""
    for text, continues in texts:
        stripped = text.strip()
        if not stripped:
            continue
        out = join_trimming_overlap(out, stripped) if continues else (
            f"{out} {stripped}".strip()
        )
    return out


def file_timeout(duration_s: float, *, base: float = 120.0) -> float:
    """How long one file may take before it is considered hung.

    Scaled by duration rather than fixed, because a flat timeout is wrong at
    both ends: too short for a two-hour recording, and long enough for a
    thirty-second clip to hang for the whole of it. One file stalled for 81
    minutes before this existed.

    `base` covers the fixed costs -- process start, model load, decode -- that
    do not scale with length.
    """
    return base + max(0.0, duration_s)
