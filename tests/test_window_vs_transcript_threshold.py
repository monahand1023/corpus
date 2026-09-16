"""One threshold was doing two jobs with opposite requirements.

`settings.max_looping_share` gated BOTH the per-window filter and the
whole-transcript verdict. Those need to differ, and not by a little:

  per-WINDOW    should be STRICT. A window is ~30 seconds. If most of it is
                repeated it is a loop, and dropping it costs one chunk while
                the rest of the recording survives -- that is the mechanism
                that catches a looping window inside genuine footage.
  per-TRANSCRIPT should be PERMISSIVE. Rejecting deletes the WHOLE recording,
                including whatever real speech is in it. A wrong call here
                deleted 149 real family recordings once already.

Raising the shared value from 0.6 to 0.85 to protect whole recordings
therefore loosened the window filter by the same amount, silently. Measured
on a live archive afterwards: 25 transcripts scoring 0.80-0.8444 are pure
Whisper decode loops -- the same Japanese sentence thirteen times over a
video of a child in a playroom -- sitting in the index, because at 0.85
neither check fires on them.
"""

from __future__ import annotations

from corpus.transcripts import quality
from corpus.transcripts.pipeline import Settings

# Scores 0.7895 -- squarely in the 0.75-0.80 band where a live archive's
# pure decode loops sit ("Storbritannia" x N, "そのため、" x N). Above the
# strict window threshold, below the permissive transcript one, which is the
# whole gap this file is about. A longer loop clears 0.85 on its own and
# would not exercise it.
PURE_LOOP = "こんばんは。ただいま準備しております。" * 5
REAL_WITH_LOOP_TAIL = (
    "¡Gracias! それはまだ新しいよね。 Hey, look at that over there, it is enormous. "
    "We should come back tomorrow when the light is better. " + "そのため、" * 30
)


def test_the_window_threshold_is_stricter_than_the_transcript_one():
    s = Settings()
    assert s.max_window_looping_share < s.max_looping_share, (
        "one value cannot serve both: strict per window, permissive per file"
    )


def test_a_window_that_is_nothing_but_a_loop_is_dropped():
    """At the shared 0.85 this scored 0.844 and was kept, becoming an
    indexed, searchable chunk of a model talking to itself."""
    from corpus.transcripts.pipeline import _window_is_junk as drop

    class _W:
        duration = 30.0

    assert drop(PURE_LOOP, _W(), Settings()) == "looping_repetition"


def test_the_whole_recording_is_still_kept_when_only_its_tail_loops():
    """The permissive side. The real speech at the front is why the
    transcript ceiling must not move with the window one."""
    verdict = quality.judge_transcript(
        REAL_WITH_LOOP_TAIL,
        duration_s=120.0,
        languages=["es", "en", "ja"],
        expected_languages=None,
    )
    assert verdict.keep, f"a real recording was deleted for its tail: {verdict.reason}"


def test_a_short_real_repeat_survives_the_window_filter():
    """The recording the archive exists for. A child repeating a word must
    not be dropped by the stricter window rule."""
    from corpus.transcripts.pipeline import _window_is_junk as drop

    class _W:
        duration = 4.0

    assert drop("Papa! Papa! Papa! Papa!", _W(), Settings()) is None


def test_the_window_threshold_reaches_the_policy_fingerprint():
    """A threshold the fingerprint cannot see is a threshold whose change
    does not invalidate the verdicts it produced."""
    assert "max_window_looping_share" in Settings().as_policy("m")
