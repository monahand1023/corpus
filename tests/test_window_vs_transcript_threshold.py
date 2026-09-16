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

    assert drop(PURE_LOOP, 30.0, Settings()) == "looping_repetition"


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

    assert drop("Papa! Papa! Papa! Papa!", 4.0, Settings()) is None


def test_the_window_threshold_reaches_the_policy_fingerprint():
    """A threshold the fingerprint cannot see is a threshold whose change
    does not invalidate the verdicts it produced."""
    assert "max_window_looping_share" in Settings().as_policy("m")


# --- "a dropped window costs one chunk" is false when there is only one -------


def test_a_single_window_recording_is_not_deleted_by_the_window_filter():
    """The argument for a STRICT window threshold was that dropping a window
    costs one chunk while the rest of the recording survives. On a short
    recording there is no rest: the file is one window, so dropping it IS
    deleting the recording -- the precise failure the permissive transcript
    ceiling exists to prevent.

    Measured before this was caught, on a full transcript archive: 176
    recordings would have been emptied, 174 of them single-window, including

        "a birthday line, then another.."   (0.647)
        "an exclamation, three times"
        "a greeting, then a chant"

    -- the exact material the archive exists for, and the same recordings a
    0.6 whole-transcript ceiling deleted earlier the same day.
    """
    from corpus.transcripts.pipeline import filter_windows

    song = "Happy Birthday to you. " * 8
    kept, dropped = filter_windows([(song, 30.0, False)], Settings())
    assert [t for t, _ in kept] == [song.strip()], (
        f"the only window was dropped: {dropped}"
    )


def test_a_looping_window_inside_a_real_recording_is_still_dropped():
    """The permissive side must not swallow the strict one. Real speech
    either side means the loop can go and the recording survives."""
    from corpus.transcripts.pipeline import filter_windows

    real_a = "We walked down to the harbour and the boats were all out."
    loop = "そのため、" * 30
    real_b = "Then we found a bakery and sat outside for an hour."
    kept, dropped = filter_windows(
        [(real_a, 30.0, False), (loop, 30.0, True), (real_b, 30.0, True)],
        Settings(),
    )
    assert [t for t, _ in kept] == [real_a, real_b]
    assert dropped == [(1, "looping_repetition")]


def test_windows_that_are_all_boilerplate_are_still_dropped_entirely():
    """The fallback is for REPETITION only. "Thank you. Thank you." is not a
    song the archive's owner wants -- it is the model filling silence, and
    reinstating it would put an empty result back into search."""
    from corpus.transcripts.pipeline import filter_windows

    kept, dropped = filter_windows(
        [("Thank you.", 20.0, False), ("Thank you.", 20.0, True)], Settings()
    )
    assert kept == []
    assert {r for _, r in dropped} == {"caption_boilerplate"}
    assert [i for i, _ in dropped] == [0, 1]


def test_an_all_looping_multi_window_file_also_falls_back():
    """Not a single-window special case. Three looping windows and nothing
    else is still a recording whose fate the permissive whole-transcript
    ceiling should decide, not the strict per-window one."""
    from corpus.transcripts.pipeline import filter_windows

    chant = "Papa Papa Papa Papa Papa Papa Papa Papa Papa Papa "
    windows = [(chant, 30.0, i > 0) for i in range(3)]
    kept, _ = filter_windows(windows, Settings())
    assert len(kept) == 3, "a chant across three windows was deleted"


# Verbatim from a live archive: a 64-second video whose ENTIRE transcript is
# this, produced from a 1.7-second window. 235 characters in 1.7 seconds is
# 138 per second, against a ceiling of 40 -- 3.4x past the point where a human
# throat is the explanation. It is indexed, searchable content today.
INDEXED_DECODE_LOOP = "ta lom, e o lomi, a stl, e4,8,1" + ",0" * 100


def test_a_reinstated_window_still_faces_the_checks_it_never_reached():
    """The fallback granted amnesty from a check that had not run.

    `_window_is_junk` returns at the FIRST rule that matches. This window
    matched `looping_repetition` (0.843 >= the strict 0.6), so the rate check
    below it never executed. The fallback then reinstated the window -- and
    reinstatement is not re-judgement, so the never-run check was treated
    exactly like a check that had passed.

    The whole-transcript judge then saw the same text against the FILE's 64
    seconds instead of the window's 1.7: 3.7 chars/s, comfortably under the
    ceiling, and 0.843 looping just under the permissive 0.85. All three
    defences declined, and the loop was indexed.

    The fallback exists so a strict REPETITION threshold cannot delete a short
    real recording -- a child chanting one word is repetitive and real. It was
    never meant to excuse text that could not physically have been spoken.
    """
    from corpus.transcripts.pipeline import filter_windows

    kept, dropped = filter_windows([(INDEXED_DECODE_LOOP, 1.7, False)], Settings())

    assert kept == [], f"a decode loop at 138 chars/s was reinstated: {kept}"
    assert dropped == [(0, "impossible_speech_rate")], (
        "dropped for the wrong reason -- the honest one is the physical rate, "
        f"and it is what the dormancy report reads: {dropped}"
    )


def test_the_rate_backstop_does_not_touch_a_real_short_recording():
    """The fallback's whole purpose, unchanged. A child repeating one word at
    a human rate is repetitive, physically possible, and the reason this
    archive exists."""
    from corpus.transcripts.pipeline import filter_windows

    chant = "Daddy! Daddy! Daddy! Daddy! Daddy! "
    kept, dropped = filter_windows([(chant, 12.0, False)], Settings())

    assert [t for t, _ in kept] == [chant.strip()], f"dropped a real clip: {dropped}"
