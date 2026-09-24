"""Tests for window arithmetic and rejoining.

This is the part of transcription with no model in it, so it is the part that
can be tested exhaustively -- and it is where the subtle errors live. Two of
the cases below are bugs that reached a real archive.
"""

from __future__ import annotations

from itertools import pairwise

from corpus.transcripts.segment import (
    OVERLAP_S,
    WINDOW_S,
    Window,
    file_timeout,
    fixed_windows,
    join_trimming_overlap,
    join_windows,
    windows_for_regions,
)

# --- fixed windows ---------------------------------------------------------


def test_fixed_windows_cover_the_whole_recording() -> None:
    windows = list(fixed_windows(100.0))
    assert windows[0].start == 0.0
    assert windows[-1].end == 100.0


def test_fixed_windows_leave_no_gap_between_consecutive_windows() -> None:
    # A gap here is speech nobody ever transcribes, and it would be invisible:
    # the run reports success either way.
    windows = list(fixed_windows(200.0))
    for previous, nxt in pairwise(windows):
        assert nxt.start <= previous.end, "a gap here silently drops speech"


def test_every_fixed_window_after_the_first_continues_the_previous() -> None:
    # Under fixed windows every seam is a cut mid-speech, so every seam's
    # duplicated text is an artefact worth trimming.
    windows = list(fixed_windows(200.0))
    assert windows[0].continues_previous is False
    assert all(w.continues_previous for w in windows[1:])


def test_a_recording_shorter_than_one_window_yields_exactly_one() -> None:
    windows = list(fixed_windows(12.0))
    assert len(windows) == 1
    assert windows[0] == Window(0.0, 12.0, False)


def test_an_empty_recording_yields_nothing() -> None:
    assert list(fixed_windows(0.0)) == []
    assert list(fixed_windows(-5.0)) == []


# --- windows from detected speech regions ----------------------------------


def test_a_short_region_becomes_one_window_with_no_overlap() -> None:
    windows = list(windows_for_regions([(10.0, 25.0)]))
    assert windows == [Window(10.0, 25.0, False)]


def test_a_long_region_is_split_and_only_those_cuts_overlap() -> None:
    windows = list(windows_for_regions([(0.0, 100.0)]))
    assert windows[0].continues_previous is False
    assert all(w.continues_previous for w in windows[1:])
    assert windows[-1].end == 100.0


def test_consecutive_regions_do_not_continue_each_other() -> None:
    # THE load-bearing case. Separate regions are separated by real silence,
    # so repetition across that boundary may be genuine -- "Happy birthday to
    # you" following "Happy birthday to you" is a duplicate to a matcher and a
    # birthday to everyone else.
    windows = list(windows_for_regions([(0.0, 10.0), (60.0, 70.0)]))
    assert [w.continues_previous for w in windows] == [False, False]


def test_splits_inside_a_region_cover_it_without_gaps() -> None:
    windows = list(windows_for_regions([(10.0, 100.0)]))
    for previous, nxt in pairwise(windows):
        assert nxt.start <= previous.end, "a gap here silently drops speech"


def test_silence_between_regions_is_never_transcribed() -> None:
    # The whole anti-hallucination strategy: a model invents caption
    # boilerplate from silence because its training data paired them, and no
    # threshold on the OUTPUT fixes that because the model is confidently
    # wrong. The defence is not handing it the silence.
    windows = list(windows_for_regions([(0.0, 5.0), (600.0, 605.0)]))
    assert not any(5.0 < w.start < 600.0 for w in windows)


def test_no_regions_means_no_windows() -> None:
    assert list(windows_for_regions([])) == []


# --- rejoining -------------------------------------------------------------


def test_a_duplicated_seam_is_trimmed() -> None:
    joined = join_trimming_overlap(
        "we walked down to the water and talked about the trip",
        "and talked about the trip for a while longer",
    )
    assert joined == (
        "we walked down to the water and talked about the trip for a while longer"
    )


def test_matching_ignores_case_and_trailing_punctuation() -> None:
    # The seam is found despite "day." vs "day," and differing case. Note what
    # survives: the text already accepted is kept VERBATIM and only the
    # duplicated prefix of the new window is dropped. Rewriting accepted text
    # to match a later window would be a second guess on top of the first.
    joined = join_trimming_overlap("it was a long day.", "It was a long day, really")
    assert joined == "it was a long day. really"


def test_a_short_coincidental_match_is_not_treated_as_a_seam() -> None:
    # Two or three common words coincide constantly; trimming on that would
    # delete real speech.
    joined = join_trimming_overlap("we went to the park", "the park was empty")
    assert joined == "we went to the park the park was empty"


def test_no_match_falls_back_to_a_plain_join() -> None:
    # The fallback direction is deliberate: a duplicated phrase is a far
    # smaller error than a dropped one, so this never deletes on a guess.
    joined = join_trimming_overlap("first half", "completely different text")
    assert joined == "first half completely different text"


def test_joining_onto_nothing_returns_the_new_text() -> None:
    assert join_trimming_overlap("", "opening words") == "opening words"


def test_joining_nothing_onto_text_leaves_it_alone() -> None:
    assert join_trimming_overlap("existing words", "") == "existing words"


# --- joining a whole recording ---------------------------------------------


def test_only_continuing_windows_have_their_seam_trimmed() -> None:
    trimmed = join_windows([
        ("happy birthday to you", False),
        ("happy birthday to you", True),
    ])
    assert trimmed == "happy birthday to you"

    kept = join_windows([
        ("happy birthday to you", False),
        ("happy birthday to you", False),
    ])
    assert kept == "happy birthday to you happy birthday to you", (
        "a boundary on real silence must not be trimmed -- the repetition is "
        "a birthday, not an artefact"
    )


def test_empty_windows_are_skipped_without_leaving_double_spaces() -> None:
    assert join_windows([("first", False), ("   ", True), ("second", False)]) == (
        "first second"
    )


def test_joining_no_windows_gives_empty_text() -> None:
    assert join_windows([]) == ""


# --- timeouts --------------------------------------------------------------


def test_the_timeout_scales_with_the_length_of_the_recording() -> None:
    # A flat timeout is wrong at both ends: too short for a two-hour recording,
    # and long enough for a thirty-second clip to hang for the whole of it. One
    # file stalled for 81 minutes before this existed.
    assert file_timeout(7200.0) > file_timeout(30.0)


def test_even_a_zero_length_file_gets_the_fixed_costs() -> None:
    # Process start, model load and decode do not scale with length.
    assert file_timeout(0.0) == 120.0
    assert file_timeout(-10.0) == 120.0


# --- the constants are the measured ones -----------------------------------


def test_the_window_and_overlap_are_the_measured_values() -> None:
    # 30s is the window most speech models are trained on and the unit their
    # language detection operates over; the overlap has to span a word cut in
    # half. Changing either invalidates every stored transcript, so they are
    # pinned rather than tuned casually.
    assert WINDOW_S == 30.0
    assert OVERLAP_S == 2.0
    assert OVERLAP_S < WINDOW_S, "a step of zero or less would never terminate"


# --- packing speech regions into windows ------------------------------------------


def test_nearby_regions_are_packed_into_one_window() -> None:
    """One window per detected region sent 1-2 second clips to the model, and
    a clip that short is where it guesses the language: 41% of windows under
    2s came back in a language the rest of the recording was not in, against
    ~7% for windows of 10s or more."""
    from corpus.transcripts.segment import pack_regions

    regions = [(0.0, 1.5), (2.0, 3.0), (5.0, 9.0), (40.0, 41.0)]
    assert pack_regions(regions, window_s=30.0) == [(0.0, 9.0), (40.0, 41.0)]


def test_packing_never_makes_a_window_longer_than_the_limit() -> None:
    from corpus.transcripts.segment import pack_regions

    regions = [(0.0, 10.0), (11.0, 21.0), (22.0, 32.0), (33.0, 43.0)]
    packed = pack_regions(regions, window_s=30.0)
    assert packed == [(0.0, 21.0), (22.0, 43.0)]
    assert all(end - start <= 30.0 for start, end in packed)


def test_a_region_longer_than_the_limit_is_left_for_windowing_to_cut() -> None:
    from corpus.transcripts.segment import pack_regions

    regions = [(0.0, 2.0), (3.0, 80.0), (81.0, 82.0)]
    assert pack_regions(regions, window_s=30.0) == [(0.0, 2.0), (3.0, 80.0), (81.0, 82.0)]


def test_packing_is_part_of_the_policy() -> None:
    """Changing how audio becomes windows changes the transcripts, so it must
    invalidate the stored ones: --redo-stale is how they get redone."""
    from corpus.transcripts.pipeline import Settings

    assert Settings().as_policy("m") != Settings(pack_regions=False).as_policy("m")
