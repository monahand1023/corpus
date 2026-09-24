"""An Audacity project's real layout, read from its binary-XML `project` row.

Without it the connector guessed 44100 Hz mono and joined every sample block
into one stream. Checked against three real projects: all three were stereo,
so every reported duration was about double, and one was 48 kHz, so its
duration was wrong again and extracted audio would have played too fast.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from corpus.connectors.aup3_layout import LayoutError, parse_binary_xml, read_layout
from tests.aup3_fixture import ClipSpec, TrackSpec, make_project, tone


def _layout(path: Path):
    conn = sqlite3.connect(path)
    try:
        return read_layout(conn)
    finally:
        conn.close()


# --- the binary XML --------------------------------------------------------------


def test_the_serialized_tree_round_trips(tmp_path: Path) -> None:
    path = make_project(tmp_path / "p.aup3", [TrackSpec([ClipSpec(tone(10, 0.1))], rate=22050)])
    conn = sqlite3.connect(path)
    dict_blob, doc_blob = conn.execute("SELECT dict, doc FROM project").fetchone()
    root = parse_binary_xml(dict_blob, doc_blob)

    assert root.tag == "project"
    assert root.attrs["audacityversion"] == "3.2.5"
    (track,) = root.find_all("wavetrack")
    assert track.attrs["rate"] == 22050
    assert track.attrs["mute"] is False


def test_a_truncated_document_is_an_error_not_a_partial_tree() -> None:
    from tests.aup3_fixture import _Encoder

    enc = _Encoder()
    enc.start("project")
    enc.start("wavetrack")
    with pytest.raises(LayoutError):
        parse_binary_xml(enc.dict_blob(), bytes(enc.doc))


def test_an_unknown_opcode_is_an_error() -> None:
    from tests.aup3_fixture import _Encoder

    enc = _Encoder()
    enc.start("project")
    enc.doc += bytes([0x7F])
    with pytest.raises(LayoutError):
        parse_binary_xml(enc.dict_blob(), bytes(enc.doc))


# --- the layout ------------------------------------------------------------------


def test_a_stereo_pair_is_one_recording_not_two_in_a_row(tmp_path: Path) -> None:
    samples = tone(8000, 0.2)  # one second at 8 kHz, per channel
    path = make_project(tmp_path / "p.aup3", [
        TrackSpec([ClipSpec(samples)], channel=0, linked=3),
        TrackSpec([ClipSpec(samples)], channel=1, linked=0),
    ])
    layout = _layout(path)

    assert layout.rate == 8000
    assert len(layout.tracks) == 2
    assert layout.duration_s == pytest.approx(1.0)


def test_the_real_sample_rate_is_used(tmp_path: Path) -> None:
    path = make_project(tmp_path / "p.aup3", [TrackSpec([ClipSpec(tone(48000, 0.1))], rate=48000)])
    assert _layout(path).duration_s == pytest.approx(1.0)


def test_clip_offsets_and_trims_set_the_span(tmp_path: Path) -> None:
    # 2s of audio placed at 1s, with 0.5s trimmed from each end: audible 1.5s..2.5s.
    path = make_project(tmp_path / "p.aup3", [
        TrackSpec([ClipSpec(tone(16000, 0.1), offset_s=1.0, trim_left_s=0.5, trim_right_s=0.5)]),
    ])
    layout = _layout(path)
    assert layout.start_s == pytest.approx(1.5)
    assert layout.duration_s == pytest.approx(1.0)


def test_a_project_with_no_project_row_has_no_layout(tmp_path: Path) -> None:
    path = make_project(tmp_path / "p.aup3", [TrackSpec([ClipSpec(tone(10, 0.1))])])
    conn = sqlite3.connect(path)
    conn.execute("DELETE FROM project")
    conn.commit()
    conn.close()
    assert _layout(path) is None


def test_a_block_the_document_names_but_the_database_lacks_is_an_error(tmp_path: Path) -> None:
    path = make_project(tmp_path / "p.aup3", [TrackSpec([ClipSpec(tone(3000, 0.1))])])
    conn = sqlite3.connect(path)
    conn.execute("DELETE FROM sampleblocks WHERE blockid = 2")
    conn.commit()
    conn.close()
    with pytest.raises(LayoutError, match="block"):
        _layout(path)


def test_a_track_at_a_different_rate_from_the_project_is_refused(tmp_path: Path) -> None:
    path = make_project(tmp_path / "p.aup3", [
        TrackSpec([ClipSpec(tone(8000, 0.1))], rate=8000),
        TrackSpec([ClipSpec(tone(16000, 0.1))], rate=16000),
    ], project_rate=8000)
    with pytest.raises(LayoutError, match="rate"):
        _layout(path)


# --- the mono mix ----------------------------------------------------------------

np = pytest.importorskip("numpy")


def _mix(path: Path):
    from corpus.connectors.aup3_layout import mix_to_mono

    conn = sqlite3.connect(path)
    try:
        return mix_to_mono(conn, read_layout(conn))
    finally:
        conn.close()


def test_a_stereo_pair_mixes_to_one_channel_of_the_same_length(tmp_path: Path) -> None:
    path = make_project(tmp_path / "p.aup3", [
        TrackSpec([ClipSpec(tone(8000, 0.2))], channel=0, linked=3),
        TrackSpec([ClipSpec(tone(8000, 0.4))], channel=1, linked=0),
    ])
    samples, rate = _mix(path)
    assert rate == 8000
    assert len(samples) == 8000
    assert samples.dtype == np.float32
    assert np.allclose(samples, 0.3)


def test_clips_land_at_their_offsets_with_silence_between(tmp_path: Path) -> None:
    path = make_project(tmp_path / "p.aup3", [
        TrackSpec([
            ClipSpec(tone(8000, 0.5), offset_s=0.0),
            ClipSpec(tone(8000, 0.5), offset_s=2.0),
        ]),
    ])
    samples, _ = _mix(path)
    assert len(samples) == 3 * 8000
    assert np.allclose(samples[:8000], 0.5)
    assert np.allclose(samples[8000:16000], 0.0)
    assert np.allclose(samples[16000:], 0.5)


def test_trimmed_audio_is_not_heard(tmp_path: Path) -> None:
    values = tone(4000, 0.1) + tone(8000, 0.9) + tone(4000, 0.1)
    path = make_project(tmp_path / "p.aup3", [
        TrackSpec([ClipSpec(values, trim_left_s=0.5, trim_right_s=0.5)]),
    ])
    samples, _ = _mix(path)
    assert len(samples) == 8000
    assert np.allclose(samples, 0.9)


def test_a_muted_track_is_left_out(tmp_path: Path) -> None:
    path = make_project(tmp_path / "p.aup3", [
        TrackSpec([ClipSpec(tone(8000, 0.4))]),
        TrackSpec([ClipSpec(tone(8000, 0.9))], mute=True),
    ])
    samples, _ = _mix(path)
    assert np.allclose(samples, 0.4)


def test_a_block_that_fails_its_own_checksum_is_refused(tmp_path: Path) -> None:
    path = make_project(tmp_path / "p.aup3", [TrackSpec([ClipSpec(tone(3000, 0.1))])])
    conn = sqlite3.connect(path)
    conn.execute("UPDATE sampleblocks SET summax = 0.8 WHERE blockid = 2")
    conn.commit()
    conn.close()
    with pytest.raises(LayoutError, match="block 2"):
        _mix(path)
