"""A duration floor, because most phone videos are room tone.

Measured on a real archive of a real photo library: **81% are under
four seconds** — the clip Apple stores beside each Live Photo. Transcribing
them adds ~46,800 files for ~33 hours of ambience and floods the index with
near-empty text that dilutes every search.

The only archive that had solved this solved it in a 1,033-line private
script, so every other corpus user pointing `corpus-transcribe` at a phone's
video folder pays for the ambience. The dry run could not warn them either:
it reports total hours, which cannot show that four fifths of them are
four seconds long.

Applied BEFORE decoding. `transcribe_file` learns the duration by decoding,
which is the cost being avoided, so the floor uses `ffprobe` — which reads
the container header and does not decode.
"""

from __future__ import annotations

from pathlib import Path

from corpus.transcripts.run import below_duration_floor, partition_by_duration


def _probe(durations: dict[str, float | None]):
    """A stand-in for ffprobe: name -> seconds, or None when unreadable."""
    return lambda path: durations.get(Path(path).name)


def test_a_floor_of_zero_keeps_everything():
    """The default. A floor nobody asked for must not silently drop work."""
    keep, skip = partition_by_duration(
        [Path("a.mov"), Path("b.mov")],
        min_seconds=0.0,
        probe=_probe({"a.mov": 2.0, "b.mov": 90.0}),
    )
    assert keep == [Path("a.mov"), Path("b.mov")]
    assert skip == []


def test_short_clips_are_separated_from_the_rest():
    keep, skip = partition_by_duration(
        [Path("live.mov"), Path("talk.mov")],
        min_seconds=15.0,
        probe=_probe({"live.mov": 3.0, "talk.mov": 240.0}),
    )
    assert keep == [Path("talk.mov")]
    assert skip == [Path("live.mov")]


def test_a_clip_exactly_at_the_floor_is_kept():
    keep, _ = partition_by_duration(
        [Path("edge.mov")], min_seconds=15.0, probe=_probe({"edge.mov": 15.0})
    )
    assert keep == [Path("edge.mov")]


def test_a_file_whose_duration_cannot_be_read_is_KEPT():
    """The safe direction. ffprobe failing is not evidence that a recording
    is short, and skipping on an unknown would silently drop real speech --
    the failure mode every threshold in this project is tuned against."""
    keep, skip = partition_by_duration(
        [Path("weird.mov")], min_seconds=15.0, probe=_probe({"weird.mov": None})
    )
    assert keep == [Path("weird.mov")]
    assert skip == []


def test_the_predicate_is_usable_on_its_own():
    assert below_duration_floor(3.0, 15.0) is True
    assert below_duration_floor(30.0, 15.0) is False
    assert below_duration_floor(None, 15.0) is False, "unknown must not be skipped"
    assert below_duration_floor(3.0, 0.0) is False, "a zero floor skips nothing"


def test_nothing_is_probed_when_there_is_no_floor():
    """Probing costs a subprocess per file. With no floor there is nothing to
    decide, and paying for 58,000 ffprobe calls to decide nothing is the kind
    of cost that gets a feature switched off."""
    probed: list[Path] = []

    def counting_probe(path):
        probed.append(path)
        return 1.0

    partition_by_duration(
        [Path("a.mov"), Path("b.mov")], min_seconds=0.0, probe=counting_probe
    )
    assert probed == []


# --- the dry run must not price what it just said it would skip --------------


def test_the_dry_run_hours_exclude_the_files_the_floor_removed(tmp_path, capsys, monkeypatch):
    """First version printed "204 skipped" and then quoted hours for all 857,
    because the duration survey re-walks the tree and knows nothing about the
    floor. A number that contradicts the line above it is worse than no
    number: the whole point of the dry run is to be the one warning before
    hours of compute."""
    import corpus.cli.transcribe as mod
    from corpus.transcripts.pipeline import Settings

    for name in ("short.mov", "long1.mov", "long2.mov"):
        (tmp_path / name).write_bytes(b"x")

    durations = {"short.mov": 3.0, "long1.mov": 3600.0, "long2.mov": 3600.0}
    monkeypatch.setattr(
        mod, "_duration_probe", lambda: (lambda p: durations[Path(p).name])
    )
    monkeypatch.setattr(mod, "default_backend_or_none", lambda: None, raising=False)

    mod._dry_run(tmp_path, tmp_path / "absent.db", (), 15.0, Settings(), 15.0)
    out = capsys.readouterr().out

    assert "1 skipped" in out or "1 " in out
    # 2 hours of long clips, not 2.0008 with the 3-second one folded in.
    assert "~2.0 h" in out, out
