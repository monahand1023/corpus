"""A transcription walk skipped 58,024 home videos and said nothing.

`.photoslibrary` is excluded from every walk by default, and that default is
correct for INDEXING: the bundle's contents are Apple's internal SQLite,
plists and derivative images, not documents.

It is exactly wrong for TRANSCRIPTION, where `originals/` holds the home
video. A reference archive measured ~8,100 clips of 15 seconds or more in
there -- "where birthdays and family history live".

The bug is not the default. It is that pointing `corpus-transcribe` at a
folder containing a photo library returns "0 files" -- identical, to the
person reading it, to a folder that genuinely has no recordings in it. One
means "nothing to do" and the other means "I refused to look", and the
archive that depends on knowing the difference is the one that has family
video in it.

Pointing the root AT the bundle already works. Nothing said so.
"""

from __future__ import annotations

from pathlib import Path


def _tree(tmp_path: Path) -> Path:
    lib = tmp_path / "Pictures" / "Family Library.photoslibrary" / "originals" / "9"
    lib.mkdir(parents=True)
    (lib / "birthday.mov").write_bytes(b"")
    plain = tmp_path / "Pictures" / "Camera"
    plain.mkdir(parents=True)
    (plain / "talk.mp4").write_bytes(b"")
    return tmp_path


def test_the_walk_names_the_photo_library_it_pruned(tmp_path):
    from corpus.survey.walk import WalkStats
    from corpus.transcripts.run import find_media

    stats = WalkStats()
    found = [p.name for p in find_media(_tree(tmp_path), stats=stats)]

    assert found == ["talk.mp4"], "the default exclusion itself should not change"
    assert stats.media_bundles_pruned, (
        "the walk skipped a photo library and recorded nothing about it"
    )
    assert "Family Library.photoslibrary" in stats.media_bundles_pruned[0]


def test_an_ordinary_walk_reports_no_bundles(tmp_path):
    """The counter must stay empty when nothing was withheld -- otherwise the
    hint fires on every run and stops being read."""
    from corpus.survey.walk import WalkStats
    from corpus.transcripts.run import find_media

    plain = tmp_path / "Camera"
    plain.mkdir()
    (plain / "a.mov").write_bytes(b"")

    stats = WalkStats()
    assert [p.name for p in find_media(tmp_path, stats=stats)] == ["a.mov"]
    assert stats.media_bundles_pruned == []


def test_the_dry_run_tells_you_how_to_reach_the_recordings(tmp_path, capsys):
    """Reporting the skip is only half of it. The person needs the next
    command, and the answer -- point the root at the bundle -- is not
    something anyone should have to derive from the exclusion rules."""
    import sys

    from corpus.cli import transcribe as cli

    root = _tree(tmp_path)
    argv = ["corpus-transcribe", str(root), "--dry-run", "--db", str(tmp_path / "t.db")]
    old = sys.argv
    sys.argv = argv
    try:
        cli.main()
    finally:
        sys.argv = old
    out = capsys.readouterr().out

    assert "photoslibrary" in out.lower(), f"the skip was never mentioned:\n{out}"
    assert "Family Library.photoslibrary" in out, (
        f"the bundle was not named, so there is nothing to point at:\n{out}"
    )
