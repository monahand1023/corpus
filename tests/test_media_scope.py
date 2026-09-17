"""An archive's own "which media is in scope" policy has to reach the walk.

one consumer excludes, among other things, media inside another person's iMessage
and WhatsApp backups -- "their voices would be searchable here forever, and
they never chose that". That policy lives in the ARCHIVE, as a predicate over
the full path, and the engine has no business knowing it.

`--exclude GLOB` is not a substitute, and the gap is not theoretical:
`fnmatch` is case-sensitive on POSIX, so `--exclude '*karaoke*'` does not
match `Karaoke Track.m4a`, while the archive's `pat in path.lower()` does.
Hand-translating a predicate into globs silently changes it, and this is the
one policy where a silent change means transcribing someone who never agreed.

So the consumer supplies the predicate itself, the way it already supplies
its connectors.
"""

from __future__ import annotations


def _tree(tmp_path):
    (tmp_path / "keep").mkdir()
    (tmp_path / "Messages" / "Attachments").mkdir(parents=True)
    (tmp_path / "keep" / "talk.m4a").write_bytes(b"")
    (tmp_path / "keep" / "Karaoke Track.m4a").write_bytes(b"")
    (tmp_path / "Messages" / "Attachments" / "voice.m4a").write_bytes(b"")
    return tmp_path


def test_a_consumers_predicate_decides_what_is_in_scope(tmp_path):
    from corpus.transcripts.run import find_media, media_scope

    def in_scope(path) -> bool:
        low = str(path).lower()
        return "karaoke" not in low and "/messages/attachments/" not in low

    with media_scope(in_scope):
        found = sorted(p.name for p in find_media(_tree(tmp_path)))

    assert found == ["talk.m4a"], found


def test_the_predicate_does_not_leak_out_of_its_block(tmp_path):
    """A ContextVar that is not reset makes the NEXT run quietly narrower --
    and a transcription pass that silently skips files is the failure this
    whole area keeps producing."""
    from corpus.transcripts.run import find_media, media_scope

    root = _tree(tmp_path)
    with media_scope(lambda p: False):
        assert list(find_media(root)) == []

    assert len(list(find_media(root))) == 3


def test_excluded_files_are_counted_not_silently_dropped(tmp_path):
    """The count is what makes "0 files" answerable afterwards."""
    from corpus.survey.walk import WalkStats
    from corpus.transcripts.run import find_media, media_scope

    stats = WalkStats()
    with media_scope(lambda p: "karaoke" not in str(p).lower()):
        list(find_media(_tree(tmp_path), stats=stats))

    assert stats.files_excluded >= 1, "an out-of-scope file left no trace"
