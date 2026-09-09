"""Tests for `corpus.connectors.music`.

Fixtures are real files written by mutagen, not mocks: the whole difficulty
of this connector is that the same logical field has three unrelated
spellings across ID3, MP4 and Vorbis, and a mock would let a wrong spelling
pass.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from corpus.connectors.music import (
    MUSIC_EXTENSIONS,
    MusicConnector,
    _commonest,
    _track_number,
)

pytest.importorskip("mutagen")


def _mp3(path: Path, **tags: str) -> Path:
    """A minimal but genuinely valid MP3 with ID3 tags."""
    import mutagen.id3
    from mutagen.mp3 import MP3

    path.parent.mkdir(parents=True, exist_ok=True)
    # Four silent MPEG-1 Layer III frames (128 kbps, 44.1 kHz => 417 bytes
    # each). mutagen needs several consecutive valid headers before it will
    # accept a stream, so a single frame is not enough.
    path.write_bytes((b"\xff\xfb\x90\x00" + b"\x00" * 413) * 4)
    audio = MP3(str(path))
    audio.add_tags()
    frames = {
        "album": mutagen.id3.TALB,
        "artist": mutagen.id3.TPE1,
        "album_artist": mutagen.id3.TPE2,
        "title": mutagen.id3.TIT2,
        "track": mutagen.id3.TRCK,
        "genre": mutagen.id3.TCON,
        "date": mutagen.id3.TDRC,
    }
    for field, value in tags.items():
        audio.tags.add(frames[field](encoding=3, text=[value]))
    audio.save()
    return path


def _load(root: Path) -> list:
    return list(MusicConnector(source_type="music", path=root).load())


# --- helpers ----------------------------------------------------------------


def test_track_number_handles_the_shapes_tags_actually_use() -> None:
    assert _track_number("7") == 7
    assert _track_number("7/12") == 7  # "of N" suffix
    assert _track_number("07") == 7
    assert _track_number("") == 0
    assert _track_number("A-side") == 0


def test_commonest_wins_so_one_bad_tag_cannot_rename_an_album() -> None:
    # Half an album tagged "The Beatles" and half "Beatles, The" is normal.
    assert _commonest(["The Beatles", "The Beatles", "Beatles, The"]) == "The Beatles"
    assert _commonest(["", "", "Real"]) == "Real"
    assert _commonest(["", ""]) == ""


# --- the album document -----------------------------------------------------


def test_one_document_per_album_not_per_track(tmp_path: Path) -> None:
    # A track is title + artist + number: too thin to retrieve on, and 4,000
    # of them would be near-identical chunks competing with each other.
    album = tmp_path / "Pet Shop Boys" / "Please"
    for i, title in enumerate(["West End Girls", "Opportunities", "Suburbia"], start=1):
        _mp3(album / f"{i:02d}.mp3", album="Please", artist="Pet Shop Boys",
             title=title, track=str(i))

    docs = _load(tmp_path)

    assert len(docs) == 1
    assert docs[0].title == "Pet Shop Boys — Please"


def test_the_track_listing_is_in_the_body(tmp_path: Path) -> None:
    album = tmp_path / "a" / "b"
    _mp3(album / "1.mp3", album="Please", artist="Pet Shop Boys",
         title="West End Girls", track="1")
    _mp3(album / "2.mp3", album="Please", artist="Pet Shop Boys",
         title="Suburbia", track="2")

    body = _load(tmp_path)[0].raw["body"]

    assert "West End Girls" in body
    assert "Suburbia" in body
    assert "2 track(s)" in body


def test_tracks_are_ordered_by_track_number_not_filename(tmp_path: Path) -> None:
    album = tmp_path / "a" / "b"
    _mp3(album / "zzz.mp3", album="X", artist="Y", title="First", track="1")
    _mp3(album / "aaa.mp3", album="X", artist="Y", title="Second", track="2")

    body = _load(tmp_path)[0].raw["body"]

    assert body.index("First") < body.index("Second")


def test_year_and_genre_are_carried(tmp_path: Path) -> None:
    album = tmp_path / "a" / "b"
    _mp3(album / "1.mp3", album="Please", artist="PSB", title="T",
         track="1", date="1986", genre="Synth-pop")

    doc = _load(tmp_path)[0]

    assert "1986" in doc.raw["body"]
    assert "Synth-pop" in doc.raw["body"]
    assert doc.created_at == "1986-01-01"


def test_album_artist_beats_per_track_artist(tmp_path: Path) -> None:
    # On a compilation every track has a different artist; the album still
    # belongs to whoever the album artist says.
    album = tmp_path / "a" / "b"
    _mp3(album / "1.mp3", album="Now 42", album_artist="Various Artists",
         artist="Band One", title="Song One", track="1")
    _mp3(album / "2.mp3", album="Now 42", album_artist="Various Artists",
         artist="Band Two", title="Song Two", track="2")

    doc = _load(tmp_path)[0]

    assert doc.title == "Various Artists — Now 42"
    # A differing per-track artist is what makes a compilation legible.
    assert "Band One" in doc.raw["body"]
    assert "Band Two" in doc.raw["body"]


def test_a_mistagged_track_does_not_split_the_album(tmp_path: Path) -> None:
    album = tmp_path / "a" / "b"
    _mp3(album / "1.mp3", album="Please", artist="Pet Shop Boys", title="A", track="1")
    _mp3(album / "2.mp3", album="Please", artist="Pet Shop Boys", title="B", track="2")
    _mp3(album / "3.mp3", album="Pleaes", artist="Pet Shop Boys", title="C", track="3")

    docs = _load(tmp_path)

    assert len(docs) == 1
    assert "Please" in docs[0].title


def test_separate_directories_are_separate_albums(tmp_path: Path) -> None:
    _mp3(tmp_path / "artist" / "one" / "1.mp3", album="One", artist="A", title="T")
    _mp3(tmp_path / "artist" / "two" / "1.mp3", album="Two", artist="A", title="T")

    assert len(_load(tmp_path)) == 2


# --- what is NOT music ------------------------------------------------------


def test_untagged_audio_is_skipped_not_emitted(tmp_path: Path) -> None:
    # Measured on a real library: `.m4a` files included VoiceMemos captures
    # with no album or artist tag. Emitting "Unknown Album" for those would be
    # unsearchable AND misleading about what the library holds.
    _mp3(tmp_path / "recordings" / "memo.mp3")
    connector = MusicConnector(source_type="music", path=tmp_path)

    docs = list(connector.load())

    assert docs == []
    assert connector.skipped_files == 1
    assert connector.failed_files == 0


def test_a_title_alone_is_not_enough_to_be_an_album(tmp_path: Path) -> None:
    # A voice memo often carries a title and nothing else.
    _mp3(tmp_path / "recordings" / "memo.mp3", title="Meeting with the bank")

    assert _load(tmp_path) == []


def test_an_album_tag_alone_is_enough(tmp_path: Path) -> None:
    _mp3(tmp_path / "a" / "b" / "1.mp3", album="Untitled Demo", title="T")

    docs = _load(tmp_path)

    assert len(docs) == 1
    assert "Untitled Demo" in docs[0].title


def test_non_music_extensions_are_ignored(tmp_path: Path) -> None:
    # The glob sweep is per-extension precisely so a folder of notes is not
    # detected as a music source.
    (tmp_path / "notes.md").write_text("# Not music")
    (tmp_path / "readme.markdown").write_text("# Also not music")

    assert _load(tmp_path) == []


def test_every_swept_extension_is_lowercase_and_dotted() -> None:
    # `path.suffix.lower()` is matched against this tuple, so an entry that
    # is uppercase or missing its dot silently never matches.
    for ext in MUSIC_EXTENSIONS:
        assert ext.startswith(".")
        assert ext == ext.lower()


# --- connector contract -----------------------------------------------------


def test_missing_directory_raises(tmp_path: Path) -> None:
    # Must raise, not yield nothing: an empty enumeration makes the ingester
    # treat every indexed album as an orphan and delete it.
    connector = MusicConnector(source_type="music", path=tmp_path / "nope")

    with pytest.raises(FileNotFoundError):
        list(connector.load())


def test_source_key_is_the_album_directory(tmp_path: Path) -> None:
    _mp3(tmp_path / "Pet Shop Boys" / "Please" / "1.mp3",
         album="Please", artist="Pet Shop Boys", title="T")

    assert _load(tmp_path)[0].source_key == "Pet Shop Boys/Please"


def test_counters_reset_between_runs(tmp_path: Path) -> None:
    _mp3(tmp_path / "recordings" / "memo.mp3")
    connector = MusicConnector(source_type="music", path=tmp_path)

    list(connector.load())
    first = connector.skipped_files
    list(connector.load())

    assert connector.skipped_files == first


def test_a_tag_lookup_that_raises_does_not_abort_the_field() -> None:
    # Every container is probed with every spelling, so most lookups ask for
    # a key that container has never heard of — and they disagree about what
    # that means. Vorbis (FLAC/OGG) raises a BARE ValueError, no message, for
    # a key outside its legal ASCII range, which is exactly what the MP4
    # atoms in the field table are. Unguarded, one FLAC file aborted an
    # entire real source and the CLI printed "ERROR:" with nothing after it.
    class _Picky:
        def get(self, key: str):
            if not key.isascii():
                raise ValueError  # bare, exactly as mutagen raises it
            return "Kid A" if key == "album" else None

    from corpus.connectors.music import _tag

    assert _tag(_Picky(), "album") == "Kid A"


def test_a_tag_container_that_raises_for_everything_yields_empty() -> None:
    class _Broken:
        def get(self, key: str):
            raise RuntimeError("container is unreadable")

    from corpus.connectors.music import _tag

    assert _tag(_Broken(), "album") == ""


def test_a_mixed_playlist_folder_is_titled_by_its_directory(tmp_path: Path) -> None:
    # An event set, a singles rip, a playlist — most tracks disagree about
    # the album, and naming the folder after whichever tag was least rare
    # gives a title nobody will search for. Measured on a real library, one
    # such folder inherited a mojibake album tag from a single corrupt file
    # and was titled with it.
    folder = tmp_path / "Playlists" / "Party Mix"
    # Four tracks, four different albums: the commonest tag covers a quarter
    # of the folder. (Exactly half is deliberately NOT mixed — a two-disc set
    # tagged "(Disc 1)"/"(Disc 2)" splits that way and is still one record.)
    for i, album in enumerate(["Album A", "Album B", "Album C", "Album D"], start=1):
        _mp3(folder / f"{i}.mp3", album=album, artist=f"Artist {i}",
             title=f"Song {i}", track=str(i))

    doc = _load(tmp_path)[0]

    assert "Party Mix" in doc.title
    assert "Mixed folder" in doc.raw["body"]


def test_a_real_album_is_not_treated_as_mixed(tmp_path: Path) -> None:
    folder = tmp_path / "a" / "b"
    for i in range(1, 5):
        _mp3(folder / f"{i}.mp3", album="Please", artist="Pet Shop Boys",
             title=f"Song {i}", track=str(i))

    doc = _load(tmp_path)[0]

    assert doc.title == "Pet Shop Boys — Please"
    assert "Mixed folder" not in doc.raw["body"]


def test_a_two_track_folder_is_never_called_mixed(tmp_path: Path) -> None:
    # A single with a B-side legitimately carries two different album tags.
    folder = tmp_path / "a" / "b"
    _mp3(folder / "1.mp3", album="Single A", artist="X", title="A side", track="1")
    _mp3(folder / "2.mp3", album="Single B", artist="X", title="B side", track="2")

    assert "Mixed folder" not in _load(tmp_path)[0].raw["body"]
