"""Music library connector — one document per album, from tags only.

Answers "what albums do I have", "did I ever own that record", "what else is
by this artist". It reads tags and never touches audio: no transcription, no
analysis, no lyrics. A song's audio is not searchable text and pretending
otherwise would cost hours of compute for nothing.

**One document per album directory, not per track.** A track yields a title,
an artist, and a number — three fields that make a document too thin to
retrieve on and would turn a 4,000-file library into 4,000 near-identical
chunks competing with each other. An album is the unit people actually
remember and ask about, and its track listing gives the document enough text
to match against.

The album's identity comes from the DIRECTORY, not from the tags. Music is
near-universally laid out `Artist/Album/track.mp3`, the directory is what
`source_key` needs anyway (stable, unique, a real path), and tags disagree
with each other constantly — half an album tagged "The Beatles" and half
"Beatles, The" would otherwise split into two albums. Tag values are read
from the files inside and the commonest value wins, so one mistagged track
does not rename the record.

**Not everything with an audio extension is music.** A real library mixes in
voice memos, interview recordings, and app exports that share the same
container: measured on one library, `.m4a` files included VoiceMemos captures
with no album or artist tag at all. Those are transcription material, not
catalogue entries, and a directory whose files carry no album/artist tags is
skipped rather than emitted as an "Unknown Album" document — which would be
both useless to search and actively misleading about what the library holds.

Install: `pip install corpus-rag[music]` or `uv add mutagen`.
"""

from __future__ import annotations

import logging
import os
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from corpus.connectors.discovery import discover_files
from corpus.types import SourceDocument

logger = logging.getLogger(__name__)

# Extensions this connector claims. Kept explicit rather than "whatever
# mutagen opens", because mutagen also opens the voice memos and video
# soundtracks that share these containers.
#
# Swept one at a time rather than expressed as a single glob because
# `Path.glob` has no alternation: any single pattern wide enough to catch
# `.mp3` and `.flac` also catches `.md` and `.markdown`, which made
# `corpus-ingest --path` detect a "music" source in a folder of notes.
MUSIC_EXTENSIONS: tuple[str, ...] = (
    ".mp3", ".m4a", ".m4p", ".flac", ".ogg", ".oga", ".wma", ".aac",
)

# Tag names per container, in the order they should be consulted. ID3 uses
# four-character frame ids, MP4 uses iTunes atoms, Vorbis/FLAC uses plain
# words — the same field has three unrelated spellings, so every lookup goes
# through this table.
_FIELDS: dict[str, tuple[str, ...]] = {
    "album": ("TALB", "\xa9alb", "album"),
    "album_artist": ("TPE2", "aART", "albumartist"),
    "artist": ("TPE1", "\xa9ART", "artist"),
    "title": ("TIT2", "\xa9nam", "title"),
    "date": ("TDRC", "TYER", "\xa9day", "date", "originaldate"),
    "genre": ("TCON", "\xa9gen", "genre"),
    "track": ("TRCK", "trkn", "tracknumber"),
}


def _tag(tags: Any, field: str) -> str:
    """First non-empty value for a logical field, whatever the container calls it."""
    if not tags:
        return ""
    for key in _FIELDS[field]:
        try:
            value = tags.get(key)
        except (TypeError, KeyError):
            continue
        if value is None:
            continue
        if isinstance(value, list) and value:
            value = value[0]
        # MP4 track numbers arrive as a (number, total) tuple.
        if isinstance(value, tuple) and value:
            value = value[0]
        text = str(value).strip()
        if text and text != "None":
            return text
    return ""


def _track_number(raw: str) -> int:
    """Leading integer of a track tag, or 0. Handles "7", "7/12", "07"."""
    digits = ""
    for ch in raw.strip():
        if ch.isdigit():
            digits += ch
        else:
            break
    return int(digits) if digits else 0


def _duration(seconds: float) -> str:
    minutes, secs = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"


def _commonest(values: Iterable[str]) -> str:
    """Most frequent non-empty value, so one mistagged track cannot rename an album."""
    counts = Counter(v for v in values if v)
    return counts.most_common(1)[0][0] if counts else ""


class MusicConnector:
    def __init__(
        self,
        source_type: str,
        path: Path | str,
        glob: str | None = None,
    ):
        self.source_type = source_type
        self._root = Path(os.path.expanduser(str(path))).resolve()
        # None means "sweep every known music extension", which is the useful
        # default. An explicit glob narrows to exactly one pattern, for a
        # library where that is what someone wants.
        self._globs = [glob] if glob else [f"**/*{ext}" for ext in MUSIC_EXTENSIONS]
        self.failed_files = 0
        self.skipped_files = 0

    def load(self) -> Iterable[SourceDocument]:
        # Reset per run so a stale count cannot suppress orphan pruning
        # forever. Same contract as every other file connector.
        self.failed_files = 0
        self.skipped_files = 0

        if not self._root.is_dir():
            raise FileNotFoundError(
                f"Music source '{self.source_type}': directory not found: {self._root}"
            )

        by_directory: dict[Path, list[dict[str, Any]]] = defaultdict(list)
        seen: set[Path] = set()
        for pattern in self._globs:
            for path in discover_files(self._root, pattern):
                # A case-insensitive filesystem can return the same file for
                # more than one pattern; a track counted twice would double an
                # album's length and duplicate its listing.
                if path in seen or path.suffix.lower() not in MUSIC_EXTENSIONS:
                    continue
                seen.add(path)
                self._collect(path, by_directory)

        for directory, tracks in sorted(by_directory.items()):
            document = self._album_document(directory, tracks)
            if document is not None:
                yield document

    def _collect(self, path: Path, by_directory: dict[Path, list[dict[str, Any]]]) -> None:
        import mutagen

        try:
            audio = mutagen.File(str(path))
        except Exception as e:  # mutagen raises many types for bad files
            logger.warning(
                "Music source '%s': cannot read %s: %s", self.source_type, path, e
            )
            self.failed_files += 1
            return
        if audio is None:
            # Not a container mutagen recognizes. Permanent, so skipped.
            self.skipped_files += 1
            return
        tags = audio.tags
        by_directory[path.parent].append(
            {
                "path": path,
                "album": _tag(tags, "album"),
                "album_artist": _tag(tags, "album_artist"),
                "artist": _tag(tags, "artist"),
                "title": _tag(tags, "title"),
                "date": _tag(tags, "date"),
                "genre": _tag(tags, "genre"),
                "track": _track_number(_tag(tags, "track")),
                "seconds": float(getattr(audio.info, "length", 0.0) or 0.0),
            }
        )

    def _album_document(
        self, directory: Path, tracks: list[dict[str, Any]]
    ) -> SourceDocument | None:
        album = _commonest(t["album"] for t in tracks)
        artist = _commonest(t["album_artist"] for t in tracks) or _commonest(
            t["artist"] for t in tracks
        )
        if not album and not artist:
            # Voice memos, interview captures, app exports — audio that shares
            # a container with music but is not a catalogue entry. Emitting an
            # "Unknown Album" document here would be unsearchable AND actively
            # misleading about what the library contains. Permanent, so
            # skipped rather than failed.
            self.skipped_files += len(tracks)
            logger.info(
                "%s: skipping '%s' — %d audio file(s) with no album or artist tag",
                self.source_type,
                directory.name,
                len(tracks),
            )
            return None

        tracks.sort(key=lambda t: (t["track"], t["path"].name))
        year = _commonest(t["date"] for t in tracks)[:4]
        genre = _commonest(t["genre"] for t in tracks)
        total = sum(t["seconds"] for t in tracks)

        header = [f"{artist} — {album}" if artist and album else (album or artist)]
        if year:
            header.append(f"Year: {year}")
        if genre:
            header.append(f"Genre: {genre}")
        header.append(f"{len(tracks)} track(s), {_duration(total)}")

        lines = ["", "Tracks:"]
        for t in tracks:
            title = t["title"] or t["path"].stem
            number = f"{t['track']:2d}. " if t["track"] else "    "
            # Per-track artist is printed only when it differs from the
            # album's, which is what makes a compilation legible.
            performer = (
                f"  [{t['artist']}]" if t["artist"] and t["artist"] != artist else ""
            )
            lines.append(f"{number}{title}{performer}  ({_duration(t['seconds'])})")

        return SourceDocument(
            source_type=self.source_type,
            source_key=str(directory.relative_to(self._root)),
            title=f"{artist} — {album}" if artist and album else (album or artist),
            url=None,
            created_at=f"{year}-01-01" if year.isdigit() and len(year) == 4 else None,
            raw={
                "body": "\n".join(header + lines),
                "path": str(directory),
                "album": album,
                "artist": artist,
                "year": year,
                "genre": genre,
                "track_count": len(tracks),
            },
        )
