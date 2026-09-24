"""Make transcribed audio and video searchable, from a sidecar database.

`corpus.transcripts.store` defines what a transcription run writes; this reads
it. Anything that can fill that schema — this project's own pipeline, or your
own wrapper around whatever speech-to-text you prefer — becomes indexable
without further work.

Three decisions, each measured rather than chosen:

**One chunk per transcribed window, and no merging.** Every window carries its
own detected language and timestamps, so splitting there keeps a Japanese
passage from straddling a boundary with English and lets a hit say WHERE in a
two-hour recording it came from.

Merging adjacent windows into larger chunks was tried, because these chunks
average ~60 tokens against ~400 for document sources in the same index, and
that looked undersized. It was much worse: recall@5 0.625 -> 0.375, and two
queries that had been passing stopped. The comparison was a false analogy. In
a document the unit of interest is a topical SECTION, so a bigger chunk holds
more of the answer; in a recording it is a MOMENT, and a thirty-second window
where someone shouts about a goalie is a precise match that stops being one
when averaged into five minutes of surrounding chatter. Small chunks are the
shape of this source, not a defect in it. Contextual retrieval failed on the
same archive for the same reason -- a context sentence would have been ~78% of
the embedded text.

**The length floor is LANGUAGE-AWARE.** A flat character count discriminates
against CJK: "誕生日おめでとうございます、みなさん。" is a complete sentence in 19
characters and the English it translates to needs about 40, so a flat floor of
25 silently drops the Japanese and keeps the English. On an archive that is
roughly half Japanese that is not a rounding error. (Caught by a test, not by
review.)

**A sign-off is cut before the window is judged, not after.** A window reading
"...まって、まってご視聴ありがとうございました" is real speech with a caption
artefact glued to the end. `subtitle_boilerplate` deliberately will not remove
it -- it answers "is this text ENTIRELY an artefact?", and dropping text merely
for containing one measurably deletes 12.3% of a real archive. `strip_caption_tail`
cuts the tail and keeps the speech.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

from corpus.connectors.discovery import _excluded
from corpus.transcripts import (
    DEFAULT_MAX_LOOPING_SHARE,
    looping_share,
    strip_caption_tail,
    subtitle_boilerplate,
)
from corpus.types import Chunk, ChunkKind, ChunkMetadata, SourceDocument
from corpus.util.hash import chunk_id, sha256
from corpus.util.scrub import scrub
from corpus.util.sqlite_ro import connect_ro
from corpus.util.tokens import estimate_tokens

# A window that survived filtering but says almost nothing is not worth a chunk
# of its own; it dilutes the index without adding anything findable. See the
# module docstring for why there are two floors rather than one.
MIN_SEGMENT_CHARS = 25
MIN_SEGMENT_CHARS_CJK = 8

_CJK_RANGES = (
    (0x3040, 0x30FF),   # Hiragana + Katakana
    (0x3400, 0x4DBF),   # CJK Extension A
    (0x4E00, 0x9FFF),   # CJK Unified Ideographs
    (0xF900, 0xFAFF),   # CJK Compatibility Ideographs
    (0xFF66, 0xFF9F),   # Halfwidth Katakana
)


def _has_cjk(text: str) -> bool:
    return any(any(lo <= ord(ch) <= hi for lo, hi in _CJK_RANGES) for ch in text)


def _long_enough(text: str) -> bool:
    floor = MIN_SEGMENT_CHARS_CJK if _has_cjk(text) else MIN_SEGMENT_CHARS
    return len(text) >= floor


def _rescuable(text: str) -> bool:
    """Junk tests only, WITHOUT the length floor.

    The rescue below exists to restore a window that is merely short, so it
    must skip `_long_enough` -- but every other reason `worth_indexing` says
    no still applies. Keeping them in one place is the point: the rescue was
    written to check boilerplate alone, so when the loop signal arrived it was
    a hole from the first day. 12 looping chunks reached a live index through
    it after the loop filter had already been added everywhere else.
    """
    return (
        not subtitle_boilerplate(text)
        and looping_share(text) < DEFAULT_MAX_LOOPING_SHARE
    )


def worth_indexing(text: str) -> bool:
    """A window worth a chunk: long enough, not boilerplate, not a loop.

    Every check here is per-WINDOW, not per-transcript, and that is the point.
    A recording can be entirely real and still contain one window where the
    model emitted nothing but "Terima kasih telah menonton", and that window
    becomes a chunk of its own, indexed and searchable. 86 such chunks were
    found in a live index after whole-transcript filtering had already run:
    the transcripts were genuine, only those windows were not.

    The loop check is here for the identical reason and was measured the same
    way. In one live index 716 chunks were the model looping, but only 553 of
    them belonged to transcripts a whole-transcript judgement would reject --
    the remaining 163 were single looping windows inside genuine recordings,
    reachable only from here.
    """
    return _long_enough(text) and _rescuable(text)


class TranscriptConnector:
    """Reads a transcript sidecar, yielding one document per media file."""

    def __init__(
        self,
        source_type: str,
        path: Path | str,
        *,
        exclude: Sequence[str] = (),
        exclude_fn: Callable[[str], bool] | None = None,
    ) -> None:
        self.source_type = source_type
        self._db = Path(path).expanduser()
        # Substrings of the media path to skip. A consumer re-applies its own
        # exclusions at INGEST as well as at transcription time, because rules
        # tighten while a long run is in flight and the database ends up
        # holding rows written under the older ones. Enumeration-time filtering
        # cannot retract those; this can.
        self._exclude = tuple(exclude)
        # A predicate, for exclusions a substring cannot express -- "media
        # inside a photo library unless it was opted in", "karaoke backing
        # tracks". The MECHANISM belongs here because the retraction problem
        # above is general; the POLICY belongs to the archive, which is why
        # this is a hook and not a list of rules.
        self._exclude_fn = exclude_fn
        if not self._db.is_file():
            # Raise rather than yield nothing: an empty enumeration makes the
            # ingester treat every already-indexed transcript as an orphan and
            # delete it.
            raise FileNotFoundError(
                f"transcript database not found: {self._db}. "
                "Run a transcription pass first."
            )
        self.failed_files = 0
        self.skipped_files = 0
        self.excluded_files = 0

    def _is_excluded(self, media_path: str) -> bool:
        """A substring of the media path (this connector's original meaning),
        or the fnmatch / folder-name forms every file connector accepts."""
        if not self._exclude:
            return False
        if any(pattern in media_path for pattern in self._exclude):
            return True
        path = PurePosixPath(media_path)
        return _excluded(path, path.name, self._exclude)

    def load(self) -> Iterable[SourceDocument]:
        conn = connect_ro(self._db)
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                "SELECT path, duration_s, text, languages, segments,"
                " transcribed_at, dropped_windows FROM transcripts"
            ).fetchall()
        finally:
            conn.close()

        for row in rows:
            if self._is_excluded(row["path"]) or (
                self._exclude_fn is not None and self._exclude_fn(row["path"])
            ):
                self.excluded_files += 1
                continue
            try:
                segments = json.loads(row["segments"])
                languages = json.loads(row["languages"])
            except (TypeError, ValueError):
                # A row we cannot parse is a real FAILURE, not a policy skip.
                # Counting it as failed suppresses pruning, rather than letting
                # this run delete the chunks it could not read.
                self.failed_files += 1
                continue
            if not (row["text"] or "").strip():
                self.skipped_files += 1
                continue
            media = Path(row["path"])
            yield SourceDocument(
                source_type=self.source_type,
                source_key=row["path"],
                title=media.name,
                url=f"file://{row['path']}",
                created_at=row["transcribed_at"],
                updated_at=row["transcribed_at"],
                raw={
                    "text": row["text"],
                    "segments": segments,
                    "languages": languages,
                    "duration_s": row["duration_s"],
                    "dropped_windows": row["dropped_windows"],
                    "media_path": str(media),
                },
            )


class TranscriptChunker:
    """One chunk per transcribed window, so language and timing survive."""

    def __init__(self, source_type: str) -> None:
        self.source_type = source_type

    def chunk(self, doc: SourceDocument) -> list[Chunk]:
        raw = doc.raw or {}
        segments: list[dict[str, Any]] = list(raw.get("segments") or [])

        # Cut a caption sign-off off the end of each window BEFORE judging it,
        # so a window of real speech is not thrown away for what a model
        # appended to it. The database keeps the transcriber's original output;
        # the index is the derived artefact and is the one that gets cleaned,
        # which also means a rule change needs only a re-ingest.
        segments = [
            {**s, "text": strip_caption_tail(s.get("text") or "")} for s in segments
        ]
        kept = [s for s in segments if worth_indexing((s.get("text") or "").strip())]

        # The length floor must not silently delete a whole recording.
        # Measured on a sample of transcribed files, 6% produced NO chunks at
        # all and were not junk: an eleven-character sentence, a short thank
        # you. Short voice memos and three-second clips whose entire
        # transcript is one sentence. Each was counted as a document and indexed as nothing:
        # present in the archive, unfindable forever.
        #
        # A short window inside a long recording is still worth dropping. A
        # short window that IS the recording is the recording.
        if not kept and segments:
            # The rescue restores a SHORT window, never a worthless one. It
            # used to take the longest segment unconditionally, so a recording
            # whose every window was caption boilerplate had its sign-off
            # indexed as the document -- reintroducing exactly what
            # `worth_indexing` exists to remove.
            best = max(segments, key=lambda s: len((s.get("text") or "").strip()))
            text = (best.get("text") or "").strip()
            if text and _rescuable(text):
                kept = [best]

        out: list[Chunk] = []
        for idx, seg in enumerate(kept):
            # SCRUBBED, like every other chunker. This one did not, and
            # transcripts are the worst source type to miss: they are speech,
            # so a credential here was read ALOUD -- dictated on a call,
            # walked through on a screen share. It went into the index
            # verbatim and into the embedding request, which leaves the
            # machine. scrub.py's threat model is "the .db file leaks"; this
            # was the one path where that model did not hold.
            #
            # Before the hash, not after: a hash of the unscrubbed text would
            # leak the original through change detection, since a re-ingest
            # would only look unchanged while the secret was still present.
            text = scrub((seg.get("text") or "").strip())
            start, end = seg.get("start"), seg.get("end")
            out.append(
                Chunk(
                    id=chunk_id(self.source_type, doc.source_key, ChunkKind.SECTION, idx),
                    content=text,
                    content_hash=sha256(text),
                    metadata=ChunkMetadata(
                        source_type=self.source_type,
                        source_key=doc.source_key,
                        chunk_kind=ChunkKind.SECTION,
                        chunk_index=idx,
                        title=doc.title,
                        url=doc.url,
                        created_at=doc.created_at,
                        updated_at=doc.updated_at,
                        token_count=estimate_tokens(text),
                        extra={
                            "language": seg.get("lang"),
                            "start_s": start,
                            "end_s": end,
                            "timestamp": hms(start),
                            "media_path": raw.get("media_path"),
                            "duration_s": raw.get("duration_s"),
                        },
                    ),
                )
            )
        return out


def hms(seconds: float | None) -> str | None:
    """`3725.0` -> `"1:02:05"`, so a hit says where in the recording it is."""
    if seconds is None:
        return None
    total = int(seconds)
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"
