"""An Audacity 3 project's real layout, read from its binary-XML `project` row.

`sampleblocks` holds the audio, but not what it means: the sample rate, which
blocks belong to which track, where each clip sits on the timeline, and what
is trimmed or muted all live in `project.doc`. That column is Audacity's
serialized XML (ProjectSerializer): a `dict` blob naming every tag and
attribute, then a stream of opcodes that reference those names by id.

Without it, `aup3.py` could only guess 44100 Hz mono and join every block in
id order. On three real projects that guess was wrong for all three: each was
a stereo pair, so the joined stream was the left channel followed by the
right and every duration came out doubled, and one was 48 kHz.

Anything this module does not recognise is a `LayoutError`, never a guess:
a wrong layout produces audio that sounds plausible and transcribes to
nonsense, which is worse than refusing.
"""

from __future__ import annotations

import math
import sqlite3
import struct
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np

FLOAT_SAMPLE_FORMAT = 0x0004000F
_BYTES_PER_SAMPLE = 4

# Same tolerance as aup3.py's own block check, for the same reason: Audacity
# computed the stored stats in float32, this recomputes them in float64.
_VERIFY_REL_TOL = 1e-3
_VERIFY_ABS_TOL = 1e-5

# ProjectSerializer's FieldTypes, in enum order.
_CHARSIZE, _START, _END, _STRING, _INT, _BOOL, _LONG, _LONGLONG = 0, 1, 2, 3, 4, 5, 6, 7
_SIZET, _FLOAT, _DOUBLE, _DATA, _RAW, _PUSH, _POP, _NAME = 8, 9, 10, 11, 12, 13, 14, 15

# Opcode -> (struct format after the name id, value index). Every value
# opcode is followed by a 2-byte name id; FLOAT and DOUBLE carry a trailing
# `digits` int that only matters for re-serializing.
_VALUE_FORMATS: dict[int, str] = {
    _INT: "<i",
    _BOOL: "<B",
    _LONG: "<i",
    _LONGLONG: "<q",
    _SIZET: "<I",
    _FLOAT: "<fi",
    _DOUBLE: "<di",
}

_ENCODINGS = {1: "utf-8", 2: "utf-16-le", 4: "utf-32-le"}


class LayoutError(ValueError):
    """The project's layout could not be read, or describes something this
    module does not handle. Never a partial answer."""


@dataclass
class Element:
    tag: str
    attrs: dict[str, Any] = field(default_factory=dict)
    children: list[Element] = field(default_factory=list)

    def find_all(self, tag: str) -> Iterator[Element]:
        """Every descendant with `tag`, in document order."""
        for child in self.children:
            if child.tag == tag:
                yield child
            yield from child.find_all(tag)


class _Reader:
    def __init__(self, blob: bytes) -> None:
        self.blob = blob
        self.pos = 0

    def unpack(self, fmt: str) -> tuple[Any, ...]:
        size = struct.calcsize(fmt)
        if self.pos + size > len(self.blob):
            raise LayoutError(f"project document truncated at byte {self.pos}")
        values = struct.unpack_from(fmt, self.blob, self.pos)
        self.pos += size
        return values

    def take(self, n: int) -> bytes:
        if n < 0 or self.pos + n > len(self.blob):
            raise LayoutError(f"project document truncated at byte {self.pos}")
        data = self.blob[self.pos : self.pos + n]
        self.pos += n
        return data

    def done(self) -> bool:
        return self.pos >= len(self.blob)


def _text(data: bytes, charsize: int) -> str:
    try:
        return data.decode(_ENCODINGS[charsize])
    except (KeyError, UnicodeDecodeError) as e:
        raise LayoutError(f"unreadable text (char size {charsize})") from e


def parse_binary_xml(dict_blob: bytes, doc_blob: bytes) -> Element:
    """Decode a `project` row into its root element."""
    names: dict[int, str] = {}
    charsize = 1

    def read_name(r: _Reader) -> None:
        ident, length = r.unpack("<HH")
        names[ident] = _text(r.take(length), charsize)

    r = _Reader(bytes(dict_blob))
    while not r.done():
        (op,) = r.unpack("<B")
        if op == _CHARSIZE:
            (charsize,) = r.unpack("<B")
        elif op == _NAME:
            read_name(r)
        else:
            raise LayoutError(f"unexpected opcode {op} in the project dictionary")

    def name(ident: int) -> str:
        try:
            return names[ident]
        except KeyError:
            raise LayoutError(f"name id {ident} is not in the project dictionary") from None

    top = Element(tag="")
    stack = [top]
    r = _Reader(bytes(doc_blob))
    while not r.done():
        (op,) = r.unpack("<B")
        if op == _CHARSIZE:
            (charsize,) = r.unpack("<B")
        elif op == _START:
            (ident,) = r.unpack("<H")
            element = Element(tag=name(ident))
            stack[-1].children.append(element)
            stack.append(element)
        elif op == _END:
            (ident,) = r.unpack("<H")
            if len(stack) == 1 or stack[-1].tag != name(ident):
                raise LayoutError(f"mismatched end tag '{name(ident)}'")
            stack.pop()
        elif op == _STRING:
            ident, length = r.unpack("<Hi")
            stack[-1].attrs[name(ident)] = _text(r.take(length), charsize)
        elif op in _VALUE_FORMATS:
            (ident,) = r.unpack("<H")
            value = r.unpack(_VALUE_FORMATS[op])[0]
            stack[-1].attrs[name(ident)] = bool(value) if op == _BOOL else value
        elif op in (_DATA, _RAW):
            (length,) = r.unpack("<i")
            r.take(length)
        elif op in (_PUSH, _POP):
            pass
        elif op == _NAME:
            read_name(r)
        else:
            raise LayoutError(f"unknown opcode {op} at byte {r.pos - 1}")

    if len(stack) != 1:
        raise LayoutError(f"project document ends inside '{stack[-1].tag}'")
    roots = [child for child in top.children if child.tag == "project"]
    if len(roots) != 1:
        raise LayoutError("project document has no single <project> root")
    return roots[0]


# --- the layout ------------------------------------------------------------------


@dataclass(frozen=True)
class Block:
    start: int  # sample index within the clip's sequence
    blockid: int  # negative: a silent block of that many samples
    length: int


@dataclass(frozen=True)
class Clip:
    offset_s: float  # where the untrimmed sequence would start
    trim_left_s: float
    trim_right_s: float
    num_samples: int
    blocks: tuple[Block, ...]

    def play_span(self, rate: int) -> tuple[int, int]:
        """(first, end) sample index of the audible part, on the timeline."""
        first = round((self.offset_s + self.trim_left_s) * rate)
        audible = self.num_samples - round(self.trim_left_s * rate) - round(self.trim_right_s * rate)
        return first, first + max(audible, 0)


@dataclass(frozen=True)
class Track:
    clips: tuple[Clip, ...]
    mute: bool
    solo: bool
    gain: float


@dataclass(frozen=True)
class ProjectLayout:
    rate: int
    tracks: tuple[Track, ...]

    def audible_tracks(self) -> tuple[Track, ...]:
        """What Audacity would play: solo tracks if any are soloed, else
        every unmuted one."""
        soloed = tuple(t for t in self.tracks if t.solo and not t.mute)
        return soloed or tuple(t for t in self.tracks if not t.mute)

    def _span(self) -> tuple[int, int]:
        spans = [c.play_span(self.rate) for t in self.audible_tracks() for c in t.clips]
        spans = [s for s in spans if s[1] > s[0]]
        if not spans:
            return 0, 0
        return min(s[0] for s in spans), max(s[1] for s in spans)

    @property
    def start_s(self) -> float:
        return self._span()[0] / self.rate

    @property
    def duration_s(self) -> float:
        first, end = self._span()
        return (end - first) / self.rate


def _number(element: Element, attr: str, default: float | None = None) -> float:
    value = element.attrs.get(attr, default)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise LayoutError(f"<{element.tag}> has no numeric '{attr}'")
    return float(value)


def read_layout(conn: sqlite3.Connection) -> ProjectLayout | None:
    """The project's layout, or None when it has no `project` row (an
    unsaved or hand-built file). Raises `LayoutError` when the row exists
    but cannot be trusted."""
    try:
        row = conn.execute("SELECT dict, doc FROM project ORDER BY id LIMIT 1").fetchone()
    except sqlite3.OperationalError:
        return None
    if row is None or row[0] is None or row[1] is None:
        return None
    root = parse_binary_xml(row[0], row[1])

    lengths = {
        int(blockid): int(nbytes) // _BYTES_PER_SAMPLE
        for blockid, nbytes in conn.execute("SELECT blockid, LENGTH(samples) FROM sampleblocks")
    }
    rate = round(_number(root, "rate"))
    if rate <= 0:
        raise LayoutError(f"project sample rate {rate} is not usable")

    tracks = []
    for wavetrack in root.find_all("wavetrack"):
        track_rate = round(_number(wavetrack, "rate", rate))
        if track_rate != rate:
            raise LayoutError(
                f"a track's sample rate ({track_rate} Hz) differs from the project's "
                f"({rate} Hz); mixing them needs resampling this reader does not do"
            )
        clips = tuple(_read_clip(clip, lengths) for clip in wavetrack.find_all("waveclip"))
        tracks.append(
            Track(
                clips=clips,
                mute=bool(wavetrack.attrs.get("mute", False)),
                solo=bool(wavetrack.attrs.get("solo", False)),
                gain=_number(wavetrack, "gain", 1.0),
            )
        )
    return ProjectLayout(rate=rate, tracks=tuple(tracks))


def _read_clip(waveclip: Element, lengths: dict[int, int]) -> Clip:
    sequences = [c for c in waveclip.children if c.tag == "sequence"]
    if len(sequences) != 1:
        # Later Audacity versions nest per-channel clips; not seen yet, so
        # refuse rather than read one channel of it.
        raise LayoutError(f"a clip has {len(sequences)} sequences; expected exactly one")
    sequence = sequences[0]
    if int(sequence.attrs.get("sampleformat", FLOAT_SAMPLE_FORMAT)) != FLOAT_SAMPLE_FORMAT:
        raise LayoutError("a clip is not in floatSample format")
    num_samples = int(_number(sequence, "numsamples"))

    blocks = []
    expected_start = 0
    for waveblock in sequence.children:
        if waveblock.tag != "waveblock":
            continue
        start = int(_number(waveblock, "start"))
        blockid = int(_number(waveblock, "blockid"))
        if blockid < 0:
            length = -blockid
        elif blockid in lengths:
            length = lengths[blockid]
        else:
            raise LayoutError(f"block {blockid} is named by the project but missing from it")
        if start != expected_start:
            raise LayoutError(f"block {blockid} starts at {start}, expected {expected_start}")
        blocks.append(Block(start=start, blockid=blockid, length=length))
        expected_start += length
    if expected_start != num_samples:
        raise LayoutError(f"a clip's blocks hold {expected_start} samples, it declares {num_samples}")

    return Clip(
        offset_s=_number(waveclip, "offset", 0.0),
        trim_left_s=_number(waveclip, "trimLeft", 0.0),
        trim_right_s=_number(waveclip, "trimRight", 0.0),
        num_samples=num_samples,
        blocks=tuple(blocks),
    )


# --- the mono mix ----------------------------------------------------------------


def mix_to_mono(conn: sqlite3.Connection, layout: ProjectLayout) -> tuple[np.ndarray, int]:
    """Every audible track mixed to one float32 channel at the project rate,
    starting at the first audible sample. Returns (samples, rate).

    Tracks are averaged rather than summed: a stereo pair becomes the mean of
    its channels, which keeps the level where it was. Each block is checked
    against its own stored min/max/RMS as it is read.

    Holds the whole mix in memory: 4 bytes per sample at the project rate,
    about 690 MB for an hour at 48 kHz.
    """
    import numpy as np

    first, end = layout._span()
    mix = np.zeros(end - first, dtype=np.float32)
    tracks = layout.audible_tracks()
    for track in tracks:
        for clip in track.clips:
            clip_first, clip_end = clip.play_span(layout.rate)
            skip = round(clip.trim_left_s * layout.rate)  # sequence samples before the audible part
            for block in clip.blocks:
                if block.blockid < 0:
                    continue  # silence
                # The block's audible slice, in sequence coordinates.
                lo = max(block.start, skip)
                hi = min(block.start + block.length, skip + (clip_end - clip_first))
                if hi <= lo:
                    continue
                samples = _read_block(conn, block.blockid)
                at = clip_first - first + (lo - skip)
                mix[at : at + (hi - lo)] += np.float32(track.gain) * samples[lo - block.start : hi - block.start]
    if tracks:
        mix /= len(tracks)
    return mix, layout.rate


def _read_block(conn: sqlite3.Connection, blockid: int) -> np.ndarray:
    import numpy as np

    row = conn.execute(
        "SELECT sampleformat, summin, summax, sumrms, samples FROM sampleblocks WHERE blockid = ?",
        (blockid,),
    ).fetchone()
    if row is None:
        raise LayoutError(f"block {blockid} is missing")
    fmt, summin, summax, sumrms, blob = row
    if int(fmt) != FLOAT_SAMPLE_FORMAT:
        raise LayoutError(f"block {blockid} is not floatSample")
    samples = np.frombuffer(bytes(blob), dtype="<f4").astype(np.float32)
    if samples.size:
        wide = samples.astype(np.float64)
        stats = (float(wide.min()), float(wide.max()), float(np.sqrt(np.mean(wide**2))))
        stored = (float(summin), float(summax), float(sumrms))
        if not all(
            math.isclose(a, b, rel_tol=_VERIFY_REL_TOL, abs_tol=_VERIFY_ABS_TOL)
            for a, b in zip(stats, stored, strict=True)
        ):
            raise LayoutError(
                f"block {blockid}: recomputed min/max/RMS do not match its stored summary"
            )
    return samples
