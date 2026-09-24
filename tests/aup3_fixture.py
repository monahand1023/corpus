"""Build synthetic Audacity 3 projects, including the binary-XML `project`
row, so the layout reader is tested without a real recording in the repo.

The encoding mirrors Audacity's ProjectSerializer: a `dict` blob of
FT_CharSize + FT_Name entries, then a `doc` blob of opcodes whose value
types match what a real 3.2 project uses for each attribute.
"""

from __future__ import annotations

import math
import sqlite3
import struct
from dataclasses import dataclass, field
from pathlib import Path

FLOAT_SAMPLE_FORMAT = 0x0004000F

# Opcodes, in ProjectSerializer's enum order.
FT_CHARSIZE, FT_START, FT_END, FT_STRING, FT_INT, FT_BOOL, FT_LONG = 0, 1, 2, 3, 4, 5, 6
FT_LONGLONG, FT_SIZET, FT_FLOAT, FT_DOUBLE, FT_DATA, FT_RAW, FT_PUSH, FT_POP, FT_NAME = (
    7, 8, 9, 10, 11, 12, 13, 14, 15,
)


@dataclass
class ClipSpec:
    samples: list[float]
    offset_s: float = 0.0
    trim_left_s: float = 0.0
    trim_right_s: float = 0.0
    block_size: int = 1000


@dataclass
class TrackSpec:
    clips: list[ClipSpec]
    rate: int = 8000
    channel: int = 2  # Audacity: 0 left, 1 right, 2 mono
    linked: int = 0
    mute: bool = False
    solo: bool = False
    gain: float = 1.0


@dataclass
class _Encoder:
    names: dict[str, int] = field(default_factory=dict)
    doc: bytearray = field(default_factory=bytearray)

    def _id(self, name: str) -> int:
        return self.names.setdefault(name, len(self.names))

    def start(self, tag: str) -> None:
        self.doc += struct.pack("<BH", FT_START, self._id(tag))

    def end(self, tag: str) -> None:
        self.doc += struct.pack("<BH", FT_END, self._id(tag))

    def string(self, name: str, value: str) -> None:
        raw = value.encode("utf-32-le")
        self.doc += struct.pack("<BHi", FT_STRING, self._id(name), len(raw)) + raw

    def int_(self, name: str, value: int) -> None:
        self.doc += struct.pack("<BHi", FT_INT, self._id(name), value)

    def bool_(self, name: str, value: bool) -> None:
        self.doc += struct.pack("<BHB", FT_BOOL, self._id(name), int(value))

    def long(self, name: str, value: int) -> None:
        self.doc += struct.pack("<BHi", FT_LONG, self._id(name), value)

    def longlong(self, name: str, value: int) -> None:
        self.doc += struct.pack("<BHq", FT_LONGLONG, self._id(name), value)

    def sizet(self, name: str, value: int) -> None:
        self.doc += struct.pack("<BHI", FT_SIZET, self._id(name), value)

    def double(self, name: str, value: float) -> None:
        self.doc += struct.pack("<BHdi", FT_DOUBLE, self._id(name), value, 19)

    def raw(self, text: str) -> None:
        data = text.encode("utf-32-le")
        self.doc += struct.pack("<Bi", FT_RAW, len(data)) + data

    def dict_blob(self) -> bytes:
        out = bytearray(struct.pack("<BB", FT_CHARSIZE, 4))
        for name, ident in self.names.items():
            raw = name.encode("utf-32-le")
            out += struct.pack("<BHH", FT_NAME, ident, len(raw)) + raw
        return bytes(out)


def _stats(values: list[float]) -> tuple[float, float, float]:
    if not values:
        return 0.0, 0.0, 0.0
    return min(values), max(values), math.sqrt(sum(v * v for v in values) / len(values))


def make_project(path: Path, tracks: list[TrackSpec], project_rate: float | None = None) -> Path:
    """Write a `.aup3` whose `project` row describes `tracks` and whose
    `sampleblocks` hold their samples."""
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE project (id INTEGER PRIMARY KEY, dict BLOB, doc BLOB)")
    conn.execute("CREATE TABLE autosave (id INTEGER PRIMARY KEY, dict BLOB, doc BLOB)")
    conn.execute(
        "CREATE TABLE sampleblocks ("
        "blockid INTEGER PRIMARY KEY AUTOINCREMENT, sampleformat INTEGER, "
        "summin REAL, summax REAL, sumrms REAL, "
        "summary256 BLOB, summary64k BLOB, samples BLOB)"
    )

    enc = _Encoder()
    enc.raw('<?xml version="1.0" standalone="no" ?>\n')
    enc.start("project")
    enc.string("version", "1.3.0")
    enc.string("audacityversion", "3.2.5")
    enc.double("rate", float(project_rate or (tracks[0].rate if tracks else 44100)))
    for track in tracks:
        enc.start("wavetrack")
        enc.string("name", "Audio")
        enc.int_("channel", track.channel)
        enc.int_("linked", track.linked)
        enc.bool_("mute", track.mute)
        enc.bool_("solo", track.solo)
        enc.int_("rate", track.rate)
        enc.double("gain", track.gain)
        enc.long("sampleformat", FLOAT_SAMPLE_FORMAT)
        for clip in track.clips:
            enc.start("waveclip")
            enc.double("offset", clip.offset_s)
            enc.double("trimLeft", clip.trim_left_s)
            enc.double("trimRight", clip.trim_right_s)
            enc.start("sequence")
            enc.sizet("maxsamples", clip.block_size)
            enc.sizet("sampleformat", FLOAT_SAMPLE_FORMAT)
            enc.longlong("numsamples", len(clip.samples))
            for start in range(0, len(clip.samples), clip.block_size):
                values = clip.samples[start : start + clip.block_size]
                lo, hi, rms = _stats(values)
                cur = conn.execute(
                    "INSERT INTO sampleblocks (sampleformat, summin, summax, sumrms, samples) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (FLOAT_SAMPLE_FORMAT, lo, hi, rms, struct.pack(f"<{len(values)}f", *values)),
                )
                enc.start("waveblock")
                enc.longlong("start", start)
                enc.longlong("blockid", int(cur.lastrowid))
                enc.end("waveblock")
            enc.end("sequence")
            enc.end("waveclip")
        enc.end("wavetrack")
    enc.end("project")

    conn.execute(
        "INSERT INTO project (id, dict, doc) VALUES (1, ?, ?)",
        (enc.dict_blob(), bytes(enc.doc)),
    )
    conn.commit()
    conn.close()
    return path


def tone(n: int, value: float) -> list[float]:
    """A constant signal: easy to reason about once mixed."""
    return [value] * n
