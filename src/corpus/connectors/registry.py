"""Connector registry — maps the `type` field from corpus.toml's [[sources]]
to a (Connector, Chunker) pair.

To add a new source type:
  1. Write a Connector and a Chunker (see `markdown.py` for the reference)
  2. Add an entry below
  3. Reference it in your corpus.toml [[sources]] with `type = "your_name"`
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from corpus.config import SourceConfig
from corpus.connectors.markdown import MarkdownChunker

# Default glob per connector type. Single source of truth: the factories below
# read it, and `corpus.util.autodetect` scans a folder with it to work out which
# connectors apply. Adding a connector without an entry here means `--path`
# silently never finds its files, so `tests/test_autodetect.py` asserts this
# mapping covers every registered type.
DEFAULT_GLOBS: dict[str, str] = {
    "markdown": "**/*.md",
    "text": "**/*.txt",
    "pdf": "**/*.pdf",
    "html": "**/*.html",
    "docx": "**/*.docx",
    "xls": "**/*.xls",
    "xlsx": "**/*.xlsx",
    "rtf": "**/*.rtf",
    "zip": "**/*.zip",
    "pptx": "**/*.pptx",
    "csv": "**/*.csv",
    "tsv": "**/*.tsv",
    "aup3": "**/*.aup3",
    "abcdp": "**/*.abcdp",
    # Detection pattern only. The connector itself sweeps every extension in
    # `music.MUSIC_EXTENSIONS`; Path.glob has no alternation, so one pattern
    # cannot express them all without also matching `.md`.
    "music": "**/*.mp3",
    "olm": "**/*.olm",
    # Detection pattern only. The connector reads a sidecar DATABASE, not
    # the media itself, so pointing `--path` at a folder of recordings
    # finds nothing until a transcription pass has produced one.
    "transcripts": "**/*transcripts.db",
}


class _ConnectorFactory(Protocol):
    def __call__(self, cfg: SourceConfig) -> tuple[Any, Any]: ...



@dataclass(frozen=True)
class _FileType:
    """A connector that takes (source_type, path, glob) and is chunked as
    markdown -- most of them. `extra` is (probe import, extra name, pip
    package) for one that needs an optional dependency."""

    module: str
    cls: str
    label: str
    extra: tuple[str, str, str] | None = None
    kwargs: tuple[tuple[str, Any], ...] = ()


_FILE_TYPES: dict[str, _FileType] = {
    "markdown": _FileType("corpus.connectors.markdown", "MarkdownConnector", "Markdown"),
    "text": _FileType("corpus.connectors.text", "TextConnector", "Text"),
    "pdf": _FileType("corpus.connectors.pdf", "PdfConnector", "PDF", ("pypdf", "pdf", "pypdf")),
    "html": _FileType(
        "corpus.connectors.html", "HtmlConnector", "HTML", ("trafilatura", "html", "trafilatura")
    ),
    "docx": _FileType(
        "corpus.connectors.docx", "DocxConnector", "Docx", ("docx", "docx", "python-docx")
    ),
    "xls": _FileType("corpus.connectors.xls", "XlsConnector", "Xls", ("xlrd", "xls", "xlrd")),
    "xlsx": _FileType(
        "corpus.connectors.xlsx", "XlsxConnector", "Xlsx", ("openpyxl", "xlsx", "openpyxl")
    ),
    "rtf": _FileType(
        "corpus.connectors.rtf", "RtfConnector", "Rtf", ("striprtf.striprtf", "rtf", "striprtf")
    ),
    "pptx": _FileType(
        "corpus.connectors.pptx", "PptxConnector", "Pptx", ("pptx", "pptx", "python-pptx")
    ),
    # stdlib only -- see csv_.py for why pandas was not added.
    "csv": _FileType("corpus.connectors.csv_", "CsvConnector", "CSV"),
    # Same class, tab default (still sniffed per file), and its own glob so a
    # folder of `.tsv` exports is found without a hand-written override.
    "tsv": _FileType(
        "corpus.connectors.csv_", "CsvConnector", "TSV", kwargs=(("default_delimiter", "\t"),)
    ),
    # plistlib is stdlib, so a Contacts backup needs no extra.
    "abcdp": _FileType("corpus.connectors.abcdp", "AbcdpConnector", "Abcdp"),
}


def _file_builder(type_name: str, spec: _FileType) -> _ConnectorFactory:
    def build(cfg: SourceConfig) -> tuple[Any, MarkdownChunker]:
        if spec.extra is not None:
            probe, extra, package = spec.extra
            try:
                # `__import__`, not importlib: the import statement's own hook,
                # which is what a missing package is simulated through in tests.
                __import__(probe)
            except ImportError as e:
                raise ImportError(
                    f"{spec.label} connector requires the [{extra}] extra. "
                    f"Install with `pip install corpus-rag[{extra}]` or `uv add {package}`."
                ) from e
        connector_cls = getattr(__import__(spec.module, fromlist=[spec.cls]), spec.cls)
        connector = connector_cls(
            source_type=cfg.name,
            path=cfg.path,
            glob=cfg.glob or DEFAULT_GLOBS[type_name],
            **dict(spec.kwargs),
        )
        return connector, MarkdownChunker(source_type=cfg.name)

    build.__name__ = f"_build_{type_name}"
    return build



def _build_transcripts(cfg: SourceConfig) -> tuple[Any, Any]:
    """Transcribed audio/video, read from a sidecar database.

    `path` here is the DATABASE, not a media folder -- transcription is slow
    and expensive and indexing is neither, so they are separate steps joined
    by that file. No optional extra is needed to READ one; producing it is
    what costs.
    """
    from corpus.connectors.transcripts import TranscriptChunker, TranscriptConnector

    connector = TranscriptConnector(
        source_type=cfg.name,
        path=cfg.path,
        exclude=getattr(cfg, "exclude", ()) or (),
    )
    return connector, TranscriptChunker(source_type=cfg.name)






















def _build_aup3(cfg: SourceConfig) -> tuple[Any, MarkdownChunker]:
    # No import guard, like zip/csv: aup3.py depends only on the stdlib
    # (sqlite3, wave, array). ffmpeg is an external binary, not a pip
    # package, and is only touched by `extract_audio(..., audio_format=
    # "flac"/"mp3")` — a separate, explicitly-called function this factory
    # never invokes.
    from corpus.connectors.aup3 import AupThreeConnector

    connector = AupThreeConnector(
        source_type=cfg.name,
        path=cfg.path,
        glob=cfg.glob or DEFAULT_GLOBS["aup3"],
        sample_rate=cfg.sample_rate,
        channels=cfg.channels,
    )
    return connector, MarkdownChunker(source_type=cfg.name)




def _build_music(cfg: SourceConfig) -> tuple[Any, MarkdownChunker]:
    try:
        import mutagen  # noqa: F401

        from corpus.connectors.music import MusicConnector
    except ImportError as e:
        raise ImportError(
            "Music connector requires the [music] extra. "
            "Install with `pip install corpus-rag[music]` or `uv add mutagen`."
        ) from e
    connector = MusicConnector(
        source_type=cfg.name,
        path=cfg.path,
        glob=cfg.glob,
    )
    return connector, MarkdownChunker(source_type=cfg.name)


def _build_olm(cfg: SourceConfig) -> tuple[Any, MarkdownChunker]:
    # No import guard: olm.py needs only the stdlib `zipfile`, and it reads
    # members straight out of the archive rather than extracting and
    # delegating the way `zip.py` does -- a real `.olm` is a large amount, so there
    # is no per-file-type extra to be missing here.
    from corpus.connectors.olm import build as build_olm

    return build_olm(cfg), MarkdownChunker(source_type=cfg.name)


def _build_zip(cfg: SourceConfig) -> tuple[Any, MarkdownChunker]:
    # No import guard here, unlike the factories above: zip.py depends only on
    # the stdlib `zipfile`. It composes the OTHER factories in this dict at
    # `load()` time (per file type actually found inside an archive), so a
    # missing optional extra for e.g. pdf surfaces there instead — the same
    # actionable ImportError, just discovered lazily since which extras are
    # needed depends on archive contents, not on configuring this source.
    from corpus.connectors.zip import ZipConnector

    connector = ZipConnector(
        source_type=cfg.name,
        path=cfg.path,
        glob=cfg.glob or DEFAULT_GLOBS["zip"],
        exclude_dependencies=cfg.exclude_dependencies,
    )
    return connector, MarkdownChunker(source_type=cfg.name)


CONNECTOR_REGISTRY: dict[str, _ConnectorFactory] = {
    **{name: _file_builder(name, spec) for name, spec in _FILE_TYPES.items()},
    "music": _build_music,
    "olm": _build_olm,
    "zip": _build_zip,
    "aup3": _build_aup3,
    "transcripts": _build_transcripts,
}

# A pristine copy of what the ENGINE provides, taken at import time before any
# consumer can register over it. Comparing the live registry against this is
# the only way to see a SHADOWED connector, and a shadowed connector is how
# engine fixes silently stop arriving: one consumer held a full copy of the
# transcript connector and registered it over this one, so fixes landed here
# and did nothing there while every ingest reported success.
#
# See `corpus.cli.doctor._check_shadowed_components`.
_BUILTIN_BUILDERS: dict[str, _ConnectorFactory] = dict(CONNECTOR_REGISTRY)


def build_pipeline(cfg: SourceConfig) -> tuple[Any, Any]:
    """Return (connector, chunker) for a configured source."""
    if cfg.type not in CONNECTOR_REGISTRY:
        raise ValueError(
            f"Source '{cfg.name}' uses type='{cfg.type}', which is not registered. "
            f"Available: {sorted(CONNECTOR_REGISTRY)}. "
            f"Add your own by editing src/corpus/connectors/registry.py."
        )
    return CONNECTOR_REGISTRY[cfg.type](cfg)
