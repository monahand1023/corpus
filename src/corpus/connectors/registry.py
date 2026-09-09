"""Connector registry — maps the `type` field from corpus.toml's [[sources]]
to a (Connector, Chunker) pair.

To add a new source type:
  1. Write a Connector and a Chunker (see `markdown.py` for the reference)
  2. Add an entry below
  3. Reference it in your corpus.toml [[sources]] with `type = "your_name"`
"""

from __future__ import annotations

from typing import Any, Protocol

from corpus.config import SourceConfig
from corpus.connectors.markdown import MarkdownChunker, MarkdownConnector

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
    "olm": "**/*.olm",
}


class _ConnectorFactory(Protocol):
    def __call__(self, cfg: SourceConfig) -> tuple[Any, Any]: ...


def _build_markdown(cfg: SourceConfig) -> tuple[MarkdownConnector, MarkdownChunker]:
    connector = MarkdownConnector(
        source_type=cfg.name,
        path=cfg.path,
        glob=cfg.glob or DEFAULT_GLOBS["markdown"],
    )
    chunker = MarkdownChunker(source_type=cfg.name)
    return connector, chunker


def _build_text(cfg: SourceConfig) -> tuple[Any, MarkdownChunker]:
    from corpus.connectors.text import TextConnector

    connector = TextConnector(
        source_type=cfg.name,
        path=cfg.path,
        glob=cfg.glob or DEFAULT_GLOBS["text"],
    )
    return connector, MarkdownChunker(source_type=cfg.name)


def _build_pdf(cfg: SourceConfig) -> tuple[Any, MarkdownChunker]:
    try:
        import pypdf  # noqa: F401

        from corpus.connectors.pdf import PdfConnector
    except ImportError as e:
        raise ImportError(
            "PDF connector requires the [pdf] extra. "
            "Install with `pip install corpus-rag[pdf]` or `uv add pypdf`."
        ) from e
    connector = PdfConnector(
        source_type=cfg.name,
        path=cfg.path,
        glob=cfg.glob or DEFAULT_GLOBS["pdf"],
    )
    return connector, MarkdownChunker(source_type=cfg.name)


def _build_html(cfg: SourceConfig) -> tuple[Any, MarkdownChunker]:
    try:
        import trafilatura  # noqa: F401

        from corpus.connectors.html import HtmlConnector
    except ImportError as e:
        raise ImportError(
            "HTML connector requires the [html] extra. "
            "Install with `pip install corpus-rag[html]` or `uv add trafilatura`."
        ) from e
    connector = HtmlConnector(
        source_type=cfg.name,
        path=cfg.path,
        glob=cfg.glob or DEFAULT_GLOBS["html"],
    )
    return connector, MarkdownChunker(source_type=cfg.name)


def _build_docx(cfg: SourceConfig) -> tuple[Any, MarkdownChunker]:
    try:
        import docx  # noqa: F401

        from corpus.connectors.docx import DocxConnector
    except ImportError as e:
        raise ImportError(
            "Docx connector requires the [docx] extra. "
            "Install with `pip install corpus-rag[docx]` or `uv add python-docx`."
        ) from e
    connector = DocxConnector(
        source_type=cfg.name,
        path=cfg.path,
        glob=cfg.glob or DEFAULT_GLOBS["docx"],
    )
    return connector, MarkdownChunker(source_type=cfg.name)


def _build_xls(cfg: SourceConfig) -> tuple[Any, MarkdownChunker]:
    try:
        import xlrd  # noqa: F401

        from corpus.connectors.xls import XlsConnector
    except ImportError as e:
        raise ImportError(
            "Xls connector requires the [xls] extra. "
            "Install with `pip install corpus-rag[xls]` or `uv add xlrd`."
        ) from e
    connector = XlsConnector(
        source_type=cfg.name,
        path=cfg.path,
        glob=cfg.glob or DEFAULT_GLOBS["xls"],
    )
    return connector, MarkdownChunker(source_type=cfg.name)


def _build_xlsx(cfg: SourceConfig) -> tuple[Any, MarkdownChunker]:
    try:
        import openpyxl  # noqa: F401

        from corpus.connectors.xlsx import XlsxConnector
    except ImportError as e:
        raise ImportError(
            "Xlsx connector requires the [xlsx] extra. "
            "Install with `pip install corpus-rag[xlsx]` or `uv add openpyxl`."
        ) from e
    connector = XlsxConnector(
        source_type=cfg.name,
        path=cfg.path,
        glob=cfg.glob or DEFAULT_GLOBS["xlsx"],
    )
    return connector, MarkdownChunker(source_type=cfg.name)


def _build_rtf(cfg: SourceConfig) -> tuple[Any, MarkdownChunker]:
    try:
        import striprtf.striprtf  # noqa: F401

        from corpus.connectors.rtf import RtfConnector
    except ImportError as e:
        raise ImportError(
            "Rtf connector requires the [rtf] extra. "
            "Install with `pip install corpus-rag[rtf]` or `uv add striprtf`."
        ) from e
    connector = RtfConnector(
        source_type=cfg.name,
        path=cfg.path,
        glob=cfg.glob or DEFAULT_GLOBS["rtf"],
    )
    return connector, MarkdownChunker(source_type=cfg.name)


def _build_pptx(cfg: SourceConfig) -> tuple[Any, MarkdownChunker]:
    try:
        import pptx  # noqa: F401

        from corpus.connectors.pptx import PptxConnector
    except ImportError as e:
        raise ImportError(
            "Pptx connector requires the [pptx] extra. "
            "Install with `pip install corpus-rag[pptx]` or `uv add python-pptx`."
        ) from e
    connector = PptxConnector(
        source_type=cfg.name,
        path=cfg.path,
        glob=cfg.glob or DEFAULT_GLOBS["pptx"],
    )
    return connector, MarkdownChunker(source_type=cfg.name)


def _build_csv(cfg: SourceConfig) -> tuple[Any, MarkdownChunker]:
    # No import guard, unlike the factories above: this connector uses only
    # the stdlib `csv` module — see csv_.py's module docstring for why
    # pandas was deliberately not added.
    from corpus.connectors.csv_ import CsvConnector

    connector = CsvConnector(
        source_type=cfg.name,
        path=cfg.path,
        glob=cfg.glob or DEFAULT_GLOBS["csv"],
    )
    return connector, MarkdownChunker(source_type=cfg.name)


def _build_tsv(cfg: SourceConfig) -> tuple[Any, MarkdownChunker]:
    # Same connector class as `csv`, just a tab default (still auto-detected
    # per file via csv.Sniffer — see csv_.py) and its own default glob so a
    # folder of `.tsv` exports is discovered without hand-writing a
    # `[[sources]]` glob override.
    from corpus.connectors.csv_ import CsvConnector

    connector = CsvConnector(
        source_type=cfg.name,
        path=cfg.path,
        glob=cfg.glob or DEFAULT_GLOBS["tsv"],
        default_delimiter="\t",
    )
    return connector, MarkdownChunker(source_type=cfg.name)


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
    "markdown": _build_markdown,
    "text": _build_text,
    "pdf": _build_pdf,
    "html": _build_html,
    "docx": _build_docx,
    "xls": _build_xls,
    "xlsx": _build_xlsx,
    "rtf": _build_rtf,
    "olm": _build_olm,
    "zip": _build_zip,
    "pptx": _build_pptx,
    "csv": _build_csv,
    "tsv": _build_tsv,
    "aup3": _build_aup3,
}


def build_pipeline(cfg: SourceConfig) -> tuple[Any, Any]:
    """Return (connector, chunker) for a configured source."""
    if cfg.type not in CONNECTOR_REGISTRY:
        raise ValueError(
            f"Source '{cfg.name}' uses type='{cfg.type}', which is not registered. "
            f"Available: {sorted(CONNECTOR_REGISTRY)}. "
            f"Add your own by editing src/corpus/connectors/registry.py."
        )
    return CONNECTOR_REGISTRY[cfg.type](cfg)
