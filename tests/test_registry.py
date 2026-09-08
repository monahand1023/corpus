from __future__ import annotations

import builtins

import pytest

from corpus.config import SourceConfig
from corpus.connectors.registry import CONNECTOR_REGISTRY, build_pipeline


def test_all_builtin_types_registered() -> None:
    assert "markdown" in CONNECTOR_REGISTRY
    assert "text" in CONNECTOR_REGISTRY
    assert "pdf" in CONNECTOR_REGISTRY
    assert "html" in CONNECTOR_REGISTRY
    assert "docx" in CONNECTOR_REGISTRY
    assert "xlsx" in CONNECTOR_REGISTRY
    assert "rtf" in CONNECTOR_REGISTRY


def test_build_markdown(tmp_path) -> None:
    cfg = SourceConfig(name="notes", type="markdown", path=str(tmp_path))
    connector, chunker = build_pipeline(cfg)
    assert connector.source_type == "notes"
    assert chunker.source_type == "notes"


def test_build_text(tmp_path) -> None:
    cfg = SourceConfig(name="notes", type="text", path=str(tmp_path))
    connector, _chunker = build_pipeline(cfg)
    assert connector.source_type == "notes"


def test_unknown_type_raises() -> None:
    cfg = SourceConfig(name="foo", type="nonexistent", path="/tmp")
    with pytest.raises(ValueError, match="not registered"):
        build_pipeline(cfg)


# ---------------------------------------------------------------------------
# FIX 1: each connector's third-party library is imported LAZILY inside
# load() (pdf.py, html.py, docx.py, xlsx.py, rtf.py), so the module-level
# `try/except ImportError` in each `_build_*` factory used to never fire --
# the friendly "install the extra" message was dead code, and a missing
# extra surfaced as a raw ModuleNotFoundError deep inside connector.load().
# Each factory now probes the real third-party import inside the try block.
# We simulate a missing module (never uninstall a real dependency) by making
# builtins.__import__ raise ImportError for that module's name, matching the
# pattern in test_factory.py's missing-SDK tests.
# ---------------------------------------------------------------------------


def _simulate_missing_module(monkeypatch: pytest.MonkeyPatch, missing_prefix: str) -> None:
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):  # type: ignore[no-untyped-def]
        if name == missing_prefix or name.startswith(missing_prefix + "."):
            raise ImportError(f"No module named '{missing_prefix}'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)


def test_build_pdf_missing_extra_gives_actionable_error(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _simulate_missing_module(monkeypatch, "pypdf")
    cfg = SourceConfig(name="docs", type="pdf", path=str(tmp_path))
    with pytest.raises(ImportError, match=r"pip install corpus-rag\[pdf\]"):
        build_pipeline(cfg)


def test_build_html_missing_extra_gives_actionable_error(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _simulate_missing_module(monkeypatch, "trafilatura")
    cfg = SourceConfig(name="docs", type="html", path=str(tmp_path))
    with pytest.raises(ImportError, match=r"pip install corpus-rag\[html\]"):
        build_pipeline(cfg)


def test_build_docx_missing_extra_gives_actionable_error(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _simulate_missing_module(monkeypatch, "docx")
    cfg = SourceConfig(name="docs", type="docx", path=str(tmp_path))
    with pytest.raises(ImportError, match=r"pip install corpus-rag\[docx\]"):
        build_pipeline(cfg)


def test_build_xlsx_missing_extra_gives_actionable_error(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _simulate_missing_module(monkeypatch, "openpyxl")
    cfg = SourceConfig(name="docs", type="xlsx", path=str(tmp_path))
    with pytest.raises(ImportError, match=r"pip install corpus-rag\[xlsx\]"):
        build_pipeline(cfg)


def test_build_rtf_missing_extra_gives_actionable_error(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _simulate_missing_module(monkeypatch, "striprtf")
    cfg = SourceConfig(name="docs", type="rtf", path=str(tmp_path))
    with pytest.raises(ImportError, match=r"pip install corpus-rag\[rtf\]"):
        build_pipeline(cfg)
