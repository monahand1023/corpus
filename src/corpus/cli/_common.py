"""Shared CLI helpers."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from corpus.config import ConfigError, CorpusConfig


def load_config_or_exit(path: Path | str | None) -> CorpusConfig:
    """Load config, or print a clean one-line error to stderr and exit(1).

    Keeps raw tracebacks (TOML parse errors, pydantic validation dumps) out of
    the user's face — CLI entrypoints should call this instead of
    ``CorpusConfig.load`` directly.
    """
    try:
        return CorpusConfig.load(path)
    except ConfigError as e:
        print(f"error: {e}", file=sys.stderr)
        raise SystemExit(1) from e


def load_python_export(path: Path, attr: str) -> Any:
    """Load `attr` from a Python file at `path`.

    Registers the module in sys.modules before exec so dataclasses that use
    `from __future__ import annotations` can resolve their string annotations
    (same fix as the eval query loader). Returns `Any` so callers can `list(...)`
    the exported value under mypy --strict.
    """
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location(f"_corpus_dyn_{attr}", str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return getattr(module, attr)


# Third-party loggers that report per-page/per-document extraction problems at
# ERROR, one line each, for conditions corpus already handles. On a real
# archive these bury the CLI's own output: a single Japanese PDF set emitted
# ~2,000 `pypdf._cmap` ERROR lines for an unimplemented CMap (the text still
# extracts, just imperfectly), which scrolled the per-source ingest summaries
# off the screen entirely.
#
# Raised to CRITICAL rather than silenced, so a genuine failure still shows.
# `--verbose` restores them in full: that is what it is for, and diagnosing a
# specific document's extraction needs exactly these lines.
_NOISY_LIBRARY_LOGGERS: tuple[str, ...] = (
    "pypdf._cmap",  # "Advanced encoding /90msp-RKSJ-H not implemented yet"
    "pypdf._page",
    "trafilatura.core",  # "discarding data: None" per empty HTML fragment
    "trafilatura.utils",  # "parsed tree length: 0, wrong data type..."
    "trafilatura.metadata",
)


def configure_logging(verbose: bool) -> None:
    """Set up CLI logging, quieting known-noisy extraction libraries.

    Every corpus CLI should call this instead of `logging.basicConfig`
    directly, so the set of muted loggers stays in one place.
    """
    import logging

    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    # Set both ways rather than only muting: the level is process-global, so
    # a bare `if not verbose` would leave a logger muted by an earlier call
    # even when this one asked for verbose. NOTSET restores inheritance from
    # the root logger, which is what these loggers had before we touched them.
    level = logging.NOTSET if verbose else logging.CRITICAL
    for name in _NOISY_LIBRARY_LOGGERS:
        logging.getLogger(name).setLevel(level)
