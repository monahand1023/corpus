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
