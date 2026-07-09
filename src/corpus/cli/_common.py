"""Shared CLI helpers."""

from __future__ import annotations

import sys
from pathlib import Path

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
