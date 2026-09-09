"""Tests for `corpus.cli._common.configure_logging`.

Third-party extraction libraries report per-page problems at ERROR for
conditions corpus already handles. On a real archive a single Japanese PDF
set emitted ~2,000 `pypdf._cmap` ERROR lines, which scrolled every per-source
ingest summary off the screen — the CLI's actual output became unreadable.
"""

from __future__ import annotations

import logging

import pytest

from corpus.cli._common import _NOISY_LIBRARY_LOGGERS, configure_logging


@pytest.fixture(autouse=True)
def _restore_logger_levels() -> object:
    """configure_logging mutates process-global logger state; put it back so
    test order can't matter."""
    saved = {name: logging.getLogger(name).level for name in _NOISY_LIBRARY_LOGGERS}
    root = logging.getLogger().level
    yield
    for name, level in saved.items():
        logging.getLogger(name).setLevel(level)
    logging.getLogger().setLevel(root)


@pytest.mark.parametrize("name", _NOISY_LIBRARY_LOGGERS)
def test_noisy_library_errors_are_muted_by_default(name: str) -> None:
    configure_logging(verbose=False)

    assert not logging.getLogger(name).isEnabledFor(logging.ERROR)


@pytest.mark.parametrize("name", _NOISY_LIBRARY_LOGGERS)
def test_verbose_restores_noisy_libraries(name: str) -> None:
    # --verbose exists precisely to diagnose one document's extraction, which
    # needs exactly these lines.
    configure_logging(verbose=False)
    configure_logging(verbose=True)

    assert logging.getLogger(name).isEnabledFor(logging.ERROR)


def test_corpus_own_warnings_still_reach_the_user() -> None:
    # Muting is scoped to named third-party loggers — corpus's own warnings
    # (a skipped file, a refused prune) must never be swallowed with them.
    configure_logging(verbose=False)

    assert logging.getLogger("corpus.ingester").isEnabledFor(logging.WARNING)


def test_muting_is_a_raise_not_a_silence() -> None:
    # CRITICAL, not disabled: a genuine library failure still gets through.
    configure_logging(verbose=False)

    assert logging.getLogger("pypdf._cmap").isEnabledFor(logging.CRITICAL)
