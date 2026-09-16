"""Resume state for the PAID batch contextualizer.

`corpus-contextualize` submits Anthropic batch jobs. If its resume state does
not load, an interrupted run re-submits work already paid for -- the cost is
not a slow re-run but a second invoice. This module had no tests at all.

The state handling is pure logic and needs no API client, which is exactly
why leaving it untested was avoidable.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

from corpus.contextual.batch_runner import BatchContextualizer


def _runner(tmp_path: Path) -> BatchContextualizer:
    return BatchContextualizer(
        store=MagicMock(),
        embedder=MagicMock(),
        client=MagicMock(),          # never called by state handling
        state_path=tmp_path / "state.json",
    )


def test_state_round_trips(tmp_path: Path) -> None:
    r = _runner(tmp_path)
    state = {"source_type": "notes", "batches": {"batch_1": {"a": ["1", "2"]}}}
    r._write_state(state)
    assert r._load_state("notes") == state


def test_state_for_a_DIFFERENT_source_is_refused(tmp_path: Path) -> None:
    """Resuming another source's batch would apply its contexts to the wrong
    chunks -- silently, and at full cost."""
    r = _runner(tmp_path)
    r._write_state({"source_type": "papers", "batches": {"b": {}}})
    assert r._load_state("notes") is None


def test_legacy_state_without_a_mapping_is_refused(tmp_path: Path) -> None:
    """Older state recorded only a batch id, with no per-window mapping, so
    its results cannot be attributed to chunks. Refusing it re-runs the work;
    accepting it would mis-apply the results."""
    r = _runner(tmp_path)
    (tmp_path / "state.json").write_text(json.dumps(
        {"source_type": "notes", "batch_id": "batch_legacy"}))
    assert r._load_state("notes") is None


def test_corrupt_state_is_refused_rather_than_raising(tmp_path: Path) -> None:
    """A half-written file from a killed process must not crash the retry."""
    r = _runner(tmp_path)
    (tmp_path / "state.json").write_text('{"source_type": "notes", "batch')
    assert r._load_state("notes") is None


def test_state_that_is_not_an_object_is_refused(tmp_path: Path) -> None:
    r = _runner(tmp_path)
    (tmp_path / "state.json").write_text("[1, 2, 3]")
    assert r._load_state("notes") is None


def test_a_missing_state_file_is_not_an_error(tmp_path: Path) -> None:
    assert _runner(tmp_path)._load_state("notes") is None


def test_no_state_path_configured_means_no_resume(tmp_path: Path) -> None:
    r = BatchContextualizer(
        store=MagicMock(), embedder=MagicMock(), client=MagicMock(),
        state_path=None,
    )
    r._write_state({"source_type": "notes", "batches": {}})   # must not raise
    assert r._load_state("notes") is None


def test_a_missing_api_key_fails_before_any_work(monkeypatch) -> None:
    """Constructed without a client and without a key, it must refuse up front
    rather than after the caller has prepared a batch."""
    import pytest

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        BatchContextualizer(store=MagicMock(), embedder=MagicMock())
