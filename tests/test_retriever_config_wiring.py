"""Settings that corpus.toml documents must actually reach the retriever.

`max_per_source_type` is documented in three places -- configuration.md with a
full explanation of the semantics, the README, and corpus.toml.example -- as a
user-settable diversity cap. Nothing read it. Setting it to 10, or to null to
disable the cap, silently got the hardcoded default of 3.

A setting that does nothing is worse than a missing one: the user believes
they have tuned something, and every conclusion they draw afterwards rests on
a control that was never connected.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from corpus.config import CorpusConfig, RetrieverConfig


def _config(tmp_path: Path, **retriever) -> CorpusConfig:
    return CorpusConfig(
        db_path=tmp_path / "c.db", retriever=RetrieverConfig(**retriever)
    )


@pytest.mark.parametrize("configured", [1, 7, None])
def test_the_query_cli_passes_the_configured_diversity_cap(
    tmp_path, monkeypatch, configured
) -> None:
    import corpus.cli.query as mod

    captured: dict[str, object] = {}

    class _Retriever:
        def __init__(self, *a, **k) -> None:
            pass

        def query(self, *a, **k):
            captured.update(k)
            return MagicMock(chunks=[])

        def close(self) -> None:
            pass

    monkeypatch.setattr(mod, "Retriever", _Retriever)
    monkeypatch.setattr(mod, "load_config_or_exit",
                        lambda _p: _config(tmp_path, max_per_source_type=configured))
    monkeypatch.setattr(mod, "make_embedder", lambda **k: MagicMock())
    monkeypatch.setattr(
        mod, "ChunkStore", SimpleNamespace(from_config=lambda *a, **k: MagicMock())
    )
    monkeypatch.setattr("sys.argv", ["corpus-query", "a question"])

    mod.main()

    assert "max_per_source_type" in captured, (
        "the configured diversity cap never reached the retriever"
    )
    assert captured["max_per_source_type"] == configured


def test_the_mcp_server_passes_the_configured_diversity_cap(
    tmp_path, monkeypatch
) -> None:
    """The path that actually serves Claude, and the one that mattered most.

    It called `retriever.query(query, top_k, filter_sources)` positionally, so
    the cap could not be passed at all -- every MCP search used the hardcoded
    default however corpus.toml was written.
    """
    import corpus.mcp_server as mod

    captured: dict[str, object] = {}

    class _Retriever:
        def query(self, *a, **k):
            captured.update(k)
            return MagicMock(chunks=[])

    cfg = _config(tmp_path, max_per_source_type=9)
    store = MagicMock()
    store.stats.return_value = {"by_source": {}}
    # `_init` is how this module hands out its store/retriever/config, and how
    # the existing MCP tests stub them.
    monkeypatch.setattr(
        mod, "_init", lambda: (store, MagicMock(), _Retriever(), cfg)
    )
    monkeypatch.setattr(mod, "_record", lambda *a, **k: None)

    # pytest-asyncio is not installed; this repo drives coroutines directly.
    asyncio.run(mod.search_knowledge("a question", top_k=5))

    assert captured.get("max_per_source_type") == 9, captured
