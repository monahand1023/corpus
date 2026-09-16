"""corpus-summarize reports what a paid run cost, and got it wrong.

The number this command prints is the only thing a user has to decide
whether to run it again, or on a bigger source. Two defects made it
unreliable, and both were in the direction that encourages spending:

  - The cost formula treated `cache_read_input_tokens` as a SUBSET of
    `input_tokens` and subtracted a discount for it. The Anthropic API
    reports the two as DISJOINT counts -- `input_tokens` is already the
    uncached portion -- so every cached token was scored as a 90-cent
    refund per million instead of a 10-cent charge.
  - The `--dry-run` estimate ignored `MAX_INPUT_CHARS`, the 80,000-char
    truncation the summarizer actually applies, and ignored the per-call
    prompt sent with every document.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from corpus.cli.summarize import (
    PRICE_CACHED,
    PRICE_INPUT,
    PRICE_OUTPUT,
    estimate_doc_tokens,
    run_cost,
)
from corpus.summarizer.anthropic_summarizer import MAX_INPUT_CHARS

# --- what a run cost ---------------------------------------------------------


def test_a_cache_hit_is_a_charge_not_a_refund():
    """Cached tokens are billed, at ~10%. The old formula subtracted 90% of
    the base rate for each one, so a run with enough cache hits reported a
    NEGATIVE cost."""
    assert run_cost(input_tokens=0, output_tokens=0, cached_tokens=1_000_000) > 0


def test_caching_never_reports_a_run_as_costing_less_than_its_uncached_part():
    uncached_only = run_cost(input_tokens=1000, output_tokens=200, cached_tokens=0)
    with_cache = run_cost(input_tokens=1000, output_tokens=200, cached_tokens=500_000)
    assert with_cache > uncached_only


def test_cost_prices_each_disjoint_token_count_at_its_own_rate():
    cost = run_cost(input_tokens=1_000_000, output_tokens=1_000_000, cached_tokens=1_000_000)
    expected = 1_000_000 * (PRICE_INPUT + PRICE_OUTPUT + PRICE_CACHED)
    assert cost == pytest.approx(expected)


def test_caching_is_still_cheaper_than_not_caching():
    """The whole point of the cache. Same total input, split two ways."""
    all_fresh = run_cost(input_tokens=1_000_000, output_tokens=0, cached_tokens=0)
    mostly_cached = run_cost(input_tokens=100_000, output_tokens=0, cached_tokens=900_000)
    assert mostly_cached < all_fresh


# --- what a run will cost ----------------------------------------------------


def test_the_estimate_stops_at_the_truncation_the_summarizer_applies():
    """A 5 MB document is not a 1.25-million-token request: `summarize()`
    sends `content[:MAX_INPUT_CHARS]`. Estimating the untruncated length
    priced documents that will never be sent in full."""
    huge, _ = estimate_doc_tokens(body_chars=5_000_000)
    capped, _ = estimate_doc_tokens(body_chars=MAX_INPUT_CHARS)
    assert huge == capped


def test_the_estimate_includes_the_prompt_sent_with_every_document():
    """An archive of many small documents is mostly prompt. Counting only
    the body understated those runs by the fixed overhead times the doc
    count."""
    body_only = 100 // 4
    assert estimate_doc_tokens(body_chars=100)[0] > body_only


def test_an_empty_document_still_costs_a_call():
    assert estimate_doc_tokens(body_chars=0)[0] > 0


# --- the dry run as a whole --------------------------------------------------


def _store_with(tmp_path, sources: dict[str, int], *, stale: bool = False) -> Path:
    """A real store with `n` single-chunk documents under each source name."""
    import math

    from corpus.db.sqlite import ChunkStore
    from corpus.types import Chunk, ChunkKind, ChunkMetadata
    from corpus.util.hash import chunk_id, sha256

    db = tmp_path / "s.db"
    store = ChunkStore(db, embedding_dim=64)
    rows = []
    for name, n in sources.items():
        for i in range(n):
            key = f"{name}-doc{i}"
            rows.append((
                Chunk(
                    id=chunk_id(name, key, ChunkKind.BODY, 0),
                    content="body " * 200,
                    content_hash=sha256(key),
                    metadata=ChunkMetadata(
                        source_type=name, source_key=key,
                        chunk_kind=ChunkKind.BODY, chunk_index=0, title=key,
                    ),
                ),
                [math.sin((i + 1) * (j + 1) * 0.001) for j in range(64)],
            ))
    store.upsert_batch(rows)
    if stale:
        store._conn.execute("DELETE FROM schema_meta WHERE key = 'fts_version'")
        store._conn.commit()
    store.close()
    return db


def _config_for(db: Path, names: list[str]):
    from corpus.config import CorpusConfig, EmbedderConfig, SourceConfig

    return CorpusConfig(
        db_path=db,
        embedder=EmbedderConfig(provider="hash", dim=64),
        sources=[SourceConfig(name=n, type="markdown", path=".") for n in names],
    )


def _dry_run(monkeypatch, cfg, argv=("corpus-summarize", "--all", "--dry-run")):
    import corpus.cli.summarize as mod

    monkeypatch.setattr(mod, "load_config_or_exit", lambda _p: cfg)
    monkeypatch.setattr("sys.argv", list(argv))
    return mod.main()


def test_dry_run_sums_every_source_it_priced(tmp_path, monkeypatch, capsys):
    """`--all --dry-run` across an archive's sources printed one number per
    source and no sum, on the one path whose whole job is answering "how
    much before I spend it"."""
    db = _store_with(tmp_path, {"notes": 3, "papers": 2})

    assert _dry_run(monkeypatch, _config_for(db, ["notes", "papers"])) == 0
    out = capsys.readouterr().out

    def _summed(label: str) -> int:
        return sum(
            int(ln.split(":")[1].strip().replace(",", ""))
            for ln in out.splitlines()
            if ln.strip().startswith(label)
        )

    per_source_costs = [ln for ln in out.splitlines() if "est. cost:" in ln]
    assert len(per_source_costs) == 2, "each source should be priced separately"

    total_line = next(ln for ln in out.splitlines() if "TOTAL (estimated)" in ln)
    assert "5 docs" in total_line
    # Compared in tokens, not printed dollars: five small documents round to
    # $0.00 apiece, so a dollar comparison would pass on any arithmetic.
    expected = run_cost(
        input_tokens=_summed("est. input tokens"),
        output_tokens=_summed("est. output tokens"),
        cached_tokens=0,
    )
    assert float(total_line.split("$")[1]) == pytest.approx(expected, abs=0.005)
    assert _summed("est. input tokens") > 0


def test_dry_run_on_a_missing_database_does_not_create_one(tmp_path, monkeypatch):
    """Otherwise a typo in db_path reports "0 docs, $0.00" -- read as
    "already done", not as "wrong path"."""
    db = tmp_path / "absent.db"
    with pytest.raises(SystemExit):
        _dry_run(monkeypatch, _config_for(db, ["notes"]))
    assert not db.exists()


def test_dry_run_does_not_rebuild_a_stale_index(tmp_path, monkeypatch):
    from corpus.db.sqlite import ChunkStore

    db = _store_with(tmp_path, {"notes": 2}, stale=True)
    assert _dry_run(monkeypatch, _config_for(db, ["notes"])) == 0

    ro = ChunkStore(db, embedding_dim=64, read_only=True)
    try:
        assert ro.fts_version() is None, "a cost estimate rewrote the database"
    finally:
        ro.close()
