from __future__ import annotations

from corpus.cli.benchmark import _percentile, _summary


def test_percentile_single_value() -> None:
    assert _percentile([1.0], 50) == 1.0
    assert _percentile([1.0], 99) == 1.0


def test_percentile_two_values() -> None:
    # Linear interpolation between 1.0 (index 0) and 2.0 (index 1):
    # p0 -> 1.0, p100 -> 2.0, p50 -> 1.5
    vals = [1.0, 2.0]
    assert _percentile(vals, 0) == 1.0
    assert _percentile(vals, 100) == 2.0
    assert _percentile(vals, 50) == 1.5


def test_percentile_ten_values() -> None:
    vals = [float(i) for i in range(1, 11)]  # 1..10
    # p50 of 1..10 is 5.5 (linear interpolation)
    assert _percentile(vals, 50) == 5.5
    # p99 is very close to the max
    assert _percentile(vals, 99) > 9.5
    # p0 is the min
    assert _percentile(vals, 0) == 1.0


def test_percentile_empty() -> None:
    assert _percentile([], 50) == 0.0


def test_summary_rounds_to_2dp() -> None:
    s = _summary([0.001, 0.002, 0.003, 0.004, 0.005])
    # All values converted from seconds to ms with 2 decimal places
    assert s["min_ms"] == 1.0
    assert s["max_ms"] == 5.0
    assert s["mean_ms"] == 3.0


def test_summary_empty() -> None:
    s = _summary([])
    assert s["mean_ms"] == 0.0
    assert s["p50_ms"] == 0.0
    assert s["max_ms"] == 0.0


def test_benchmark_embed_only_compare(monkeypatch) -> None:
    """The --compare path: _benchmark_embed_only measures embed latency per
    provider without touching the DB or running the full pipeline."""
    from unittest.mock import MagicMock, patch

    from corpus.cli.benchmark import _benchmark_embed_only

    fake_embedder = MagicMock()
    fake_embedder.embed_query = MagicMock(return_value=[0.0] * 1024)

    with patch("corpus.cli.benchmark.make_embedder", return_value=fake_embedder):
        report = _benchmark_embed_only(
            provider="voyage",
            model="voyage-3-large",
            dim=1024,
            queries=["q1", "q2", "q3"],
            runs_per_query=4,
        )

    assert report["provider"] == "voyage"
    assert report["mode"] == "embed-only"
    assert report["total_calls"] == 12  # 3 queries × 4 runs
    assert "embed_latency" in report
    assert report["embed_latency"]["mean_ms"] >= 0.0
    # embed_query was called exactly total_calls times
    assert fake_embedder.embed_query.call_count == 12


def test_default_dim_for() -> None:
    from corpus.cli.benchmark import _default_dim_for

    assert _default_dim_for("voyage") == 1024
    assert _default_dim_for("gemini") == 1536
    assert _default_dim_for("unknown") == 1024  # safe fallback
