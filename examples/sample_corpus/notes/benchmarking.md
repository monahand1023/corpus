---
title: "Benchmarking latency"
id: benchmarking
---

# Benchmarking latency

`corpus-benchmark` measures per-stage retrieval latency: embed, vector search, FTS search, fusion, and dedupe/diversity, each timed separately across many repeated queries. It reports mean, p50, p95, and p99 latency per stage, plus an overall throughput (queries per second), so you can see exactly where time goes instead of just one end-to-end number.

On a typical few-thousand-chunk corpus on an M-series Mac, `embed` dominates at 100-300ms because it's a network round trip to the provider, while vector search, FTS search, fusion, and dedupe are collectively under about 5ms. The optimization lever is almost always "fewer or concurrent embed calls," not "faster SQLite."

`--compare provider1 provider2` A/B's embed-only latency between two embedder providers, since full retrieval quality can't be fairly compared against one DB. Use `--queries path.py` against your own eval query set, and `--json out.json` for machine-readable output.
