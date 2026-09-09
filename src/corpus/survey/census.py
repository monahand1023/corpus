"""Type census: what's in a tree, split into what corpus can index, what it
would ignore as known noise, and — the valuable part — the gap: extensions
present in real numbers that corpus has no connector for at all.

Read-only, streaming: walks once via `corpus.survey.walk.walk_files`,
accumulating only bounded per-bucket counters (one entry per distinct
extension/noise-pattern seen), never a file list.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from corpus.survey.classify import Category, classify_file, indexable_extension_map
from corpus.survey.walk import WalkStats, walk_files


@dataclass
class BucketStat:
    bucket: str
    detail: str  # connector name (indexable) or reason (noise/gap)
    count: int = 0
    total_bytes: int = 0


@dataclass
class CensusResult:
    root: str
    excludes: tuple[str, ...]
    use_default_excludes: bool
    files_scanned: int
    total_bytes: int
    walk_stats: WalkStats
    indexable: list[BucketStat] = field(default_factory=list)
    noise: list[BucketStat] = field(default_factory=list)
    gap: list[BucketStat] = field(default_factory=list)


def _sorted_buckets(buckets: dict[tuple[Category, str], BucketStat]) -> dict[str, list[BucketStat]]:
    by_category: dict[str, list[BucketStat]] = {"indexable": [], "noise": [], "gap": []}
    for (category, _key), stat in buckets.items():
        by_category[category].append(stat)
    for stats_list in by_category.values():
        stats_list.sort(key=lambda s: (-s.total_bytes, -s.count, s.bucket))
    return by_category


def run_census(
    root: Path,
    excludes: tuple[str, ...] = (),
    use_default_excludes: bool = True,
) -> CensusResult:
    indexable_map = indexable_extension_map()
    buckets: dict[tuple[Category, str], BucketStat] = {}
    stats = WalkStats()
    files_scanned = 0
    total_bytes = 0

    for wf in walk_files(root, excludes, use_default_excludes, stats=stats):
        c = classify_file(wf.path.name, indexable_map)
        key = (c.category, c.bucket)
        bucket = buckets.get(key)
        if bucket is None:
            bucket = BucketStat(bucket=c.bucket, detail=c.detail)
            buckets[key] = bucket
        bucket.count += 1
        bucket.total_bytes += wf.size
        files_scanned += 1
        total_bytes += wf.size

    grouped = _sorted_buckets(buckets)
    return CensusResult(
        root=str(root),
        excludes=excludes,
        use_default_excludes=use_default_excludes,
        files_scanned=files_scanned,
        total_bytes=total_bytes,
        walk_stats=stats,
        indexable=grouped["indexable"],
        noise=grouped["noise"],
        gap=grouped["gap"],
    )
