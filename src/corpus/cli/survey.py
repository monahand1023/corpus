"""corpus-survey: read-only reconnaissance for deciding what to index.

    corpus-survey census PATH [--exclude PATTERN ...] [--json]
    corpus-survey archives PATH [--exclude PATTERN ...] [--json]
    corpus-survey media PATH [--rate 15] [--sample-size N] [--json]
    corpus-survey overlap PATH --db corpus.db [--sample-size N] [--json]

Every subcommand is read-only: nothing here ever writes to a database,
extracts to a non-temporary location, or modifies the tree being surveyed.
None of it follows symlinks, matching corpus's own ingestion discovery.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from corpus.survey.census import BucketStat, CensusResult, run_census
from corpus.survey.format import human_count, human_size

_DESCRIPTION_BY_CATEGORY = {
    "indexable": "Indexable (corpus has a connector)",
    "gap": "Gap — no connector (the interesting part)",
    "noise": "Known noise (would be ignored)",
}


def _add_common_tree_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("path", help="Directory to survey")
    p.add_argument(
        "--exclude",
        action="append",
        default=[],
        dest="excludes",
        metavar="PATTERN",
        help="fnmatch pattern to exclude (repeatable), e.g. --exclude '*.log'",
    )
    p.add_argument(
        "--no-default-excludes",
        action="store_true",
        help="Don't prune common noise dirs (node_modules, .git, __pycache__, "
        "*.photoslibrary, ...) — see corpus.survey.walk.DEFAULT_EXCLUDED_DIR_NAMES",
    )
    p.add_argument("--json", action="store_true", dest="as_json", help="Emit machine-readable JSON")


def _print_bucket_table(title: str, rows: list[BucketStat], show_detail: bool) -> None:
    print(f"\n{title}")
    if not rows:
        print("  (none)")
        return
    header = f"  {'bucket':<24} {'count':>10} {'size':>12}"
    if show_detail:
        header += "  detail"
    print(header)
    for r in rows:
        line = f"  {r.bucket:<24} {human_count(r.count):>10} {human_size(r.total_bytes):>12}"
        if show_detail:
            line += f"  {r.detail}"
        print(line)


def _bucket_to_dict(b: BucketStat) -> dict[str, Any]:
    return {"bucket": b.bucket, "detail": b.detail, "count": b.count, "total_bytes": b.total_bytes}


def _census_result_to_dict(result: CensusResult) -> dict[str, Any]:
    return {
        "root": result.root,
        "excludes": list(result.excludes),
        "default_excludes_applied": result.use_default_excludes,
        "follows_symlinks": False,
        "files_scanned": result.files_scanned,
        "total_bytes": result.total_bytes,
        "walk": {
            "dirs_visited": result.walk_stats.dirs_visited,
            "dirs_pruned": result.walk_stats.dirs_pruned,
            "dir_symlinks_skipped": result.walk_stats.dir_symlinks_skipped,
            "file_symlinks_skipped": result.walk_stats.file_symlinks_skipped,
            "permission_errors": result.walk_stats.permission_errors,
            "stat_errors": result.walk_stats.stat_errors,
            "files_excluded": result.walk_stats.files_excluded,
        },
        "indexable": [_bucket_to_dict(b) for b in result.indexable],
        "gap": [_bucket_to_dict(b) for b in result.gap],
        "noise": [_bucket_to_dict(b) for b in result.noise],
    }


def _run_census(args: argparse.Namespace) -> int:
    root = Path(args.path)
    if not root.is_dir():
        print(f"error: not a directory: {root}", file=sys.stderr)
        return 1

    result = run_census(
        root,
        excludes=tuple(args.excludes),
        use_default_excludes=not args.no_default_excludes,
    )

    if args.as_json:
        print(json.dumps(_census_result_to_dict(result), indent=2))
        return 0

    print(f"corpus-survey census: {result.root}")
    print("Symlinks are not followed (matches corpus's own ingestion discovery).")
    print(
        f"Scanned {human_count(result.files_scanned)} files, "
        f"{human_size(result.total_bytes)} total."
    )
    ws = result.walk_stats
    error_bits = []
    if ws.permission_errors:
        error_bits.append(f"{ws.permission_errors} permission error(s)")
    if ws.stat_errors:
        error_bits.append(f"{ws.stat_errors} unreadable/broken path(s)")
    if ws.dir_symlinks_skipped or ws.file_symlinks_skipped:
        error_bits.append(
            f"{ws.dir_symlinks_skipped + ws.file_symlinks_skipped} symlink(s) not followed"
        )
    if ws.dirs_pruned:
        error_bits.append(f"{ws.dirs_pruned} dir(s) pruned by exclude patterns")
    if error_bits:
        print("  " + "; ".join(error_bits))

    _print_bucket_table(_DESCRIPTION_BY_CATEGORY["gap"], result.gap, show_detail=False)
    _print_bucket_table(_DESCRIPTION_BY_CATEGORY["indexable"], result.indexable, show_detail=True)
    _print_bucket_table(_DESCRIPTION_BY_CATEGORY["noise"], result.noise, show_detail=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="corpus-survey", description="Read-only reconnaissance for deciding what to index"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_census = sub.add_parser(
        "census", help="File-type census: indexable vs. gap vs. known noise"
    )
    _add_common_tree_args(p_census)
    p_census.set_defaults(func=_run_census)

    return parser


def main_argv(argv: list[str]) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    func: Any = args.func
    result: int = func(args)
    return result


def main() -> int:
    return main_argv(sys.argv[1:])


if __name__ == "__main__":
    sys.exit(main())
