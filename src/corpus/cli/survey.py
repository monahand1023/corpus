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

from corpus.survey.archives import ArchiveInfo, ArchiveSurveyResult, run_archive_survey
from corpus.survey.census import BucketStat, CensusResult, run_census
from corpus.survey.format import human_count, human_size
from corpus.survey.index_quality import (
    IndexQualityResult,
    NotACorpusIndexError,
    run_index_quality,
)
from corpus.survey.media import (
    DEFAULT_SAMPLE_SIZE_PER_TYPE,
    MediaSurveyResult,
    TypeSurvey,
    run_media_survey,
)
from corpus.survey.overlap import (
    DEFAULT_MIN_WORDS,
    DEFAULT_SAMPLE_SIZE,
    OverlapResult,
    SampledDocument,
    run_overlap_survey,
)
from corpus.util.text_yield import estimate_tokens_from_bytes

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


def _print_bucket_table(
    title: str, rows: list[BucketStat], show_detail: bool, show_estimate: bool = False
) -> None:
    print(f"\n{title}")
    if not rows:
        print("  (none)")
        return
    header = f"  {'bucket':<24} {'count':>10} {'size':>12}"
    if show_estimate:
        header += f" {'est. tokens':>14}"
    if show_detail:
        header += "  detail"
    print(header)
    total_tokens = 0
    for r in rows:
        line = f"  {r.bucket:<24} {human_count(r.count):>10} {human_size(r.total_bytes):>12}"
        if show_estimate:
            # `r.detail` is the connector type (e.g. "pdf") for an indexable
            # bucket -- see `corpus.survey.classify` -- the same key
            # `corpus.util.text_yield` is keyed by.
            tokens = estimate_tokens_from_bytes(r.detail, r.total_bytes)
            total_tokens += tokens
            line += f" {human_count(tokens):>14}"
        if show_detail:
            line += f"  {r.detail}"
        print(line)
    if show_estimate:
        print(
            f"  -> ~{human_count(total_tokens)} estimated token(s) total. Calibrated "
            "per format from a measured real corpus, not the embedder's real "
            "tokenizer count, and rounded up rather than down when uncertain — see "
            "corpus.util.text_yield. A scanned PDF with no text layer yields close "
            "to nothing until OCR'd, for instance; treat this as a ceiling, not a "
            "quote."
        )


def _bucket_to_dict(b: BucketStat) -> dict[str, Any]:
    return {"bucket": b.bucket, "detail": b.detail, "count": b.count, "total_bytes": b.total_bytes}


def _indexable_bucket_to_dict(b: BucketStat) -> dict[str, Any]:
    return {**_bucket_to_dict(b), "estimated_tokens": estimate_tokens_from_bytes(b.detail, b.total_bytes)}


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
        "indexable": [_indexable_bucket_to_dict(b) for b in result.indexable],
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
    _print_bucket_table(
        _DESCRIPTION_BY_CATEGORY["indexable"], result.indexable, show_detail=True, show_estimate=True
    )
    _print_bucket_table(_DESCRIPTION_BY_CATEGORY["noise"], result.noise, show_detail=True)
    return 0


def _archive_to_dict(a: ArchiveInfo) -> dict[str, Any]:
    return {
        "path": a.path,
        "readable": a.readable,
        "error": a.error,
        "total_members": a.total_members,
        "total_declared_bytes": a.total_declared_bytes,
        "indexable_by_type": dict(a.indexable_by_type),
        "indexable_total": a.indexable_total,
        "packaging_noise": a.packaging_noise,
        "dependency_noise": a.dependency_noise,
        "encrypted": a.encrypted,
        "nested_archive_refused": a.nested_archive_refused,
        "gap": a.gap,
        "noise_ratio": a.noise_ratio,
    }


def _archive_result_to_dict(result: ArchiveSurveyResult) -> dict[str, Any]:
    return {
        "root": result.root,
        "excludes": list(result.excludes),
        "default_excludes_applied": result.use_default_excludes,
        "follows_symlinks": False,
        "archives": [_archive_to_dict(a) for a in result.archives],
        "totals": result.totals,
        "walk": {
            "permission_errors": result.walk_stats.permission_errors,
            "stat_errors": result.walk_stats.stat_errors,
        },
    }


def _run_archives(args: argparse.Namespace) -> int:
    root = Path(args.path)
    if not root.is_dir():
        print(f"error: not a directory: {root}", file=sys.stderr)
        return 1

    result = run_archive_survey(
        root,
        excludes=tuple(args.excludes),
        use_default_excludes=not args.no_default_excludes,
    )

    if args.as_json:
        print(json.dumps(_archive_result_to_dict(result), indent=2))
        return 0

    print(f"corpus-survey archives: {result.root}")
    print("Archives are opened read-only; nothing is ever extracted to disk.")
    t = result.totals
    print(
        f"{human_count(t['archive_count'])} archive(s), "
        f"{human_count(t['unreadable_count'])} unreadable, "
        f"{human_count(t['total_members'])} member(s), "
        f"{human_size(t['total_declared_bytes'])} declared."
    )
    print(
        f"  indexable: {human_count(t['indexable_total'])}   "
        f"gap (no connector): {human_count(t['gap'])}   "
        f"packaging noise: {human_count(t['packaging_noise'])}   "
        f"dependency/build noise: {human_count(t['dependency_noise'])}"
    )
    print(
        f"  encrypted (skipped): {human_count(t['encrypted'])}   "
        f"nested archives (refused): {human_count(t['nested_archive_refused'])}"
    )
    print(f"  overall noise ratio (packaging + dependency / total members): {t['overall_noise_ratio']:.0%}")

    print(
        f"\n  {'archive':<40} {'members':>8} {'noise%':>7} {'indexable':>10} {'gap':>6}  detail"
    )
    for a in result.archives:
        if not a.readable:
            print(f"  {a.path:<40} {'—':>8}  UNREADABLE: {a.error}")
            continue
        detail = ", ".join(f"{k}={v}" for k, v in sorted(a.indexable_by_type.items()))
        print(
            f"  {a.path:<40} {human_count(a.total_members):>8} "
            f"{a.noise_ratio:>6.0%} {human_count(a.indexable_total):>10} "
            f"{human_count(a.gap):>6}  {detail}"
        )
    return 0


def _type_survey_to_dict(t: TypeSurvey) -> dict[str, Any]:
    return {
        "extension": t.extension,
        "count": t.count,
        "total_bytes": t.total_bytes,
        "sample_size": t.sample_size,
        "probed_ok": t.probed_ok,
        "probe_failed": t.probe_failed,
        "mean_duration_seconds": t.mean_duration_seconds,
        "stdev_duration_seconds": t.stdev_duration_seconds,
        "estimated_total_seconds": t.estimated_total_seconds,
    }


def _media_result_to_dict(result: MediaSurveyResult) -> dict[str, Any]:
    return {
        "root": result.root,
        "excludes": list(result.excludes),
        "default_excludes_applied": result.use_default_excludes,
        "follows_symlinks": False,
        "ffprobe_available": result.ffprobe_available,
        "rate": result.rate,
        "total_files": result.total_files,
        "total_bytes": result.total_bytes,
        "total_sampled": result.total_sampled,
        "total_probed_ok": result.total_probed_ok,
        "estimated_total_hours": result.estimated_total_hours,
        "estimated_processing_hours": result.estimated_processing_hours,
        "types": [_type_survey_to_dict(t) for t in result.types],
        "method": (
            "Reservoir-sampled up to N files per extension (stratified, not pooled "
            "across types — durations vary too much by type to pool honestly), probed "
            "each sampled file with ffprobe, and extrapolated that type's mean "
            "duration across its full file count. This is a linear extrapolation, "
            "not a census: true duration may differ, especially if a type's file "
            "lengths are skewed (many short clips plus a few long recordings). "
            "Increase --sample-size for a tighter estimate."
        ),
    }


def _run_media(args: argparse.Namespace) -> int:
    root = Path(args.path)
    if not root.is_dir():
        print(f"error: not a directory: {root}", file=sys.stderr)
        return 1

    result = run_media_survey(
        root,
        excludes=tuple(args.excludes),
        use_default_excludes=not args.no_default_excludes,
        sample_size_per_type=args.sample_size,
        rate=args.rate,
    )

    if args.as_json:
        print(json.dumps(_media_result_to_dict(result), indent=2))
        return 0

    print(f"corpus-survey media: {result.root}")
    print(f"{human_count(result.total_files)} audio/video file(s), {human_size(result.total_bytes)}.")
    if not result.ffprobe_available:
        print(
            "  ffprobe not found on PATH — duration unavailable. Install ffmpeg "
            "(e.g. `brew install ffmpeg`) to enable duration sampling."
        )
    else:
        print(
            f"  sampled {human_count(result.total_sampled)} file(s) "
            f"({human_count(result.total_probed_ok)} probed successfully) — "
            "this is an extrapolation, not a census."
        )

    print(f"\n  {'ext':<10} {'files':>8} {'size':>10} {'sampled':>8} {'mean dur':>12} {'est. hours':>11}")
    for t in result.types:
        mean = f"{t.mean_duration_seconds:.0f}s" if t.mean_duration_seconds is not None else "—"
        est_hours = f"{t.estimated_total_seconds / 3600.0:.1f}" if t.estimated_total_seconds else "—"
        print(
            f"  {t.extension:<10} {human_count(t.count):>8} {human_size(t.total_bytes):>10} "
            f"{human_count(t.sample_size):>8} {mean:>12} {est_hours:>11}"
        )

    if result.estimated_total_hours is not None:
        print(f"\nEstimated total: {result.estimated_total_hours:.1f} hours (sample-based extrapolation).")
        if result.estimated_processing_hours is not None:
            print(
                f"At {result.rate:g}x realtime: ~{result.estimated_processing_hours:.1f} hours "
                f"({result.estimated_processing_hours / 24:.1f} days) to process."
            )
    elif result.total_files:
        print("\nNo total estimate: at least one media type has files but zero successfully-probed samples.")
    return 0


def _sampled_doc_to_dict(s: SampledDocument) -> dict[str, Any]:
    return {"path": s.rel_path, "phrase": s.phrase, "matched": s.matched, "matched_source": s.matched_source}


_OVERLAP_METHOD = (
    "Reservoir-sampled up to N plain-text-decodable documents (.txt/.md/.markdown/"
    ".rst/.log/.csv/.tsv/.json/.yaml/.yml — binary formats like PDF/DOCX are not "
    "sampled by this version; see corpus-survey census for their share of the tree), "
    "extracted one distinctive phrase per document (the longest line with at least "
    "--min-words words), and checked each phrase against the target database: an "
    "FTS5 BM25 search for candidate chunks, then a literal case-insensitive "
    "substring confirmation against each candidate's content (not just shared "
    "vocabulary). estimated_overlap_fraction is matched/sampled with a 95% Wilson "
    "confidence interval. Caveats: small samples carry wide intervals — increase "
    "--sample-size for a tighter one; a match means this exact phrase text is "
    "present somewhere in the target database, not that the whole document is "
    "identical or unchanged; content reformatted or edited since it was indexed "
    "may be undercounted as 'not found'."
)


def _overlap_result_to_dict(result: OverlapResult) -> dict[str, Any]:
    ci = result.confidence_interval_95
    return {
        "root": result.root,
        "db_path": result.db_path,
        "excludes": list(result.excludes),
        "default_excludes_applied": result.use_default_excludes,
        "follows_symlinks": False,
        "eligible_document_count": result.eligible_document_count,
        "sample_size": result.sample_size,
        "matched_count": result.matched_count,
        "estimated_overlap_fraction": result.estimated_overlap_fraction,
        "confidence_interval_95": list(ci) if ci is not None else None,
        "sample": [_sampled_doc_to_dict(s) for s in result.sample],
        "method": _OVERLAP_METHOD,
    }


def _print_index_quality(result: IndexQualityResult) -> None:
    print(f"scanned {human_count(result.scanned_chunks)} chunks")
    if result.clean:
        print("no transcription artefacts found")
        return

    print(
        f"\naffected: {human_count(result.affected_chunks)} chunk(s) across "
        f"{human_count(len(result.documents_affected))} document(s)"
    )
    print(f"  entirely caption boilerplate : {len(result.whole_chunk)}")
    print(f"  sign-off glued to real speech: {len(result.tails)}")

    if result.by_source_type:
        print("\nby source type:")
        for source_type, count in result.by_source_type.most_common():
            print(f"  {source_type:24} {human_count(count)}")

    shown = [f for f in result.whole_chunk if f.before]
    if shown:
        print("\nchunks that are ONLY a sign-off (these should never have been indexed):")
        for finding in shown:
            print(f"  [{finding.source_type}] {finding.before!r}")

    shown = [f for f in result.tails if f.before]
    if shown:
        print("\nsign-off glued to real speech (cut the tail, keep the speech):")
        for finding in shown:
            print(f"  [{finding.source_type}]")
            print(f"    ...{finding.before!r}")
            print(f"    -> ...{finding.after!r}")

    print(
        "\nBoth are fixed at INGEST, not by editing the database: apply "
        "corpus.transcripts.strip_caption_tail in the connector or chunker and "
        "re-run ingest. Content-hash comparison re-embeds only what changed."
    )


def _run_index_quality(args: argparse.Namespace) -> int:
    db_path = Path(args.db)
    if not db_path.exists():
        print(f"error: database not found: {db_path}", file=sys.stderr)
        return 1

    try:
        result = run_index_quality(
            db_path,
            source_types=tuple(args.source_type),
            sample_per_kind=args.samples,
        )
    except NotACorpusIndexError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.as_json:
        print(json.dumps({
            "total_chunks": result.total_chunks,
            "scanned_chunks": result.scanned_chunks,
            "affected_chunks": result.affected_chunks,
            "documents_affected": len(result.documents_affected),
            "whole_chunk_boilerplate": len(result.whole_chunk),
            "sign_off_tails": len(result.tails),
            "by_source_type": dict(result.by_source_type),
        }, indent=2))
    else:
        _print_index_quality(result)

    # A chunk that is ENTIRELY a sign-off is an unambiguous filtering defect,
    # so it fails. A glued tail is a known-imperfect state that a re-ingest
    # fixes, so it reports without failing a build over it.
    return 1 if result.whole_chunk else 0


def _run_overlap(args: argparse.Namespace) -> int:
    root = Path(args.path)
    if not root.is_dir():
        print(f"error: not a directory: {root}", file=sys.stderr)
        return 1
    db_path = Path(args.db)
    if not db_path.exists():
        print(f"error: database not found: {db_path}", file=sys.stderr)
        return 1

    try:
        result = run_overlap_survey(
            root,
            db_path,
            excludes=tuple(args.excludes),
            use_default_excludes=not args.no_default_excludes,
            sample_size=args.sample_size,
            min_words=args.min_words,
        )
    except OSError as e:
        print(f"error: could not open database {db_path}: {e}", file=sys.stderr)
        return 1

    if args.as_json:
        print(json.dumps(_overlap_result_to_dict(result), indent=2))
        return 0

    print(f"corpus-survey overlap: {result.root}  vs.  {result.db_path}")
    print(
        f"{human_count(result.eligible_document_count)} eligible plain-text document(s) found "
        "(binary formats like PDF/DOCX are not sampled — see corpus-survey census)."
    )
    if not result.sample:
        print("No eligible documents to sample — no estimate.")
        return 0

    print(f"Sampled {human_count(result.sample_size)}, {human_count(result.matched_count)} matched.")
    frac = result.estimated_overlap_fraction
    ci = result.confidence_interval_95
    if frac is not None and ci is not None:
        print(f"Estimated overlap: {frac:.0%}  (95% CI: {ci[0]:.0%}–{ci[1]:.0%})")
    print(
        "\nMethod: FTS5 candidate recall + literal substring confirmation of one "
        "distinctive phrase per sampled document. This is an estimate, not a count "
        "— small samples carry wide confidence intervals; increase --sample-size "
        "for a tighter one. A match means the exact phrase text was found "
        "somewhere in the target database, not that the whole document is "
        "identical; reformatted/edited content may be undercounted as 'not found'."
    )
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

    p_archives = sub.add_parser(
        "archives", help="Zip archive inspection: contents, noise ratio, without extracting"
    )
    _add_common_tree_args(p_archives)
    p_archives.set_defaults(func=_run_archives)

    p_media = sub.add_parser(
        "media", help="Audio/video duration survey: hours, not file count"
    )
    _add_common_tree_args(p_media)
    p_media.add_argument(
        "--sample-size",
        type=int,
        default=DEFAULT_SAMPLE_SIZE_PER_TYPE,
        help=f"Files to probe per extension (default {DEFAULT_SAMPLE_SIZE_PER_TYPE})",
    )
    p_media.add_argument(
        "--rate",
        type=float,
        default=None,
        metavar="X",
        help="Processing rate as a multiple of realtime (e.g. 15 for 15x) — "
        "projects processing time from the estimated total hours",
    )
    p_media.set_defaults(func=_run_media)

    p_quality = sub.add_parser(
        "index-quality",
        help="Scan an EXISTING index for transcription artefacts that got indexed",
    )
    p_quality.add_argument(
        "--db", required=True, metavar="PATH", help="Path to the corpus SQLite database"
    )
    p_quality.add_argument(
        "--source-type",
        action="append",
        default=[],
        metavar="NAME",
        help="Limit the scan to this source type (repeatable)",
    )
    p_quality.add_argument(
        "--samples", type=int, default=8, help="Example findings to show per kind"
    )
    p_quality.add_argument(
        "--json", action="store_true", dest="as_json", help="Emit the result as JSON"
    )
    p_quality.set_defaults(func=_run_index_quality)

    p_overlap = sub.add_parser(
        "overlap", help="Estimate how much of a directory is already indexed elsewhere"
    )
    _add_common_tree_args(p_overlap)
    p_overlap.add_argument(
        "--db", required=True, metavar="PATH", help="Path to the existing corpus SQLite database"
    )
    p_overlap.add_argument(
        "--sample-size",
        type=int,
        default=DEFAULT_SAMPLE_SIZE,
        help=f"Documents to sample (default {DEFAULT_SAMPLE_SIZE})",
    )
    p_overlap.add_argument(
        "--min-words",
        type=int,
        default=DEFAULT_MIN_WORDS,
        help=f"Minimum words for a line to count as a distinctive phrase (default {DEFAULT_MIN_WORDS})",
    )
    p_overlap.set_defaults(func=_run_overlap)

    p_dupes = sub.add_parser(
        "duplicates",
        help="Documents wholly duplicated elsewhere, paired so only one goes",
    )
    p_dupes.add_argument("--db", required=True, metavar="PATH")
    p_dupes.add_argument(
        "--top", type=int, default=25, help="How many pairs to print (default 25)"
    )
    p_dupes.add_argument(
        "--excludes-for",
        default=None,
        metavar="SOURCE",
        help=(
            "Print the paths to drop as an `exclude` list for this source, "
            "ready to paste into corpus.toml"
        ),
    )
    p_dupes.set_defaults(func=_run_duplicates)

    return parser


def _run_duplicates(args: argparse.Namespace) -> int:
    from corpus.survey.duplicates import duplicate_documents, find_duplicate_content

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"error: database not found: {db_path}", file=sys.stderr)
        return 1

    passages = find_duplicate_content(db_path, top=0)
    pairs = duplicate_documents(db_path)
    removable = sum(p.shared for p in pairs)

    print(f"duplicated passages : {passages.redundant_chunks:,} of "
          f"{passages.total_chunks:,} chunks ({passages.percent:.1f}%)")
    print(f"removable documents : {len(pairs):,} holding {removable:,} chunks "
          f"({100.0 * removable / max(passages.total_chunks, 1):.1f}%)")
    print()
    print("A document is removable only when EVERY one of its passages also")
    print("appears under another document. Duplicates come in pairs and both")
    print("members qualify, so exactly one of each is listed here -- dropping")
    print("the whole set would delete the content, not deduplicate it.")
    print("The rest of the duplicated passages are partial overlaps: two")
    print("versions sharing most of their text and differing where it counts.")

    if args.excludes_for:
        print(f"\n# paste into the [[sources]] block named {args.excludes_for!r}")
        print("exclude = [")
        for pair in pairs:
            print(f'  "{pair.drop}",')
        print("]")
        return 0

    print(f"\n{'chunks':>7}  drop / keep")
    for pair in pairs[: args.top]:
        print(f"{pair.shared:7,}  - {pair.drop}")
        print(f"{'':7}  + {pair.keep}")
    if len(pairs) > args.top:
        print(f"\n  ... {len(pairs) - args.top:,} more (--top N)")
    return 0


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
