"""corpus-index: point at a directory, get a fully indexed corpus.

The other CLIs assume expertise: `corpus-survey` tells you what's in a tree,
`corpus-ingest --path DIR` ingests one directory transiently, and getting
from "here's a messy folder" to "it's searchable, reproducibly, without
surprises" has meant running several of them by hand, in the right order,
reading the warnings, and hand-editing corpus.toml. `corpus-index` is that
whole sequence as one command:

    corpus-index PATH                    # survey, plan, confirm, ingest
    corpus-index PATH --dry-run          # survey + plan only, never writes
    corpus-index PATH --yes              # skip the confirmation prompt
    corpus-index PATH --check-overlap DB # + is this already indexed elsewhere?

It reuses `corpus.survey.census` for what's in the tree,
`corpus.util.autodetect.detect_sources` for which connectors apply, and
`corpus.ingester.Ingester` to actually ingest — see `corpus.planner` for the
planning/config-merge layer that ties them together. Nothing here
re-implements any of that; this module is CLI wiring plus a confirmation gate.

corpus.toml must already exist (`corpus-init` creates one) — corpus-index
only ever appends `[[sources]]` blocks to it, never invents db_path/embedder
settings. Detected sources are written into corpus.toml (merged, not
replaced) rather than ingested transiently, on the reasoning that a one-off
ingest nobody can repeat is a trap: the only way to pick up new/changed
files later is to re-run the same command, and that only works if it's
reproducible from config. Nothing is written on `--dry-run`, and nothing is
written before the user confirms (or passes `--yes`).

**Noise directories are excluded from BOTH the plan and the real ingest.**
`corpus-index` (via `corpus.survey.census`) excludes well-known noise
directories — `node_modules`, `.git`, build caches, `.photoslibrary`
bundles, `dist`/`build`/`target` when a corroborating ecosystem manifest
confirms it, etc. — from the PLAN's counts, the same way `corpus-survey`
already does. The underlying file connectors (`corpus.connectors.*`, via
`corpus.connectors.discovery.discover_files`) apply the identical exclusion
by default when they actually read a source's files, so "excluded from the
plan, not ingested" is a real guarantee, not just a claim about the preview.
(It used to not be: every connector globbed a source's whole `path` with no
directory-exclude mechanism of its own, so the plan's counts and the real
ingest disagreed. Fixed at `discover_files` itself — see its module
docstring — rather than by threading an excludes concept through every
connector + `SourceConfig`, since every connector already funnels through
that one function.)

**Remaining gap**: `--no-default-excludes` / `--exclude PATTERN` change only
what THIS PLAN shows you. There is currently no per-source way to turn off
default exclusion at ingest time from `corpus.toml` — the connectors always
apply it. If you genuinely need a vendored/build tree indexed, point a
source's `path` directly at that subdirectory (exclusion only ever prunes a
directory encountered *during* a walk, never the configured root itself).

**Upgrading an existing index**: if a source previously picked up files
inside what's now an excluded directory, re-running `corpus-index` (or
`corpus-ingest`) against it will prune those chunks as orphans on this
first post-upgrade run — expected, since that content was never supposed to
be searchable. A large drop is caught by the orphan-pruning blast-radius
guard (`[pruning]` in corpus.toml, default: refuse above 20% of a source's
existing chunks) rather than silently deleted; rerun with `--prune-anyway`
once you've confirmed the drop is exactly this dependency-tree content and
not a connector regression.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from corpus.cli._common import configure_logging, load_config_or_exit
from corpus.cli.ingest import _print_tokens
from corpus.config import DEFAULT_CONFIG_PATH
from corpus.credentials import resolve_dotenv
from corpus.ingester import Ingester, IngestResult
from corpus.planner import IndexPlan, MergeResult, build_plan, merge_sources_into_toml
from corpus.survey.format import human_count, human_size
from corpus.survey.overlap import run_overlap_survey
from corpus.util.text_yield import MEASURED_TEXT_YIELD_RATIOS


def _print_gap(plan: IndexPlan) -> None:
    total_count = sum(b.count for b in plan.gap)
    total_bytes = sum(b.total_bytes for b in plan.gap)
    print("\nGap — no connector (report this first; it's the whole point)")
    if not plan.gap:
        print("  (none — every file type present has a connector)")
        return
    print(f"  {'extension':<20} {'count':>10} {'size':>12}")
    for b in plan.gap:
        print(f"  {b.bucket:<20} {human_count(b.count):>10} {human_size(b.total_bytes):>12}")
    print(
        f"  -> {human_count(total_count)} file(s), {human_size(total_bytes)} that corpus "
        "cannot index today and will NOT be searchable after this run. See "
        "`corpus-survey census` for the full breakdown, or "
        "docs/adding_a_source.md to add a connector."
    )


def _media_in_gap(plan: IndexPlan) -> list[str]:
    """Media extensions the gap report just listed, if any."""
    from corpus.survey.media import MEDIA_EXTENSIONS

    return sorted(
        b.bucket for b in plan.gap if b.bucket.lower() in MEDIA_EXTENSIONS
    )


def _print_media_offer(plan: IndexPlan, root: str) -> bool:
    """Say that the media in the gap COULD be indexed, and what it would cost.

    The gap report above is honest but incomplete on its own: it says these
    files will not be searchable without saying that a transcription pass
    would change that. Somebody pointing this at a folder of recordings should
    not have to read the docs to discover the feature exists.

    Returns whether transcription is actually possible here.
    """
    extensions = _media_in_gap(plan)
    if not extensions:
        return False

    from corpus.transcripts.audio import ffmpeg_available

    print(
        f"\n  Of those, {', '.join(extensions)} hold SPEECH that can be "
        "transcribed and indexed."
    )

    missing = []
    try:
        from corpus.transcripts.backends import BackendUnavailableError, default_backend

        default_backend()
    except BackendUnavailableError as exc:
        missing.append(str(exc).split(".")[0])
    except ImportError:
        missing.append("the transcription backend is not installed")
    if not ffmpeg_available():
        missing.append("ffmpeg is not on PATH")

    if missing:
        print("  Not available here: " + "; ".join(missing) + ".")
        print(
            "  Install with: uv add 'corpus-rag[transcribe,transcribe-mlx]' "
            "(Apple Silicon), plus ffmpeg."
        )
        return False

    print(
        f"  Run `corpus-transcribe {root}` first, then re-run this command -- "
        "the transcripts become a source like any other."
    )
    return True


def _print_noise(plan: IndexPlan) -> None:
    ws = plan.census.walk_stats
    print("\nNoise (excluded from the plan, not ingested)")
    if ws.dirs_pruned:
        print(
            f"  {human_count(ws.dirs_pruned)} director{'y' if ws.dirs_pruned == 1 else 'ies'} "
            "excluded by default (node_modules, .git, caches, .photoslibrary, "
            "dist/build/target when a manifest confirms it, ...) — the connectors "
            "that actually ingest each source apply this same default exclusion, "
            "so this reflects what will really happen."
        )
    if plan.noise:
        print(f"  {'pattern':<24} {'count':>10} {'size':>12}  reason")
        for b in plan.noise:
            print(
                f"  {b.bucket:<24} {human_count(b.count):>10} {human_size(b.total_bytes):>12}"
                f"  {b.detail}"
            )
    if not ws.dirs_pruned and not plan.noise:
        print("  (none found)")
    if not plan.use_default_excludes:
        print(
            "  NOTE: --no-default-excludes changes only the preview above — the "
            "connectors that actually ingest each source always apply their own "
            "default directory exclusion regardless, and there is currently no "
            "per-source override in corpus.toml. Point a source's `path` directly "
            "at a subdirectory to index it despite the default (e.g. a vendored "
            "tree)."
        )
    elif plan.excludes:
        print(
            "  NOTE: --exclude patterns affect only the preview above — they are "
            "not applied at ingest time."
        )


def _print_overlap(plan: IndexPlan) -> None:
    result = plan.overlap
    if result is None:
        return
    print(f"\nOverlap vs. {result.db_path}")
    if not result.sample:
        print(
            f"  {human_count(result.eligible_document_count)} eligible plain-text "
            "document(s) — none sampled, no estimate."
        )
        return
    frac = result.estimated_overlap_fraction
    ci = result.confidence_interval_95
    print(
        f"  sampled {human_count(result.sample_size)} of "
        f"{human_count(result.eligible_document_count)} eligible document(s), "
        f"{human_count(result.matched_count)} matched."
    )
    if frac is not None and ci is not None:
        print(f"  estimated overlap: {frac:.0%}  (95% CI: {ci[0]:.0%}–{ci[1]:.0%})")
    print(
        "  (phrase-sampling estimate, not a full comparison — see "
        "`corpus-survey overlap`'s method notes for what this does and doesn't catch)"
    )


def _print_plan_table(plan: IndexPlan, provider: str, model: str) -> None:
    print("\nPlan — sources that would be written to corpus.toml and ingested")
    if not plan.sources:
        print("  (nothing detected)")
        return
    print(f"  {'name':<28} {'type':<10} {'files':>8} {'size':>12} {'est. tokens':>14}")
    for p in sorted(plan.sources, key=lambda p: -p.total_bytes):
        print(
            f"  {p.source.name:<28} {p.source.type:<10} {human_count(p.file_count):>8} "
            f"{human_size(p.total_bytes):>12} {human_count(p.estimated_tokens):>14}"
        )
    print(
        f"  {'TOTAL':<28} {'':<10} {human_count(plan.total_file_count):>8} "
        f"{human_size(plan.total_bytes):>12} {human_count(plan.total_estimated_tokens):>14}"
    )
    ratio_examples = ", ".join(
        f"{t} ~{MEASURED_TEXT_YIELD_RATIOS[t]:.1%}"
        for t in ("pdf", "docx", "html", "text")
        if t in MEASURED_TEXT_YIELD_RATIOS
    )
    print(
        "\n  Estimated tokens = raw bytes × a per-format text-yield ratio measured "
        f"against a real corpus ({ratio_examples} — see corpus.util.text_yield), ÷ 4. "
        "NOT the embedder's real tokenizer count, and rounded up, not down, when "
        "uncertain: an unmeasured or unusually image/scan-heavy file can still cost "
        "less or more than this shows — a scanned PDF with no text layer yields "
        f"close to nothing until OCR'd, for instance. Embedding is billed per token "
        f"by your provider (embedder: {provider}/{model}) — check its current "
        "pricing before a large first run."
    )


def _print_merge(merge: MergeResult) -> None:
    if merge.added:
        print(f"\nWrote {len(merge.added)} new source(s) to {merge.config_path}:")
        for s in merge.added:
            print(f"  + {s.name}")
    if merge.unchanged:
        print(f"Already configured (unchanged): {', '.join(s.name for s in merge.unchanged)}")
    if merge.conflicts:
        print(f"\nREFUSED to touch {len(merge.conflicts)} conflicting source name(s):")
        for o in merge.conflicts:
            print(f"  ! {o.source.name}: {o.detail}")


def _print_ingest_result(name: str, r: IngestResult) -> int:
    print(f"=== Ingesting {name} ===")
    print(f"  documents:        {r.documents:,}")
    print(f"  chunks seen:      {r.chunks_seen:,}")
    print(f"  chunks upserted:  {r.chunks_upserted:,}")
    print(f"  chunks unchanged: {r.chunks_skipped:,}")
    if r.files_skipped:
        print(f"  files skipped:    {r.files_skipped:,}  (unsupported/unreadable by design)")
    exit_bit = 0
    if r.prune_refused:
        print("  orphans deleted:  0  (pruning REFUSED — blast-radius guard tripped)")
        print(f"  {r.prune_refused_detail}")
        exit_bit = 1
    elif r.pruning_performed:
        print(f"  orphans deleted:  {r.orphans_deleted:,}")
    else:
        print(f"  files unreadable: {r.files_failed:,}")
        print(
            "  orphans deleted:  0  (pruning SKIPPED — re-run `corpus-ingest "
            f"--source {name} --prune-anyway` once you've looked at the unreadable files)"
        )
    _print_tokens(tokens_used=r.tokens_used, counted=r.tokens_counted)
    print(f"  elapsed:          {r.elapsed_seconds:.1f}s")
    print()
    return exit_bit


def _confirm(prompt: str) -> bool:
    try:
        answer = input(prompt).strip().lower()
    except EOFError:
        return False
    return answer in ("y", "yes")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="corpus-index",
        description="Survey a directory, plan connectors, confirm, and ingest — one command.",
    )
    parser.add_argument("path", help="Directory to index")
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        dest="excludes",
        metavar="PATTERN",
        help="fnmatch pattern to exclude from the plan's preview (repeatable), e.g. "
        "--exclude '*.log' — does not affect what actually gets ingested",
    )
    parser.add_argument(
        "--no-default-excludes",
        action="store_true",
        help="Don't prune common noise dirs (node_modules, .git, __pycache__, "
        "*.photoslibrary, ...) from the plan's preview — see "
        "corpus.util.exclude.DEFAULT_EXCLUDED_DIR_NAMES. The connectors that "
        "actually ingest each source apply this same exclusion regardless; there "
        "is currently no per-source override",
    )
    parser.add_argument(
        "--name-prefix",
        default=None,
        metavar="TEXT",
        help="Prefix every detected source name with this (normalized the same way a "
        "folder name is). Use it to disambiguate when two differently-located folders "
        "share a basename and corpus-index refuses a name collision.",
    )
    parser.add_argument(
        "--check-overlap",
        default=None,
        metavar="DB",
        help="Also estimate how much of PATH is already indexed in an existing corpus "
        "database, before spending anything to re-embed it (see corpus-survey overlap)",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip the confirmation prompt and proceed straight to writing config + ingesting",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show the plan and stop. Never writes corpus.toml, never ingests.",
    )
    parser.add_argument("--config", default=None, help="Path to corpus.toml (default: ./corpus.toml)")
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser


def main_argv(argv: list[str]) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    # Resolve credentials now that --config is known (env var > .env beside
    # --config > .env in cwd) — needed before Ingester constructs an embedder.
    resolve_dotenv(args.config)

    configure_logging(args.verbose)

    config = load_config_or_exit(args.config)
    config_path = Path(args.config) if args.config else DEFAULT_CONFIG_PATH

    try:
        plan = build_plan(
            args.path,
            excludes=tuple(args.excludes),
            use_default_excludes=not args.no_default_excludes,
            name_prefix=args.name_prefix,
        )
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    if args.check_overlap:
        db_path = Path(args.check_overlap)
        if not db_path.exists():
            print(f"error: --check-overlap database not found: {db_path}", file=sys.stderr)
            return 1
        plan.overlap = run_overlap_survey(plan.root, db_path)

    print(f"corpus-index: {plan.root}")
    _print_gap(plan)
    _print_media_offer(plan, args.path)
    _print_noise(plan)
    _print_overlap(plan)
    _print_plan_table(plan, config.embedder.provider, config.embedder.model)

    if not plan.sources:
        print(
            f"\nNothing ingestible found in {plan.root}. Run `corpus-survey census "
            f"{plan.root}` to see the full breakdown, or add a connector "
            "(docs/adding_a_source.md)."
        )
        return 1

    if args.dry_run:
        print("\n(dry run — nothing written, nothing ingested)")
        return 0

    if not args.yes and not _confirm("\nWrite these sources to corpus.toml and ingest? [y/N] "):
        print("Aborted — nothing written, nothing ingested.")
        return 1

    merge = merge_sources_into_toml(
        config_path, config.sources, [p.source for p in plan.sources]
    )
    _print_merge(merge)

    if not merge.ingestible_names:
        print("\nNothing left to ingest — every detected source conflicted with an "
              "existing corpus.toml entry (see above).")
        return 1

    # Reload: the merge just appended [[sources]] on disk, and the Ingester
    # reads sources from the config object, not from the plan.
    config = load_config_or_exit(args.config)

    print()
    ingester = Ingester(config)
    exit_code = 1 if merge.conflicts else 0
    try:
        for name in merge.ingestible_names:
            try:
                r = ingester.ingest(name)
            except (ValueError, OSError, ImportError) as e:
                print(f"=== Ingesting {name} ===")
                print(f"  ERROR: {e}  (index left intact; skipping this source)")
                print()
                exit_code = 1
                continue
            exit_code = max(exit_code, _print_ingest_result(name, r))
    finally:
        ingester.close()

    return exit_code


def main() -> int:
    return main_argv(sys.argv[1:])


if __name__ == "__main__":
    sys.exit(main())
