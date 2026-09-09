"""`corpus-index`'s planning layer: turn a raw directory into a reviewable,
reproducible ingest plan.

Composes two things that already exist rather than re-deciding either:

  - `corpus.survey.census.run_census` for what's in the tree (indexable /
    gap / noise), so the gap and noise reporting here is byte-identical to
    what `corpus-survey census` would say about the same directory.
  - `corpus.util.autodetect.detect_sources` for which connector types apply
    and what to name them — including its folder-basename namespacing,
    which keeps two *differently-named* folders from colliding.

What's new here, because neither of those two pieces needed it: turning a
one-off directory scan into **persisted, reproducible** `[[sources]]`
config. That raises a problem autodetect's transient `--path` mode never has
to solve — `corpus-ingest --path DIR` replaces `config.sources` in memory for
a single run and discards the result, so two folders sharing a basename
(`~/Work/Inbox` and `~/Personal/Inbox`) never meet each other. Once detected
sources get WRITTEN into one shared corpus.toml across multiple
`corpus-index` runs, they can meet, and `source_type`-scoped orphan pruning
means a name collision between two different folders would make ingesting
the second one delete the first one's chunks. `merge_sources_into_toml`
below refuses that merge outright rather than guessing — see its docstring.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from corpus.config import SourceConfig
from corpus.survey.census import BucketStat, CensusResult, run_census
from corpus.survey.overlap import OverlapResult
from corpus.util.autodetect import detect_sources, normalize_source_name
from corpus.util.text_yield import estimate_tokens_from_bytes
from corpus.util.toml_write import toml_str


@dataclass
class PlannedSource:
    """One connector type detected under the surveyed root, with the
    census-derived counts corpus-index will show before writing anything."""

    source: SourceConfig
    file_count: int
    total_bytes: int

    @property
    def estimated_tokens(self) -> int:
        # Rough token estimate, not a real tokenizer count: raw bytes ->
        # format-calibrated character estimate (`corpus.util.text_yield`,
        # measured per connector type against a real corpus, since a
        # compressed/binary container's file size says almost nothing about
        # its extracted text length) -> chars/4. Computed from bytes-on-disk
        # because nothing has parsed the files yet at planning time — the
        # whole point of a plan is to show cost *before* paying to open
        # every file with an optional connector extra that might not even
        # be installed. Reported as an estimate with that caveat spelled
        # out, never as a bare number — same honesty standard
        # `corpus-survey media`/`overlap` hold their own extrapolations to.
        return estimate_tokens_from_bytes(self.source.type, self.total_bytes)


@dataclass
class IndexPlan:
    root: Path
    census: CensusResult
    sources: list[PlannedSource] = field(default_factory=list)
    excludes: tuple[str, ...] = ()
    use_default_excludes: bool = True
    overlap: OverlapResult | None = None

    @property
    def gap(self) -> list[BucketStat]:
        return self.census.gap

    @property
    def noise(self) -> list[BucketStat]:
        return self.census.noise

    @property
    def total_file_count(self) -> int:
        return sum(p.file_count for p in self.sources)

    @property
    def total_bytes(self) -> int:
        return sum(p.total_bytes for p in self.sources)

    @property
    def total_estimated_tokens(self) -> int:
        return sum(p.estimated_tokens for p in self.sources)


def build_plan(
    path: Path | str,
    excludes: tuple[str, ...] = (),
    use_default_excludes: bool = True,
    name_prefix: str | None = None,
) -> IndexPlan:
    """Survey `path` and work out what an ingest of it would look like.

    Raises `FileNotFoundError` if `path` is not a directory (same contract
    `detect_sources` already has, so callers only need one except clause).

    A detected connector type is only included in the plan if the census —
    which, unlike `detect_sources`, DOES respect `excludes` /
    `use_default_excludes` — found indexable files of that type OUTSIDE the
    excluded directories. `detect_sources` itself has no excludes concept
    (it globs the whole tree per type); without this filter, a type whose
    only matches live inside e.g. `node_modules` would show a misleading
    "0 files" row in the plan while still getting written into corpus.toml
    as a live source — and `corpus-ingest` would glob that whole tree with
    no exclude mechanism of its own and pick the noise files up anyway. See
    "Known limitation" in `corpus/cli/index.py`'s module docstring for the
    piece of this that filtering alone cannot fix.
    """
    root = Path(os.path.expanduser(str(path))).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"not a directory: {root}")

    census = run_census(root, excludes=excludes, use_default_excludes=use_default_excludes)
    detected = detect_sources(root)

    if name_prefix:
        prefix = normalize_source_name(name_prefix)
        detected = [s.model_copy(update={"name": f"{prefix}_{s.name}"}) for s in detected]

    bytes_by_type: dict[str, int] = {}
    count_by_type: dict[str, int] = {}
    for b in census.indexable:
        bytes_by_type[b.detail] = bytes_by_type.get(b.detail, 0) + b.total_bytes
        count_by_type[b.detail] = count_by_type.get(b.detail, 0) + b.count

    sources = [
        PlannedSource(
            source=s,
            file_count=count_by_type[s.type],
            total_bytes=bytes_by_type[s.type],
        )
        for s in detected
        if count_by_type.get(s.type, 0) > 0
    ]

    return IndexPlan(
        root=root,
        census=census,
        sources=sources,
        excludes=excludes,
        use_default_excludes=use_default_excludes,
    )


MergeStatus = Literal["added", "unchanged", "conflict"]


@dataclass
class SourceMergeOutcome:
    source: SourceConfig
    status: MergeStatus
    detail: str | None = None


@dataclass
class MergeResult:
    config_path: Path
    outcomes: list[SourceMergeOutcome]
    written: bool

    @property
    def added(self) -> list[SourceConfig]:
        return [o.source for o in self.outcomes if o.status == "added"]

    @property
    def unchanged(self) -> list[SourceConfig]:
        return [o.source for o in self.outcomes if o.status == "unchanged"]

    @property
    def conflicts(self) -> list[SourceMergeOutcome]:
        return [o for o in self.outcomes if o.status == "conflict"]

    @property
    def ingestible_names(self) -> list[str]:
        """Source names safe to ingest this run: newly written entries plus
        ones that already matched what's on disk. Conflicting names are
        deliberately excluded — see `merge_sources_into_toml`."""
        return [o.source.name for o in self.outcomes if o.status in ("added", "unchanged")]


def _render_source_block(s: SourceConfig) -> str:
    lines = [
        "[[sources]]",
        f"name = {toml_str(s.name)}",
        f"type = {toml_str(s.type)}",
        f"path = {toml_str(s.path)}",
    ]
    if s.glob is not None:
        lines.append(f"glob = {toml_str(s.glob)}")
    if not s.exclude_dependencies:
        lines.append("exclude_dependencies = false")
    return "\n".join(lines)


def merge_sources_into_toml(
    config_path: Path,
    existing_sources: Sequence[SourceConfig],
    new_sources: Sequence[SourceConfig],
) -> MergeResult:
    """Append `new_sources` to `config_path` as `[[sources]]` blocks,
    idempotently and without ever touching an existing entry.

    Three outcomes per detected source, by name:

      - **added** — no source by this name exists yet; appended.
      - **unchanged** — a source by this name already exists with the same
        `type`/resolved `path`; re-running `corpus-index` on the same
        directory is therefore a no-op on the config (still re-ingests, so
        it stays the mechanism for picking up new/changed files).
      - **conflict** — a source by this name exists with a DIFFERENT
        `type`/`path`. This is the two-different-folders-same-basename
        collision `corpus.util.autodetect` namespaces against but cannot
        fully prevent on its own (it only sees one folder at a time).
        `source_type`-scoped orphan pruning means silently overwriting this
        entry would delete the original folder's chunks on the next
        ingest — a real data-loss footgun — so this refuses instead:
        the existing entry is left untouched, the conflicting source is
        left out of `ingestible_names`, and the caller is expected to
        surface `detail` so the operator can pass `--name-prefix` or edit
        corpus.toml by hand.

    Only ever appends text to `config_path`; a config with zero new sources
    to add performs no write at all (`written` stays `False`), and nothing
    outside `config_path` is ever touched.
    """
    existing_by_name = {s.name: s for s in existing_sources}
    outcomes: list[SourceMergeOutcome] = []
    to_append: list[SourceConfig] = []

    for s in new_sources:
        prior = existing_by_name.get(s.name)
        if prior is None:
            outcomes.append(SourceMergeOutcome(s, "added"))
            to_append.append(s)
        elif prior.type == s.type and prior.resolved_path() == s.resolved_path():
            outcomes.append(SourceMergeOutcome(s, "unchanged"))
        else:
            outcomes.append(
                SourceMergeOutcome(
                    s,
                    "conflict",
                    f"corpus.toml already has a source named '{s.name}' "
                    f"(type={prior.type}, path={prior.path}); this directory needs "
                    f"type={s.type}, path={s.path}. Leaving the existing entry alone — "
                    "two different folders produced the same name. Re-run with "
                    "--name-prefix to disambiguate, or edit corpus.toml by hand.",
                )
            )

    written = False
    if to_append:
        text = config_path.read_text()
        if text and not text.endswith("\n"):
            text += "\n"
        blocks = "\n\n".join(_render_source_block(s) for s in to_append)
        text += f"\n# Added by `corpus-index {to_append[0].path}`.\n{blocks}\n"
        config_path.write_text(text)
        written = True

    return MergeResult(config_path=config_path, outcomes=outcomes, written=written)
