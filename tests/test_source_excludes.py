"""`exclude` on a source, so a reported duplicate can actually be dropped.

`corpus-survey duplicates` names the documents that are wholly duplicated
elsewhere -- on one live archive, 1,119 of them holding 5,566 chunks. Naming
them was useless: there was no per-source `exclude`, so the only way to act
was to move files on disk or hand-edit a connector. A report whose
recommendation the tool cannot carry out is the same defect as a setting
documented in three places and read in none.

Matched the way `walk_files` already matches: fnmatch against BOTH the
root-relative path and the basename, so `--exclude Backup` and
`--exclude '**/Backup/*'` both do what the reader expects without them
having to know which form the implementation wanted.
"""

from __future__ import annotations

from pathlib import Path

from corpus.connectors.discovery import discover_files


def _tree(tmp_path: Path) -> Path:
    # Distinct bodies on purpose: MarkdownConnector collapses near-duplicates
    # within one load(), so identical text would hide the exclude list's
    # effect behind a different mechanism.
    for i, rel in enumerate((
        "live/report.md",
        "live/notes.md",
        "Backup/report.md",
        "Old Backup/deep/report.md",
        "live/copy of report.md",
    )):
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(f"# Heading {i}\n\nBody number {i} with its own distinct wording.\n")
    return tmp_path


def _names(paths) -> set[str]:
    return {p.name for p in paths}


def _rel(root: Path, paths) -> set[str]:
    return {str(p.relative_to(root)) for p in paths}


def test_nothing_is_excluded_by_default(tmp_path):
    """The positive control. Without it every test below could pass on a
    walker that found nothing."""
    found = list(discover_files(_tree(tmp_path), "**/*.md"))
    assert len(found) == 5


def test_a_directory_name_excludes_everything_under_it(tmp_path):
    """Compared on PATHS, not basenames. Three of these files are called
    report.md, so a set of names collapses them and the assertion passes
    whether or not anything was excluded."""
    root = _tree(tmp_path)
    paths = _rel(root, discover_files(root, "**/*.md", exclude=("Backup",)))
    assert paths == {
        "live/report.md", "live/notes.md", "live/copy of report.md",
        "Old Backup/deep/report.md",
    }


def test_a_directory_pattern_matches_a_name_exactly_not_loosely(tmp_path):
    """`"Backup"` is the folder called Backup, not every folder with Backup
    in its name -- the same way `"report.md"` is not `"final report.md"`.
    Write `*Backup*` for that, and the next test proves it works."""
    root = _tree(tmp_path)
    paths = _rel(root, discover_files(root, "**/*.md", exclude=("Backup",)))
    assert "Old Backup/deep/report.md" in paths


def test_a_wildcard_directory_pattern_catches_the_variants(tmp_path):
    root = _tree(tmp_path)
    paths = _rel(root, discover_files(root, "**/*.md", exclude=("*Backup*",)))
    assert paths == {"live/report.md", "live/notes.md", "live/copy of report.md"}


def test_a_glob_path_pattern_works_too(tmp_path):
    found = discover_files(_tree(tmp_path), "**/*.md", exclude=("**/Backup/*",))
    assert "Backup" not in {p.parent.name for p in found}


def test_a_filename_pattern_excludes_by_basename(tmp_path):
    names = _names(discover_files(_tree(tmp_path), "**/*.md", exclude=("copy of *",)))
    assert "copy of report.md" not in names
    assert "report.md" in names


def test_an_exact_relative_path_excludes_exactly_one_file(tmp_path):
    """What `corpus-survey duplicates --excludes-for` emits: full paths, so
    one copy goes and its twin stays."""
    found = list(discover_files(_tree(tmp_path), "**/*.md", exclude=("Backup/report.md",)))
    paths = {str(p.relative_to(tmp_path)) for p in found}
    assert "Backup/report.md" not in paths
    assert "live/report.md" in paths


def test_excluding_nothing_that_matches_leaves_the_tree_alone(tmp_path):
    found = discover_files(_tree(tmp_path), "**/*.md", exclude=("nothing-matches-this",))
    assert len(list(found)) == 5


def test_the_config_field_reaches_the_connector(tmp_path):
    """The wiring, not the matcher. A field the connector never reads is the
    defect this codebase has already shipped twice."""
    from corpus.config import SourceConfig
    from corpus.connectors.discovery import source_excludes
    from corpus.connectors.registry import build_pipeline

    _tree(tmp_path)
    cfg = SourceConfig(
        name="notes", type="markdown", path=str(tmp_path),
        glob="**/*.md", exclude=["Backup"],
    )
    connector, _chunker = build_pipeline(cfg)
    with source_excludes(cfg.exclude):
        keys = {Path(d.source_key).name for d in connector.load()}
    assert keys == {"report.md", "notes.md", "copy of report.md"}
    with source_excludes(cfg.exclude):
        assert len(list(connector.load())) == 4, "only the exact folder goes"


def test_the_ingester_applies_the_source_exclude_list(tmp_path):
    """The seam that matters: a real ingest, not a hand-entered context."""
    from corpus.config import CorpusConfig, EmbedderConfig, SourceConfig
    from corpus.ingester import Ingester

    _tree(tmp_path)
    cfg = CorpusConfig(
        db_path=tmp_path / "c.db",
        embedder=EmbedderConfig(provider="hash", dim=64),
        sources=[SourceConfig(
            name="notes", type="markdown", path=str(tmp_path),
            glob="**/*.md", exclude=["*Backup*"],
        )],
    )
    ing = Ingester(cfg)
    try:
        result = ing.ingest("notes")
    finally:
        ing.close()
    assert result.documents == 3, "the exclude list did not reach the ingest"
