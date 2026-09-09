"""Tests for the zip archive connector.

Covers the safety contract documented in `corpus/connectors/zip.py`: zip-slip
containment, zip-bomb caps (declared and actual), encrypted-archive
detection, nested-archive refusal, unsupported-extension accounting, the
failed_files/skipped_files split, and the guaranteed-cleanup-on-crash
property.

All fixture archives are built in-memory with `zipfile` in `tmp_path` — no
real archives are read, per the project's public-repo constraints.
"""

from __future__ import annotations

import struct
import tempfile
import zipfile
import zlib
from pathlib import Path

import pytest

from corpus.connectors.registry import CONNECTOR_REGISTRY, DEFAULT_GLOBS
from corpus.connectors.zip import ZipConnector


def _make_zip(path: Path, members: dict[str, str]) -> Path:
    with zipfile.ZipFile(path, "w") as zf:
        for name, content in members.items():
            zf.writestr(name, content)
    return path


# ---------------------------------------------------------------------------
# Basic extraction + composition with existing connectors
# ---------------------------------------------------------------------------


def test_loads_mixed_file_types_from_one_archive(tmp_path: Path) -> None:
    archive = _make_zip(
        tmp_path / "reports.zip",
        {
            "q3/summary.txt": "Q3 went fine.",
            "notes.md": "# Notes\n\nSome markdown content.",
        },
    )
    docs = list(ZipConnector(source_type="archives", path=tmp_path).load())

    by_key = {d.source_key: d for d in docs}
    assert "reports.zip::q3/summary.txt" in by_key
    assert "reports.zip::notes.md" in by_key
    assert by_key["reports.zip::q3/summary.txt"].raw["body"] == "Q3 went fine."
    # source_type is the OUTER zip source's name, not "text"/"markdown".
    assert all(d.source_type == "archives" for d in docs)
    assert archive.exists(), "the archive itself must never be modified/deleted"


def test_two_archives_with_same_inner_name_do_not_collide(tmp_path: Path) -> None:
    _make_zip(tmp_path / "a.zip", {"report.txt": "From archive A."})
    _make_zip(tmp_path / "b.zip", {"report.txt": "From archive B."})

    docs = list(ZipConnector(source_type="archives", path=tmp_path).load())
    by_key = {d.source_key: d.raw["body"] for d in docs}

    assert by_key["a.zip::report.txt"] == "From archive A."
    assert by_key["b.zip::report.txt"] == "From archive B."


def test_archives_in_different_subfolders_with_same_name_do_not_collide(tmp_path: Path) -> None:
    (tmp_path / "2019").mkdir()
    (tmp_path / "2020").mkdir()
    _make_zip(tmp_path / "2019" / "report.zip", {"body.txt": "2019 content."})
    _make_zip(tmp_path / "2020" / "report.zip", {"body.txt": "2020 content."})

    docs = list(ZipConnector(source_type="archives", path=tmp_path).load())
    keys = {d.source_key for d in docs}

    assert keys == {"2019/report.zip::body.txt", "2020/report.zip::body.txt"}


def test_missing_dir_raises() -> None:
    with pytest.raises(FileNotFoundError):
        list(ZipConnector(source_type="archives", path="/nonexistent").load())


def test_empty_archive_yields_nothing(tmp_path: Path) -> None:
    with zipfile.ZipFile(tmp_path / "empty.zip", "w"):
        pass
    docs = list(ZipConnector(source_type="archives", path=tmp_path).load())
    assert docs == []


def test_unsupported_extension_is_skipped_not_ingested(tmp_path: Path) -> None:
    _make_zip(
        tmp_path / "mixed.zip",
        {"photo.jpg": "not really a jpeg", "note.txt": "real content"},
    )
    conn = ZipConnector(source_type="archives", path=tmp_path)
    docs = list(conn.load())

    assert {d.source_key for d in docs} == {"mixed.zip::note.txt"}
    assert conn.skipped_files == 1
    assert conn.failed_files == 0


# ---------------------------------------------------------------------------
# Archive/OS packaging noise (__MACOSX/AppleDouble, .DS_Store, Thumbs.db)
# ---------------------------------------------------------------------------


def test_macos_appledouble_members_are_ignored_not_counted(tmp_path: Path) -> None:
    """Real-world regression: a Mac-created zip's __MACOSX/._ resource-fork
    twin of a .txt/.pdf/etc. file matches that type's extension and, without
    filtering, would be handed to the real connector and either produce a
    phantom document (for a text-parseable type) or fail to parse and land
    in failed_files -- which suppresses orphan pruning for the WHOLE source,
    permanently, for files that were never documents. None of these five
    noise members should produce a document or move either counter."""
    _make_zip(
        tmp_path / "mac_export.zip",
        {
            "notes/report.txt": "real content",
            "__MACOSX/notes/._report.txt": "resource-fork junk, not real content",
            "._root_level_twin.txt": "AppleDouble twin outside __MACOSX/ too",
            ".DS_Store": "folder metadata, not a document",
            "Thumbs.db": "windows thumbnail cache, not a document",
        },
    )
    conn = ZipConnector(source_type="archives", path=tmp_path)
    docs = list(conn.load())

    assert {d.source_key for d in docs} == {"mac_export.zip::notes/report.txt"}
    assert conn.skipped_files == 0
    assert conn.failed_files == 0


def test_macosx_directory_entry_itself_is_ignored(tmp_path: Path) -> None:
    """A bare __MACOSX/ directory entry (no filename component to trip the
    `._` check) must not sneak through some other path."""
    with zipfile.ZipFile(tmp_path / "with_dir.zip", "w") as zf:
        zf.writestr("__MACOSX/", "")
        zf.writestr("__MACOSX/notes/", "")
        zf.writestr("real.txt", "real content")
    docs = list(ZipConnector(source_type="archives", path=tmp_path).load())
    assert {d.source_key for d in docs} == {"with_dir.zip::real.txt"}


# ---------------------------------------------------------------------------
# Dependency / build-output noise (node_modules, site-packages, VCS dirs,
# minified leaves, ...) -- overridable via `exclude_dependencies`, unlike the
# packaging noise above.
# ---------------------------------------------------------------------------


def test_node_modules_readme_is_excluded_real_doc_is_kept(tmp_path: Path) -> None:
    """Real-world trigger: an archive whose vendored `node_modules` tree
    contains hundreds of markdown READMEs from third-party npm packages --
    without this filter, every one of those matches the markdown connector
    and gets indexed as if it were the user's own content, degrading every
    future search. Exactly one real document should survive, both counters
    at zero (the noise is excluded outright, not downgraded to skipped)."""
    _make_zip(
        tmp_path / "bundle.zip",
        {
            "node_modules/example-pkg/README.md": (
                "# example-pkg\n\nA vendored dependency's own README, not user content."
            ),
            "docs/guide.md": "# Guide\n\nReal user-authored content.",
        },
    )
    conn = ZipConnector(source_type="archives", path=tmp_path)
    docs = list(conn.load())

    assert {d.source_key for d in docs} == {"bundle.zip::docs/guide.md"}
    assert conn.skipped_files == 0
    assert conn.failed_files == 0


def test_node_modules_filter_disabled_yields_both(tmp_path: Path) -> None:
    """`exclude_dependencies=False` is the escape hatch for someone who
    deliberately wants a vendored library's docs indexed -- same archive as
    above, both documents now survive."""
    _make_zip(
        tmp_path / "bundle.zip",
        {
            "node_modules/example-pkg/README.md": (
                "# example-pkg\n\nA vendored dependency's own README, not user content."
            ),
            "docs/guide.md": "# Guide\n\nReal user-authored content.",
        },
    )
    conn = ZipConnector(source_type="archives", path=tmp_path, exclude_dependencies=False)
    docs = list(conn.load())

    assert {d.source_key for d in docs} == {
        "bundle.zip::node_modules/example-pkg/README.md",
        "bundle.zip::docs/guide.md",
    }


def test_lookalike_directory_names_are_not_excluded(tmp_path: Path) -> None:
    """The false-positive case that matters: a path component that merely
    CONTAINS a noise directory's name must not match -- matching is on the
    exact component, never a substring. `distribution/` in particular must
    survive even though `dist` (its corroborated-only cousin, see below) is
    on the noise list."""
    _make_zip(
        tmp_path / "lookalikes.zip",
        {
            "my-node_modules-notes/idea.txt": "not actually a dependency tree",
            "distribution/plan.txt": "not actually a build-output directory",
        },
    )
    conn = ZipConnector(source_type="archives", path=tmp_path)
    docs = list(conn.load())

    assert {d.source_key for d in docs} == {
        "lookalikes.zip::my-node_modules-notes/idea.txt",
        "lookalikes.zip::distribution/plan.txt",
    }
    assert conn.skipped_files == 0
    assert conn.failed_files == 0


def test_uppercase_dependency_directory_name_is_still_excluded(tmp_path: Path) -> None:
    """Decision: directory-name matching is case-insensitive, consistent
    with this connector's existing extension matching and its
    `.DS_Store`/`Thumbs.db` handling. Real tooling always lowercases
    `node_modules`; matching case-insensitively only widens what's caught
    (a Windows tool, or a manual rezip that changed case) since a real
    personal folder deliberately named `Node_Modules` is not a realistic
    case to protect."""
    _make_zip(
        tmp_path / "cased.zip",
        {"Node_Modules/pkg/README.md": "vendored dependency", "notes.txt": "real content"},
    )
    conn = ZipConnector(source_type="archives", path=tmp_path)
    docs = list(conn.load())

    assert {d.source_key for d in docs} == {"cased.zip::notes.txt"}


def test_vcs_and_tooling_directories_are_excluded(tmp_path: Path) -> None:
    """Every unconditional entry on the dependency-noise list, exercised
    together in one archive alongside real content -- mirrors how the
    AppleDouble/.DS_Store/Thumbs.db test above combines several noise types
    in one pass. Each noise member is given a `.md` name specifically so
    that, if the filter ever missed one, it would show up as a leaked
    document in the set-equality assertion below rather than silently
    passing for the wrong reason."""
    _make_zip(
        tmp_path / "everything.zip",
        {
            ".git/config.md": "git internals, never content",
            "__pycache__/notes.md": "compiled bytecode artifacts tree, never content",
            ".venv/lib/site-packages/pkg/readme.md": "double-nested for good measure",
            "vendor/somelib/readme.md": "vendored Go/PHP/Ruby dependency",
            "bower_components/somelib/readme.md": "vendored bower dependency",
            "site-packages/pkg/readme.md": "vendored python dependency",
            ".next/cache/readme.md": "next.js build cache",
            ".svn/readme.md": "subversion metadata",
            ".hg/readme.md": "mercurial metadata",
            ".tox/readme.md": "tox build environment",
            "venv/readme.md": "python virtualenv, no leading dot",
            ".nuxt/readme.md": "nuxt build cache",
            "notes.md": "# Notes\n\nReal user content.",
        },
    )
    conn = ZipConnector(source_type="archives", path=tmp_path)
    docs = list(conn.load())

    assert {d.source_key for d in docs} == {"everything.zip::notes.md"}
    assert conn.skipped_files == 0
    assert conn.failed_files == 0


def test_dist_without_corroborating_marker_is_kept(tmp_path: Path) -> None:
    """`dist` is deliberately NOT on the unconditional noise list -- it's a
    common enough English word (a mailing "dist" list) that the bare name
    alone isn't a safe signal. With no ecosystem manifest anywhere in the
    archive, it must be treated as ordinary content."""
    _make_zip(tmp_path / "mailing.zip", {"dist/subscribers.txt": "mailing list, not a build"})
    conn = ZipConnector(source_type="archives", path=tmp_path)
    docs = list(conn.load())

    assert {d.source_key for d in docs} == {"mailing.zip::dist/subscribers.txt"}


def test_dist_corroborated_by_sibling_package_json_is_excluded(tmp_path: Path) -> None:
    """A `package.json` in the same archive corroborates `dist/` as npm
    build output. `package.json` itself has no matching connector (no JSON
    type), so it's still counted once in skipped_files -- only the dist/
    member is treated as noise."""
    _make_zip(
        tmp_path / "webapp.zip",
        {
            "package.json": '{"name": "example-app"}',
            "dist/notes.txt": "npm build output, not user content",
            "README.md": "# example-app\n\nReal project documentation.",
        },
    )
    conn = ZipConnector(source_type="archives", path=tmp_path)
    docs = list(conn.load())

    assert {d.source_key for d in docs} == {"webapp.zip::README.md"}
    assert conn.skipped_files == 1, "package.json itself: no matching connector, not noise"
    assert conn.failed_files == 0


def test_build_corroborated_by_sibling_pyproject_is_excluded(tmp_path: Path) -> None:
    _make_zip(
        tmp_path / "pypkg.zip",
        {
            "pyproject.toml": '[project]\nname = "example"',
            "build/notes.txt": "setuptools build output, not user content",
            "README.md": "# example\n\nReal project documentation.",
        },
    )
    conn = ZipConnector(source_type="archives", path=tmp_path)
    docs = list(conn.load())

    assert {d.source_key for d in docs} == {"pypkg.zip::README.md"}
    assert conn.skipped_files == 1, "pyproject.toml itself: no matching connector, not noise"


def test_target_corroborated_by_sibling_cargo_toml_is_excluded(tmp_path: Path) -> None:
    _make_zip(
        tmp_path / "rustcrate.zip",
        {
            "Cargo.toml": '[package]\nname = "example"',
            "target/notes.txt": "cargo build output, not user content",
            "README.md": "# example\n\nReal project documentation.",
        },
    )
    conn = ZipConnector(source_type="archives", path=tmp_path)
    docs = list(conn.load())

    assert {d.source_key for d in docs} == {"rustcrate.zip::README.md"}
    assert conn.skipped_files == 1, "Cargo.toml itself: no matching connector, not noise"


def test_dist_corroborated_by_manifest_higher_up_the_tree(tmp_path: Path) -> None:
    """The corroborating manifest need not be `dist`'s IMMEDIATE sibling --
    a monorepo with one root `package.json` and several nested
    `packages/*/dist/` output directories is common, so the marker search
    walks every ancestor level from the archive root down to `dist`'s
    parent. The sibling `README.md` inside the same package must survive --
    only the `dist/` subtree itself is noise."""
    _make_zip(
        tmp_path / "monorepo.zip",
        {
            "package.json": '{"name": "example-monorepo"}',
            "packages/example-lib/dist/notes.txt": "npm build output, not user content",
            "packages/example-lib/README.md": "# example-lib\n\nReal package documentation.",
        },
    )
    conn = ZipConnector(source_type="archives", path=tmp_path)
    docs = list(conn.load())

    assert {d.source_key for d in docs} == {"monorepo.zip::packages/example-lib/README.md"}


def test_minified_and_map_leaf_files_are_ignored(tmp_path: Path) -> None:
    """`*.min.js`, `*.min.css`, and `*.map` are never hand-authored content,
    regardless of which directory they're in. None of these extensions
    matches any connector today, so without this filter they'd still be
    extracted and counted in skipped_files as "no matching connector" --
    this connector treats them as noise instead, the same as any other
    dependency artifact: not extracted, not counted at all."""
    _make_zip(
        tmp_path / "assets.zip",
        {
            "app.min.js": "console.log('minified')",
            "app.min.css": "body{margin:0}",
            "app.js.map": '{"version":3}',
            "notes.txt": "real content",
        },
    )
    conn = ZipConnector(source_type="archives", path=tmp_path)
    docs = list(conn.load())

    assert {d.source_key for d in docs} == {"assets.zip::notes.txt"}
    assert conn.skipped_files == 0, "would be 3 (unclaimed extensions) without this filter"
    assert conn.failed_files == 0


def test_source_config_exclude_dependencies_defaults_true() -> None:
    from corpus.config import SourceConfig

    cfg = SourceConfig(name="archives", type="zip", path="/tmp")
    assert cfg.exclude_dependencies is True


def test_build_zip_passes_exclude_dependencies_through(tmp_path: Path) -> None:
    """End-to-end through the real registry-built pipeline: a `corpus.toml`
    `exclude_dependencies = false` on a `[[sources]]` block must actually
    reach `ZipConnector`, not just validate on `SourceConfig`."""
    from corpus.config import SourceConfig
    from corpus.connectors.registry import build_pipeline

    _make_zip(
        tmp_path / "bundle.zip",
        {"node_modules/pkg/README.md": "vendored dependency", "README.md": "real project docs"},
    )

    cfg = SourceConfig(
        name="archives", type="zip", path=str(tmp_path), exclude_dependencies=False
    )
    connector, _chunker = build_pipeline(cfg)
    docs = list(connector.load())

    assert {d.source_key for d in docs} == {
        "bundle.zip::node_modules/pkg/README.md",
        "bundle.zip::README.md",
    }


# ---------------------------------------------------------------------------
# Extension aliases and case-insensitivity
# ---------------------------------------------------------------------------

# A reasonably "real" HTML page so trafilatura has structure to chew on —
# matches the fixture in test_html_connector.py; a bare `<html><body>Hi</body>`
# snippet is too thin for trafilatura to reliably decide there's main content.
_SAMPLE_HTML = """
<!DOCTYPE html>
<html>
<head><title>The Article Title</title></head>
<body>
    <nav>Navigation menu — boilerplate</nav>
    <article>
        <h1>The Article Title</h1>
        <p>This is the main content paragraph. It contains substantive text
        that trafilatura should extract as the document body.</p>
        <p>A second paragraph with additional content for good measure.</p>
    </article>
</body>
</html>
"""


def test_htm_alias_is_recognized_as_html(tmp_path: Path) -> None:
    """The reported source_key uses the canonical extension (`.html`), not
    the archive member's exact original spelling (`.htm`) — a documented,
    purely cosmetic tradeoff (see `_normalize_to_canonical_extension`) for
    reusing each connector's existing single-glob discovery unmodified."""
    _make_zip(tmp_path / "pages.zip", {"index.htm": _SAMPLE_HTML})
    docs = list(ZipConnector(source_type="archives", path=tmp_path).load())
    assert {d.source_key for d in docs} == {"pages.zip::index.html"}


def test_markdown_alias_is_recognized(tmp_path: Path) -> None:
    _make_zip(tmp_path / "notes.zip", {"readme.markdown": "# Title\n\nBody text."})
    docs = list(ZipConnector(source_type="archives", path=tmp_path).load())
    assert {d.source_key for d in docs} == {"notes.zip::readme.md"}


def test_uppercase_extensions_are_recognized(tmp_path: Path) -> None:
    """Plain-text formats (no binary parsing to fake) stand in for the
    general uppercase-extension case; test_mixed_case_htm_alias_is_recognized
    below covers the alias+case combination together. Canonical extensions
    are always lowercase, so REPORT.TXT/NOTES.MD are found but reported with
    lowercased extensions, same rationale as the htm/markdown alias tests."""
    _make_zip(
        tmp_path / "windows_export.zip",
        {"REPORT.TXT": "uppercase text extension", "NOTES.MD": "# uppercase markdown"},
    )
    conn = ZipConnector(source_type="archives", path=tmp_path)
    docs = list(conn.load())

    assert {d.source_key for d in docs} == {
        "windows_export.zip::REPORT.txt",
        "windows_export.zip::NOTES.md",
    }
    assert conn.skipped_files == 0


def test_mixed_case_htm_alias_is_recognized(tmp_path: Path) -> None:
    _make_zip(tmp_path / "pages.zip", {"index.HTM": _SAMPLE_HTML})
    docs = list(ZipConnector(source_type="archives", path=tmp_path).load())
    assert {d.source_key for d in docs} == {"pages.zip::index.html"}


def test_legacy_doc_and_xls_are_not_aliased(tmp_path: Path) -> None:
    """.doc and .xls are different container formats python-docx/openpyxl
    cannot read -- they must fall into the unsupported-extension bucket, not
    be silently routed to the docx/xlsx connectors."""
    _make_zip(
        tmp_path / "legacy.zip",
        {"old.doc": "not really OLE2", "old.xls": "not really BIFF", "fine.txt": "ok"},
    )
    conn = ZipConnector(source_type="archives", path=tmp_path)
    docs = list(conn.load())

    assert {d.source_key for d in docs} == {"legacy.zip::fine.txt"}
    assert conn.skipped_files == 2
    assert conn.failed_files == 0


def test_case_insensitive_filesystem_collision_does_not_clobber(tmp_path: Path) -> None:
    """`notes.md` and `notes.MD` are distinct entries in the archive's own
    central directory but fold onto the same inode on a case-insensitive
    host filesystem (the macOS/Windows default) the moment both are
    extracted — a plain `target.open("wb")` would let the second write
    silently truncate-and-overwrite the first's content. `_disambiguate` in
    `_extract_members` must catch this at extraction time, before
    `_load_extracted` ever runs."""
    _make_zip(
        tmp_path / "dupes.zip",
        {"notes.md": "canonical content", "notes.MD": "aliased-casing content"},
    )
    conn = ZipConnector(source_type="archives", path=tmp_path)
    docs = list(conn.load())

    bodies = {d.raw["body"] for d in docs}
    assert len(docs) == 2, "both members must survive extraction, not just the second write"
    assert bodies == {"canonical content", "aliased-casing content"}


def test_alias_normalization_collision_does_not_clobber(tmp_path: Path) -> None:
    """`readme.md` (canonical) and `readme.markdown` (alias) are distinct,
    non-colliding files on disk right after extraction — they only collide
    once `_normalize_to_canonical_extension` renames the alias to `.md`.
    That rename must disambiguate rather than silently overwrite."""
    _make_zip(
        tmp_path / "dupes.zip",
        {"readme.md": "canonical content", "readme.markdown": "aliased-extension content"},
    )
    conn = ZipConnector(source_type="archives", path=tmp_path)
    docs = list(conn.load())

    bodies = {d.raw["body"] for d in docs}
    assert len(docs) == 2, "both members must survive the rename, not just one"
    assert bodies == {"canonical content", "aliased-extension content"}


# ---------------------------------------------------------------------------
# Mojibake member names — UTF-8/Shift-JIS filename bytes stored without the
# UTF-8 flag bit set, which zipfile then misdecodes as CP437. Fixtures are
# hand-assembled at the byte level (bypassing zipfile's own writer, which
# always sets the UTF-8 flag itself for a non-ASCII name and so can never
# produce this pattern) to reproduce exactly what real-world non-Python zip
# tools write. See `_repair_filename_encoding` in `corpus/connectors/zip.py`.
# ---------------------------------------------------------------------------


def _pack_local_header(name: bytes, content: bytes, flag_bits: int) -> bytes:
    crc = zlib.crc32(content) & 0xFFFFFFFF
    size = len(content)
    header = struct.pack(
        "<4sHHHHHIIIHH",
        b"PK\x03\x04",
        20,
        flag_bits,
        0,  # compression method: stored
        0,
        0,  # mod time, mod date
        crc,
        size,
        size,
        len(name),
        0,  # extra field length
    )
    return header + name + content


def _pack_central_header(name: bytes, content: bytes, flag_bits: int, offset: int) -> bytes:
    crc = zlib.crc32(content) & 0xFFFFFFFF
    size = len(content)
    header = struct.pack(
        "<4sHHHHHHIIIHHHHHII",
        b"PK\x01\x02",
        20,
        20,  # version made by, version needed
        flag_bits,
        0,  # compression method: stored
        0,
        0,  # mod time, mod date
        crc,
        size,
        size,
        len(name),
        0,
        0,  # extra field length, comment length
        0,  # disk number start
        0,  # internal file attributes
        0,  # external file attributes
        offset,
    )
    return header + name


def _build_raw_zip(path: Path, members: list[tuple[bytes, bytes, int]]) -> Path:
    """Hand-assemble a minimal, valid STORED-method zip from raw
    `(name_bytes, content_bytes, flag_bits)` tuples — bypassing
    `zipfile.ZipFile`'s writer entirely so the general-purpose flag bit and
    the raw filename bytes can be controlled independently. Verified
    against `zipfile.ZipFile` reads correctly (round-tripped in Python's own
    reader before use here)."""
    body = bytearray()
    central = bytearray()
    for name, content, flag_bits in members:
        offset = len(body)
        body += _pack_local_header(name, content, flag_bits)
        central += _pack_central_header(name, content, flag_bits, offset)
    cd_offset = len(body)
    eocd = struct.pack(
        "<4sHHHHIIH",
        b"PK\x05\x06",
        0,
        0,  # this disk, disk with central directory
        len(members),
        len(members),  # entries on this disk, total entries
        len(central),
        cd_offset,
        0,  # comment length
    )
    path.write_bytes(bytes(body) + bytes(central) + eocd)
    return path


def test_utf8_filename_without_flag_bit_is_repaired(tmp_path: Path) -> None:
    """The measured real-world case: a zip tool wrote UTF-8 filename bytes
    but never set the 0x800 flag, so zipfile misdecoded it as CP437."""
    real_name = "reports/第3四半期レポート.txt"
    _build_raw_zip(
        tmp_path / "archive.zip",
        [(real_name.encode("utf-8"), b"quarterly figures", 0)],
    )
    docs = list(ZipConnector(source_type="archives", path=tmp_path).load())
    assert len(docs) == 1
    assert docs[0].source_key == f"archive.zip::{real_name}"
    assert docs[0].raw["body"] == "quarterly figures"


def test_shift_jis_filename_without_flag_bit_is_repaired(tmp_path: Path) -> None:
    """Older Japanese-locale tools wrote Shift-JIS (CP932) filename bytes,
    not UTF-8 — the second repair tier."""
    real_name = "legacy/売上集計_FY13.txt"
    _build_raw_zip(
        tmp_path / "archive.zip",
        [(real_name.encode("cp932"), b"budget figures", 0)],
    )
    docs = list(ZipConnector(source_type="archives", path=tmp_path).load())
    assert len(docs) == 1
    assert docs[0].source_key == f"archive.zip::{real_name}"


def test_ascii_filename_without_flag_bit_is_left_alone(tmp_path: Path) -> None:
    """The overwhelmingly common case — an ASCII name with the flag unset
    (nothing to repair) — must round-trip to itself, not be treated as its
    own 'repair'."""
    _build_raw_zip(
        tmp_path / "archive.zip",
        [(b"reports/summary.txt", b"plain ascii content", 0)],
    )
    docs = list(ZipConnector(source_type="archives", path=tmp_path).load())
    assert len(docs) == 1
    assert docs[0].source_key == "archive.zip::reports/summary.txt"


def test_utf8_flagged_filename_is_never_touched(tmp_path: Path) -> None:
    """A correctly-flagged UTF-8 name (what zipfile's own writer always
    produces) must not be run through repair at all — it's already right."""
    name = "notes/日本語.txt"
    _make_zip(tmp_path / "archive.zip", {name: "already correct"})
    docs = list(ZipConnector(source_type="archives", path=tmp_path).load())
    assert docs[0].source_key == f"archive.zip::{name}"


def test_repair_filename_encoding_unit() -> None:
    """Direct unit coverage of the pure function, independent of the
    connector plumbing above."""
    import zipfile as zf_module

    from corpus.connectors.zip import _repair_filename_encoding

    def _info(name: str, flag_bits: int) -> zipfile.ZipInfo:
        info = zf_module.ZipInfo(filename=name)
        info.flag_bits = flag_bits
        return info

    real_name = "reports/第3四半期レポート.txt"
    mojibake = real_name.encode("utf-8").decode("cp437")
    assert _repair_filename_encoding(_info(mojibake, 0)) == real_name

    # Already UTF-8-flagged: never touched, even if it happens to look like
    # something re-encodable (it shouldn't be — this asserts the flag check
    # short-circuits before any repair attempt).
    assert _repair_filename_encoding(_info(real_name, 0x800)) == real_name

    # ASCII, unflagged: round-trips to itself, not treated as a "repair".
    assert _repair_filename_encoding(_info("plain.txt", 0)) == "plain.txt"


# ---------------------------------------------------------------------------
# Zip-slip: member paths that resolve outside the extraction root
# ---------------------------------------------------------------------------


def test_path_traversal_member_is_refused(tmp_path: Path) -> None:
    archive = tmp_path / "evil.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("../../escaped.txt", "should never be written")
        zf.writestr("safe.txt", "this one is fine")

    conn = ZipConnector(source_type="archives", path=tmp_path)
    docs = list(conn.load())

    assert {d.source_key for d in docs} == {"evil.zip::safe.txt"}
    assert conn.skipped_files == 1
    # Prove nothing escaped anywhere near the real filesystem root or tmp_path.
    assert not (tmp_path.parent / "escaped.txt").exists()
    assert not (tmp_path / "escaped.txt").exists()


def test_absolute_path_member_is_refused(tmp_path: Path) -> None:
    archive = tmp_path / "evil.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("/etc/evil-passwd", "should never be written")
        zf.writestr("safe.txt", "this one is fine")

    conn = ZipConnector(source_type="archives", path=tmp_path)
    docs = list(conn.load())

    assert {d.source_key for d in docs} == {"evil.zip::safe.txt"}
    assert conn.skipped_files == 1
    assert not Path("/etc/evil-passwd").exists()


# ---------------------------------------------------------------------------
# Zip bombs
# ---------------------------------------------------------------------------


def test_declared_size_over_cap_refuses_whole_archive(tmp_path: Path) -> None:
    _make_zip(tmp_path / "bomb.zip", {"a.txt": "x" * 1000, "b.txt": "y" * 1000})

    conn = ZipConnector(source_type="archives", path=tmp_path, max_uncompressed_bytes=500)
    docs = list(conn.load())

    assert docs == []
    assert conn.skipped_files == 1


def test_member_count_over_cap_refuses_whole_archive(tmp_path: Path) -> None:
    _make_zip(tmp_path / "many.zip", {f"f{i}.txt": "x" for i in range(10)})

    conn = ZipConnector(source_type="archives", path=tmp_path, max_members=5)
    docs = list(conn.load())

    assert docs == []
    assert conn.skipped_files == 1


def test_actual_bytes_exceeding_declared_size_is_caught_during_extraction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The streaming byte-count cap is a backstop for an extraction path that
    yields more actual bytes than the archive's central directory declared.

    Note: Python's `zipfile.ZipExtFile.read()` already bounds a normal read
    at the declared `file_size` from the SAME central-directory field the
    pre-check sums — so for a standard archive read through the public API,
    the pre-check alone is airtight and this streaming cap cannot actually
    trip on real-world input. This test exercises the connector's OWN
    accounting as a deliberate defense-in-depth measure (a different
    extraction path, a stdlib behavior change, a non-standard archive tool)
    by stubbing the read stream directly, independent of that stdlib
    guarantee.
    """
    import io

    _make_zip(tmp_path / "lying.zip", {"big.txt": "small declared payload"})

    class _HugeStream(io.BytesIO):
        def __init__(self) -> None:
            super().__init__(b"x" * 5000)

        def __enter__(self) -> _HugeStream:
            return self

        def __exit__(self, *exc: object) -> bool:
            return False

    def fake_open(self, name, *args, **kwargs):  # type: ignore[no-untyped-def]
        return _HugeStream()

    monkeypatch.setattr(zipfile.ZipFile, "open", fake_open)

    conn = ZipConnector(source_type="archives", path=tmp_path, max_uncompressed_bytes=100)
    docs = list(conn.load())

    assert docs == [], "actual bytes exceeding the cap must still refuse the archive"
    assert conn.skipped_files == 1


# ---------------------------------------------------------------------------
# Encrypted archives
# ---------------------------------------------------------------------------


def _mark_encrypted(monkeypatch: pytest.MonkeyPatch, filenames: set[str]) -> None:
    """`ZipFile.writestr`/`.open(mode='w')` unconditionally overwrite
    `flag_bits` on write (see `zipfile.ZipFile._open_to_write`), so there is
    no way to persist the standard encryption flag through the public write
    API — real encrypted archives get it from a genuine encryption step this
    project never performs. Patch `infolist()` to set it on read instead,
    which exercises the connector's detection logic exactly as it would fire
    against a real encrypted archive (both read the same attribute)."""
    real_infolist = zipfile.ZipFile.infolist

    def patched(self):  # type: ignore[no-untyped-def]
        infos = real_infolist(self)
        for info in infos:
            if info.filename in filenames:
                info.flag_bits |= 0x1
        return infos

    monkeypatch.setattr(zipfile.ZipFile, "infolist", patched)


def test_encrypted_archive_is_skipped_not_raised(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _make_zip(tmp_path / "secret.zip", {"secret.txt": "top secret content"})
    _mark_encrypted(monkeypatch, {"secret.txt"})

    conn = ZipConnector(source_type="archives", path=tmp_path)
    docs = list(conn.load())  # must not raise, must not prompt for anything

    assert docs == []
    assert conn.skipped_files == 1
    assert conn.failed_files == 0


def test_one_encrypted_member_is_skipped_others_still_ingested(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Per-member policy, matching zip-slip: an archive with a mix of
    encrypted and plain members yields the plain ones rather than being
    refused wholesale over one encrypted file."""
    _make_zip(tmp_path / "mixed_secret.zip", {"secret.txt": "shh", "plain.txt": "not secret"})
    _mark_encrypted(monkeypatch, {"secret.txt"})

    conn = ZipConnector(source_type="archives", path=tmp_path)
    docs = list(conn.load())

    assert {d.source_key for d in docs} == {"mixed_secret.zip::plain.txt"}
    assert conn.skipped_files == 1
    assert conn.failed_files == 0


# ---------------------------------------------------------------------------
# Nested archives (depth limit)
# ---------------------------------------------------------------------------


def test_nested_zip_is_refused_not_recursed(tmp_path: Path) -> None:
    import io

    # Build the inner archive's bytes in memory rather than as a real file
    # under tmp_path -- the connector's own root -- which would otherwise
    # also be discovered as its own separate top-level archive.
    inner_buf = io.BytesIO()
    with zipfile.ZipFile(inner_buf, "w") as inner_zf:
        inner_zf.writestr("deep.txt", "buried content")

    outer = tmp_path / "outer.zip"
    with zipfile.ZipFile(outer, "w") as zf:
        zf.writestr("inner.zip", inner_buf.getvalue())
        zf.writestr("top.txt", "top-level content")

    conn = ZipConnector(source_type="archives", path=tmp_path)
    docs = list(conn.load())

    keys = {d.source_key for d in docs}
    assert keys == {"outer.zip::top.txt"}
    assert not any("deep" in d.raw.get("body", "") for d in docs)
    assert conn.skipped_files == 1


# ---------------------------------------------------------------------------
# failed_files vs skipped_files bookkeeping
# ---------------------------------------------------------------------------


def test_corrupt_archive_counts_as_failed_not_skipped(tmp_path: Path) -> None:
    (tmp_path / "corrupt.zip").write_bytes(b"not actually a zip file")

    conn = ZipConnector(source_type="archives", path=tmp_path)
    docs = list(conn.load())

    assert docs == []
    assert conn.failed_files == 1
    assert conn.skipped_files == 0


def test_failed_and_skipped_reset_between_runs(tmp_path: Path) -> None:
    (tmp_path / "corrupt.zip").write_bytes(b"not actually a zip file")
    conn = ZipConnector(source_type="archives", path=tmp_path)
    list(conn.load())
    assert conn.failed_files == 1

    (tmp_path / "corrupt.zip").unlink()
    _make_zip(tmp_path / "fine.zip", {"a.txt": "content"})
    docs = list(conn.load())

    assert len(docs) == 1
    assert conn.failed_files == 0, "counter must reset at the start of load()"


def test_delegated_connector_failed_files_are_summed_in(tmp_path: Path) -> None:
    """A per-file read failure inside a delegated connector (e.g. one corrupt
    file among fine ones) must still surface through the zip connector's own
    failed_files, since the ingester only ever looks at the top-level
    connector's counters. Registers a fake 'text' factory rather than relying
    on a real connector's specific failure trigger."""
    from corpus.types import SourceDocument

    class _FlakyTextConnector:
        def __init__(self, source_type: str, **_: object) -> None:
            self.source_type = source_type
            self.failed_files = 0
            self.skipped_files = 0

        def load(self):  # type: ignore[no-untyped-def]
            self.failed_files = 1
            self.skipped_files = 2
            yield SourceDocument(
                source_type=self.source_type,
                source_key="fine.txt",
                title="fine",
                raw={"body": "still yielded despite the failure elsewhere"},
            )

    _make_zip(tmp_path / "docs.zip", {"fine.txt": "irrelevant — connector is faked"})

    original = CONNECTOR_REGISTRY["text"]
    CONNECTOR_REGISTRY["text"] = lambda cfg: (
        _FlakyTextConnector(source_type=cfg.name),
        None,
    )
    try:
        conn = ZipConnector(source_type="archives", path=tmp_path)
        docs = list(conn.load())
    finally:
        CONNECTOR_REGISTRY["text"] = original

    assert {d.source_key for d in docs} == {"docs.zip::fine.txt"}
    assert conn.failed_files == 1
    assert conn.skipped_files == 2


def test_missing_extra_is_counted_as_skipped_not_raised(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """If an archive contains a file type whose optional extra isn't
    installed, the zip connector must not blow up — those files are
    permanently unprocessable in this environment, so they're `skipped_files`,
    not a raised ImportError."""
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):  # type: ignore[no-untyped-def]
        if name == "pypdf" or name.startswith("pypdf."):
            raise ImportError("No module named 'pypdf'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    _make_zip(tmp_path / "docs.zip", {"a.pdf": "pretend pdf bytes", "b.txt": "fine"})
    conn = ZipConnector(source_type="archives", path=tmp_path)
    docs = list(conn.load())

    assert {d.source_key for d in docs} == {"docs.zip::b.txt"}
    assert conn.skipped_files == 1
    assert conn.failed_files == 0


# ---------------------------------------------------------------------------
# Guaranteed cleanup
# ---------------------------------------------------------------------------


def test_extraction_temp_dir_is_removed_even_if_processing_crashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _make_zip(tmp_path / "a.zip", {"note.txt": "hello world"})

    created_dirs: list[Path] = []
    real_mkdtemp = tempfile.mkdtemp

    def spy_mkdtemp(*args, **kwargs):  # type: ignore[no-untyped-def]
        d = real_mkdtemp(*args, **kwargs)
        created_dirs.append(Path(d))
        return d

    monkeypatch.setattr(tempfile, "mkdtemp", spy_mkdtemp)

    def boom(self, extract_dir, archive_key):  # type: ignore[no-untyped-def]
        raise RuntimeError("simulated crash mid-ingest")

    monkeypatch.setattr(ZipConnector, "_load_extracted", boom)

    conn = ZipConnector(source_type="archives", path=tmp_path)
    with pytest.raises(RuntimeError, match="simulated crash mid-ingest"):
        list(conn.load())

    assert created_dirs, "extraction should have created a temp directory"
    assert not created_dirs[0].exists(), "temp dir must be removed even after a crash"


def test_temp_dir_removed_after_successful_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _make_zip(tmp_path / "a.zip", {"note.txt": "hello world"})

    created_dirs: list[Path] = []
    real_mkdtemp = tempfile.mkdtemp

    def spy_mkdtemp(*args, **kwargs):  # type: ignore[no-untyped-def]
        d = real_mkdtemp(*args, **kwargs)
        created_dirs.append(Path(d))
        return d

    monkeypatch.setattr(tempfile, "mkdtemp", spy_mkdtemp)

    docs = list(ZipConnector(source_type="archives", path=tmp_path).load())

    assert len(docs) == 1
    assert created_dirs and not created_dirs[0].exists()


# ---------------------------------------------------------------------------
# Registry wiring
# ---------------------------------------------------------------------------


def test_zip_is_registered() -> None:
    assert "zip" in CONNECTOR_REGISTRY
    assert "zip" in DEFAULT_GLOBS
    assert DEFAULT_GLOBS["zip"] == "**/*.zip"


def test_member_connector_types_derives_from_registry_minus_denylist() -> None:
    """`_MEMBER_CONNECTOR_TYPES` used to be a hand-maintained tuple that
    silently missed `pptx`/`csv`/`tsv` for a release cycle after each was
    registered in `CONNECTOR_REGISTRY` — presentations and tables inside
    archives were invisible, with no error and no count. It's now derived
    from the registry itself (minus a small, explicit, documented denylist)
    so a future connector can't drift out of sync the same way."""
    from corpus.connectors.zip import _MEMBER_CONNECTOR_TYPES, _NON_MEMBER_CONNECTOR_TYPES

    assert set(_MEMBER_CONNECTOR_TYPES) == set(CONNECTOR_REGISTRY) - _NON_MEMBER_CONNECTOR_TYPES
    assert "pptx" in _MEMBER_CONNECTOR_TYPES
    assert "csv" in _MEMBER_CONNECTOR_TYPES
    assert "tsv" in _MEMBER_CONNECTOR_TYPES
    # Deliberately excluded, not merely "not added yet" — see the denylist's
    # own comment in zip.py for why each is different from a plain gap.
    assert "zip" not in _MEMBER_CONNECTOR_TYPES
    assert "aup3" not in _MEMBER_CONNECTOR_TYPES


def test_pptx_and_csv_and_tsv_members_become_documents(tmp_path: Path) -> None:
    """Regression test for the drift `test_member_connector_types_derives_...`
    fixes: presentations and tables inside an archive must become real
    documents, not silently vanish."""
    from pptx import Presentation

    deck_path = tmp_path / "deck.pptx"
    prs = Presentation()
    slide = prs.slides.add_slide(prs.slide_layouts[1])
    slide.shapes.title.text = "Quarterly Review"
    slide.placeholders[1].text_frame.text = "Revenue grew 12% this quarter."
    prs.save(deck_path)

    archive = tmp_path / "bundle.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.write(deck_path, "slides/deck.pptx")
        zf.writestr("data/budget.csv", "name,revenue\nAcme,1200.50\nGlobex,980\n")
        zf.writestr("data/exports.tsv", "name\trevenue\nAcme\t1200.50\nGlobex\t980\n")
    deck_path.unlink()

    docs = list(ZipConnector(source_type="archives", path=tmp_path).load())
    by_key = {d.source_key: d for d in docs}

    assert "bundle.zip::slides/deck.pptx" in by_key
    assert "Revenue grew 12% this quarter." in by_key["bundle.zip::slides/deck.pptx"].raw["body"]

    assert "bundle.zip::data/budget.csv" in by_key
    assert "Acme" in by_key["bundle.zip::data/budget.csv"].raw["body"]

    assert "bundle.zip::data/exports.tsv" in by_key
    assert "Globex" in by_key["bundle.zip::data/exports.tsv"].raw["body"]

    # source_type is the OUTER zip source's name for every member, regardless
    # of which inner connector produced it — matches the mixed-type test
    # above.
    assert all(d.source_type == "archives" for d in docs)


def test_aup3_member_is_not_routed_to_a_connector(tmp_path: Path) -> None:
    """`.aup3` is registered in `CONNECTOR_REGISTRY` (see
    `corpus/connectors/aup3.py`) but deliberately excluded from
    `_MEMBER_CONNECTOR_TYPES`: its extraction API writes a playable file
    "adjacent to the source", which has no sensible meaning inside a
    disposable zip extraction directory that's deleted before this method
    returns. A member is left in the unsupported-extension bucket instead —
    same as any other unregistered type, not a crash and not silently
    dropped without a count."""
    _make_zip(tmp_path / "archive.zip", {"session.aup3": "not a real sqlite database"})
    conn = ZipConnector(source_type="archives", path=tmp_path)
    docs = list(conn.load())
    assert docs == []
    assert conn.skipped_files == 1
    assert conn.failed_files == 0


def test_build_zip(tmp_path: Path) -> None:
    from corpus.config import SourceConfig
    from corpus.connectors.registry import build_pipeline

    cfg = SourceConfig(name="archives", type="zip", path=str(tmp_path))
    connector, chunker = build_pipeline(cfg)
    assert connector.source_type == "archives"
    assert chunker.source_type == "archives"


def test_end_to_end_chunking_produces_traceable_source_keys(tmp_path: Path) -> None:
    """Smoke test through the real registry-built pipeline, matching how the
    ingester actually drives a connector+chunker pair."""
    from corpus.config import SourceConfig
    from corpus.connectors.registry import build_pipeline

    _make_zip(tmp_path / "reports.zip", {"q3/summary.txt": "Quarterly summary content."})

    cfg = SourceConfig(name="archives", type="zip", path=str(tmp_path))
    connector, chunker = build_pipeline(cfg)

    docs = list(connector.load())
    assert len(docs) == 1
    chunks = chunker.chunk(docs[0])
    assert len(chunks) >= 1
    assert chunks[0].metadata.source_key == "reports.zip::q3/summary.txt"
    assert chunks[0].metadata.source_type == "archives"
