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

import tempfile
import zipfile
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


def test_one_encrypted_member_skips_the_whole_archive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Conservative choice: an archive with a mix of encrypted and plain
    members is refused entirely rather than partially unpacked."""
    _make_zip(tmp_path / "mixed_secret.zip", {"secret.txt": "shh", "plain.txt": "not secret"})
    _mark_encrypted(monkeypatch, {"secret.txt"})

    conn = ZipConnector(source_type="archives", path=tmp_path)
    docs = list(conn.load())

    assert docs == []
    assert conn.skipped_files == 1


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
