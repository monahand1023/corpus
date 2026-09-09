"""Tests for `corpus.survey.archives` — zip inspection without extraction.

All fixture archives are built in-memory with `zipfile` in `tmp_path`, per
the project's public-repo constraints. No temp directory should ever appear
on disk for this module — that's asserted directly, not just implied.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from corpus.survey.archives import run_archive_survey


def _make_zip(path: Path, members: dict[str, str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as zf:
        for name, content in members.items():
            zf.writestr(name, content)
    return path


def _mark_encrypted(monkeypatch: pytest.MonkeyPatch, filenames: set[str]) -> None:
    """Same technique as `tests/test_zip_connector.py`: `ZipFile.writestr`
    unconditionally overwrites `flag_bits` on write, so there's no way to
    persist the standard encryption flag through the public write API for a
    file that isn't genuinely encrypted. Patch `infolist()` to set it on
    read instead — exercises the same detection attribute a real encrypted
    archive would set."""
    real_infolist = zipfile.ZipFile.infolist

    def patched(self):  # type: ignore[no-untyped-def]
        infos = real_infolist(self)
        for info in infos:
            if info.filename in filenames:
                info.flag_bits |= 0x1
        return infos

    monkeypatch.setattr(zipfile.ZipFile, "infolist", patched)


def test_classifies_indexable_dependency_noise_and_gap(tmp_path: Path) -> None:
    _make_zip(
        tmp_path / "mixed.zip",
        {
            "notes.md": "real content",
            "node_modules/pkg/readme.md": "vendored, should not count as indexable",
            "deck.mysteryformat": "unsupported extension",
        },
    )

    result = run_archive_survey(tmp_path)

    [archive] = result.archives
    assert archive.readable
    assert archive.total_members == 3
    assert archive.indexable_by_type == {"markdown": 1}
    assert archive.dependency_noise == 1
    assert archive.gap == 1
    assert archive.noise_ratio == 1 / 3


def test_packaging_noise_is_counted_separately_from_dependency_noise(tmp_path: Path) -> None:
    _make_zip(
        tmp_path / "mac.zip",
        {
            "report.txt": "real",
            "__MACOSX/._report.txt": "appledouble stub",
            ".DS_Store": "finder junk",
        },
    )

    result = run_archive_survey(tmp_path)

    [archive] = result.archives
    assert archive.packaging_noise == 2
    assert archive.dependency_noise == 0
    assert archive.indexable_by_type == {"text": 1}


def test_encrypted_member_is_counted_and_excluded_from_noise_ratio(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _make_zip(tmp_path / "protected.zip", {"plain.txt": "readable", "secret.txt": "shh"})
    _mark_encrypted(monkeypatch, {"secret.txt"})

    result = run_archive_survey(tmp_path)

    [archive] = result.archives
    assert archive.readable
    assert archive.total_members == 2
    assert archive.encrypted == 1
    assert archive.indexable_by_type == {"text": 1}
    # encrypted members don't count as noise — they're refused for a
    # different reason than dependency/build-output or packaging junk.
    assert archive.noise_ratio == 0.0


def test_nested_archive_is_its_own_bucket_not_gap(tmp_path: Path) -> None:
    _make_zip(tmp_path / "outer.zip", {"inner.zip": "pretend zip bytes"})

    result = run_archive_survey(tmp_path)

    [archive] = result.archives
    assert archive.nested_archive_refused == 1
    assert archive.gap == 0


def test_unreadable_archive_is_counted_not_raised(tmp_path: Path) -> None:
    bad = tmp_path / "corrupt.zip"
    bad.write_bytes(b"not actually a zip file")

    result = run_archive_survey(tmp_path)

    [archive] = result.archives
    assert not archive.readable
    assert archive.error is not None
    assert result.totals["unreadable_count"] == 1


def test_no_extraction_directory_is_ever_created(tmp_path: Path) -> None:
    _make_zip(tmp_path / "a.zip", {"notes.md": "x" * 10_000})
    before = {p for p in Path(tempfile_gettempdir()).glob("corpus-zip-*")}

    run_archive_survey(tmp_path)

    after = {p for p in Path(tempfile_gettempdir()).glob("corpus-zip-*")}
    assert after == before, "archive survey must never create the zip connector's temp dirs"


def tempfile_gettempdir() -> str:
    import tempfile

    return tempfile.gettempdir()


def test_totals_aggregate_across_archives(tmp_path: Path) -> None:
    _make_zip(tmp_path / "a.zip", {"a.txt": "x"})
    _make_zip(tmp_path / "b.zip", {"b.txt": "y", "node_modules/x/readme.md": "vendored"})

    result = run_archive_survey(tmp_path)

    assert result.totals["archive_count"] == 2
    assert result.totals["total_members"] == 3
    assert result.totals["indexable_total"] == 2
    assert result.totals["dependency_noise"] == 1


def test_sorted_by_member_count_descending(tmp_path: Path) -> None:
    _make_zip(tmp_path / "small.zip", {"a.txt": "x"})
    _make_zip(tmp_path / "big.zip", {"a.txt": "x", "b.txt": "y", "c.txt": "z"})

    result = run_archive_survey(tmp_path)

    assert [a.path for a in result.archives] == ["big.zip", "small.zip"]


def test_respects_excludes(tmp_path: Path) -> None:
    _make_zip(tmp_path / "keep.zip", {"a.txt": "x"})
    _make_zip(tmp_path / "skip" / "ignored.zip", {"a.txt": "x"})

    result = run_archive_survey(tmp_path, excludes=("skip",))

    assert [a.path for a in result.archives] == ["keep.zip"]
