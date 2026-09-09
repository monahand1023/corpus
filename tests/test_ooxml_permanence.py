"""Tests for `corpus.util.ooxml` — permanent vs transient unreadability.

The distinction decides whether a connector reports a file as `failed_files`
(suppresses orphan pruning, so a momentarily-unreadable file's chunks are not
deleted) or `skipped_files` (does not suppress). Both errors cost something,
and they are not symmetric: a stale suppressed prune leaves an out-of-date
index, recoverable with `--prune-anyway`, while wrongly calling something
permanent lets the next prune delete a real document's chunks. So the bar for
"permanent" is deliberately high.
"""

from __future__ import annotations

from pathlib import Path

from corpus.util.ooxml import is_office_lock_file, permanent_read_failure_reason

OLE2 = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"


def test_office_lock_file_is_permanent(tmp_path: Path) -> None:
    path = tmp_path / "~$quarterly.docx"
    path.write_bytes(b"\x00" * 16)

    assert is_office_lock_file(path)
    assert "lock file" in (permanent_read_failure_reason(path) or "")


def test_legacy_ole2_with_a_modern_extension_is_permanent(tmp_path: Path) -> None:
    # Measured on a real archive: three `.docx` files that were legacy `.doc`
    # renamed rather than converted. python-docx can never read them.
    path = tmp_path / "renamed.docx"
    path.write_bytes(OLE2 + b"\x00" * 64)

    reason = permanent_read_failure_reason(path)

    assert reason is not None
    assert "OLE2" in reason
    assert "convert" in reason  # the message has to be actionable


def test_a_real_ooxml_file_is_not_flagged(tmp_path: Path) -> None:
    path = tmp_path / "fine.docx"
    path.write_bytes(b"PK\x03\x04" + b"\x00" * 64)

    assert permanent_read_failure_reason(path) is None


def test_an_empty_file_is_treated_as_transient(tmp_path: Path) -> None:
    # An interrupted copy leaves a zero-byte file that may be complete a
    # minute later. Calling it permanent would let the next prune delete that
    # document's chunks.
    path = tmp_path / "midcopy.docx"
    path.write_bytes(b"")

    assert permanent_read_failure_reason(path) is None


def test_a_corrupt_non_zip_is_treated_as_transient(tmp_path: Path) -> None:
    # A large file being copied over a network is not a valid zip YET. The
    # connector's own error path counts it as a failure, which suppresses
    # pruning — the conservative direction.
    path = tmp_path / "partial.docx"
    path.write_bytes(b"this is not any kind of office file")

    assert permanent_read_failure_reason(path) is None


def test_an_unreadable_path_is_transient(tmp_path: Path) -> None:
    assert permanent_read_failure_reason(tmp_path / "does-not-exist.docx") is None


def test_an_ordinary_name_is_not_a_lock_file(tmp_path: Path) -> None:
    assert not is_office_lock_file(tmp_path / "budget~$draft.docx")
