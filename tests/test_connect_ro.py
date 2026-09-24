"""One read-only SQLite open, instead of thirteen hand-written URIs.

The URI form is the only way to get `mode=ro`, and a URI treats `?` and `#`
as syntax: an unquoted path containing either named a different file, or
none. A real archive's folder names contain both.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from corpus.util.sqlite_ro import connect_ro


@pytest.mark.parametrize("name", ["plain.db", "take #2.db", "what?.db", "100%.db", "a b.db"])
def test_opens_the_file_it_was_given(tmp_path: Path, name: str) -> None:
    path = tmp_path / name
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE t (x)")
    conn.execute("INSERT INTO t VALUES (42)")
    conn.commit()
    conn.close()

    ro = connect_ro(path)
    try:
        assert ro.execute("SELECT x FROM t").fetchone() == (42,)
    finally:
        ro.close()


def test_refuses_writes(tmp_path: Path) -> None:
    path = tmp_path / "x.db"
    sqlite3.connect(path).close()
    ro = connect_ro(path)
    try:
        with pytest.raises(sqlite3.OperationalError):
            ro.execute("CREATE TABLE t (x)")
    finally:
        ro.close()


def test_never_creates_a_missing_file(tmp_path: Path) -> None:
    with pytest.raises(sqlite3.OperationalError):
        connect_ro(tmp_path / "absent.db")
    assert not (tmp_path / "absent.db").exists()
