"""Open a SQLite database strictly read-only."""

from __future__ import annotations

import sqlite3
import urllib.parse
from pathlib import Path
from typing import Any


def connect_ro(path: Path | str, *, immutable: bool = False, **kwargs: Any) -> sqlite3.Connection:
    """`mode=ro` needs the URI form, where `?`, `#` and `%` are syntax, so the
    path is quoted: unquoted, `take #2.db` opened `take `. Never creates the
    file, and SQLite refuses every write on the connection.

    `immutable=True` also promises SQLite the file will not change while it
    is open, so a WAL-mode database is read without creating `-shm`/`-wal`
    files beside it. Only for files nothing else is writing: a concurrent
    writer's changes would be missed, not merged."""
    params = "mode=ro&immutable=1" if immutable else "mode=ro"
    uri = f"file:{urllib.parse.quote(Path(path).as_posix())}?{params}"
    conn: sqlite3.Connection = sqlite3.connect(uri, uri=True, **kwargs)
    return conn
