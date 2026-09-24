"""Open a SQLite database strictly read-only."""

from __future__ import annotations

import sqlite3
import urllib.parse
from pathlib import Path
from typing import Any


def connect_ro(path: Path | str, **kwargs: Any) -> sqlite3.Connection:
    """`mode=ro` needs the URI form, where `?`, `#` and `%` are syntax, so the
    path is quoted: unquoted, `take #2.db` opened `take `. Never creates the
    file, and SQLite refuses every write on the connection."""
    uri = f"file:{urllib.parse.quote(Path(path).as_posix())}?mode=ro"
    conn: sqlite3.Connection = sqlite3.connect(uri, uri=True, **kwargs)
    return conn
