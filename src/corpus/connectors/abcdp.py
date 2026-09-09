"""Apple Contacts connector — one document per person in an `.abbu` backup.

An Address Book backup (`Contacts - <date>.abbu`) is a directory bundle. Each
person is a binary property list at
`Sources/<source-uuid>/Metadata/<person-uuid>:ABPerson.abcdp`, which
`plistlib` reads from the standard library — no dependency, no extra.

**Do not reach for the `.abcddb` in the bundle root.** It looks like the
canonical Core Data store and it is the obvious thing to point a connector
at, but in a real 379-contact backup it held 2 rows, neither with a name. A
connector reading it succeeds, yields nothing, and — because yielding nothing
is indistinguishable from "the source is empty" — hands the ingester an empty
`seen_ids`, which prunes every previously indexed contact. The per-person
plists are where the data actually lives.

Contacts are short, so the whole person becomes one document: name, company,
job title, every phone, email and address, birthday, and the free-text note.
The note is often the highest-signal field ("met at the depot") and is exactly
what someone searches for when they cannot remember a name.

Apple's multi-value fields (`Phone`, `Email`, `Address`) share one shape:
`{"labels": [...], "values": [...]}` in parallel order. Labels arrive wrapped
in Apple's internal syntax — `_$!<Home>!$_` — which is stripped to `Home`.
"""

from __future__ import annotations

import logging
import os
import plistlib
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from corpus.connectors.discovery import discover_files
from corpus.types import SourceDocument

logger = logging.getLogger(__name__)

# `_$!<Home>!$_` -> `Home`. Apple wraps its built-in labels this way; custom
# labels the user typed arrive bare and pass through untouched.
_APPLE_LABEL = re.compile(r"^_\$!<(.+)>!\$_$")

_ADDRESS_ORDER = ("Street", "City", "State", "ZIP", "Country")


def _label(raw: str) -> str:
    match = _APPLE_LABEL.match(raw or "")
    return match.group(1) if match else (raw or "")


def _multi_values(field: Any) -> list[tuple[str, Any]]:
    """(label, value) pairs from one Apple multi-value field.

    `labels` and `values` are parallel arrays. They can disagree in length in
    real files, so pairing is by index with a missing label falling back to
    empty rather than raising.
    """
    if not isinstance(field, dict):
        return []
    values = field.get("values") or []
    labels = field.get("labels") or []
    out: list[tuple[str, Any]] = []
    for i, value in enumerate(values):
        out.append((_label(labels[i] if i < len(labels) else ""), value))
    return out


def _format_address(value: Any) -> str:
    if not isinstance(value, dict):
        return str(value).strip()
    parts = [str(value[k]).strip() for k in _ADDRESS_ORDER if value.get(k)]
    return ", ".join(parts)


def _person_text(person: dict[str, Any]) -> tuple[str, str]:
    """Return (display name, document body) for one contact."""
    name = " ".join(
        str(person[k]).strip()
        for k in ("First", "Middle", "Last")
        if person.get(k)
    ).strip()
    organization = str(person.get("Organization", "")).strip()
    display = name or organization or "(unnamed contact)"

    lines: list[str] = []
    if name:
        lines.append(name)
    if organization:
        title = str(person.get("JobTitle", "")).strip()
        lines.append(f"{title}, {organization}" if title else organization)

    for field, heading in (("Phone", "Phone"), ("Email", "Email")):
        for label, value in _multi_values(person.get(field)):
            text = str(value).strip()
            if text:
                lines.append(f"{heading}{f' ({label})' if label else ''}: {text}")

    for label, value in _multi_values(person.get("Address")):
        text = _format_address(value)
        if text:
            lines.append(f"Address{f' ({label})' if label else ''}: {text}")

    birthday = person.get("Birthday")
    if birthday:
        lines.append(f"Birthday: {birthday}")

    note = str(person.get("Note", "")).strip()
    if note:
        # Kept last and unmodified: it is free text the user wrote themselves,
        # and it is usually the only field that says WHY this person is in the
        # address book.
        lines.append(f"\n{note}")

    return display, "\n".join(lines).strip()


def _iso(value: Any) -> str | None:
    """A plist `datetime` as an ISO string, or None for anything else.

    Address Book writes real `datetime` objects here, but a hand-edited or
    partially-synced record can carry a string or nothing at all.
    """
    isoformat = getattr(value, "isoformat", None)
    return str(isoformat()) if callable(isoformat) else None


class AbcdpConnector:
    def __init__(
        self,
        source_type: str,
        path: Path | str,
        glob: str = "**/*.abcdp",
    ):
        self.source_type = source_type
        self._root = Path(os.path.expanduser(str(path))).resolve()
        self._glob = glob
        self.failed_files = 0
        self.skipped_files = 0

    def load(self) -> Iterable[SourceDocument]:
        # Reset per run so a stale count cannot suppress orphan pruning
        # forever. Same contract as every other file connector.
        self.failed_files = 0
        self.skipped_files = 0

        if not self._root.is_dir():
            raise FileNotFoundError(
                f"Abcdp source '{self.source_type}': directory not found: {self._root}"
            )

        for path in discover_files(self._root, self._glob):
            try:
                with open(path, "rb") as fh:
                    person = plistlib.load(fh)
            except (OSError, plistlib.InvalidFileException, ValueError) as e:
                logger.warning(
                    "Abcdp source '%s': cannot read %s: %s", self.source_type, path, e
                )
                self.failed_files += 1
                continue

            if not isinstance(person, dict):
                self.skipped_files += 1
                continue

            display, body = _person_text(person)
            if not body:
                # A record with a UID and timestamps but no name, number, or
                # note — a sync tombstone. Permanent, so skipped not failed.
                self.skipped_files += 1
                continue

            yield SourceDocument(
                source_type=self.source_type,
                source_key=str(path.relative_to(self._root)),
                title=display,
                url=None,
                created_at=_iso(person.get("Creation")),
                updated_at=_iso(person.get("Modification")),
                raw={"body": body, "path": str(path)},
            )
