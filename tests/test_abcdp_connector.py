"""Tests for `corpus.connectors.abcdp` (Apple Contacts `.abbu` backups).

Fixtures are built with `plistlib.dump`, which writes the same binary format
Address Book does — no real contact data, and no sampled personal records.
"""

from __future__ import annotations

import plistlib
from datetime import datetime
from pathlib import Path

import pytest

from corpus.connectors.abcdp import AbcdpConnector, _label, _person_text


def _write(tmp_path: Path, name: str, person: dict) -> Path:
    src = tmp_path / "Sources" / "UUID-1" / "Metadata"
    src.mkdir(parents=True, exist_ok=True)
    path = src / name
    with open(path, "wb") as fh:
        plistlib.dump(person, fh, fmt=plistlib.FMT_BINARY)
    return path


def _multi(*pairs: tuple[str, object]) -> dict:
    return {"labels": [p[0] for p in pairs], "values": [p[1] for p in pairs]}


def _load(tmp_path: Path) -> list:
    return list(AbcdpConnector(source_type="contacts", path=tmp_path).load())


# --- field extraction -------------------------------------------------------


def test_apple_label_syntax_is_stripped() -> None:
    # Address Book wraps its built-in labels; a user's own label arrives bare.
    assert _label("_$!<Home>!$_") == "Home"
    assert _label("_$!<Mobile>!$_") == "Mobile"
    assert _label("ski house") == "ski house"
    assert _label("") == ""


def test_person_carries_name_company_and_every_contact_method(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "A:ABPerson.abcdp",
        {
            "First": "Robin",
            "Last": "Fairweather",
            "Organization": "Northwind Support",
            "JobTitle": "Support Lead",
            "Phone": _multi(("_$!<Mobile>!$_", "+819000000000"), ("", "555-0100")),
            "Email": _multi(("_$!<Work>!$_", "robin@example.com")),
        },
    )

    docs = _load(tmp_path)

    assert len(docs) == 1
    body = docs[0].raw["body"]
    assert docs[0].title == "Robin Fairweather"
    assert "Support Lead, Northwind Support" in body
    assert "Phone (Mobile): +819000000000" in body
    assert "555-0100" in body
    assert "Email (Work): robin@example.com" in body


def test_addresses_are_assembled_in_reading_order(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "B:ABPerson.abcdp",
        {
            "First": "Aiko",
            "Address": _multi(
                (
                    "_$!<Home>!$_",
                    {
                        "Street": "さくら町1-2-3",
                        "City": "港区",
                        "State": "東京都",
                        "ZIP": "100-0001",
                        "Country": "Japan",
                        "CountryCode": "jp",
                    },
                )
            ),
        },
    )

    body = _load(tmp_path)[0].raw["body"]

    assert "Address (Home): さくら町1-2-3, 港区, 東京都, 100-0001, Japan" in body
    assert "CountryCode" not in body  # an internal field, not part of the address


def test_the_note_is_preserved_verbatim(tmp_path: Path) -> None:
    # Usually the highest-signal field, and the only one that says why this
    # person is in the address book at all.
    note = "Employer: Northwind\nPhone 1 type: mobile\nNotes: met at the depot\n"
    _write(tmp_path, "C:ABPerson.abcdp", {"First": "Sam", "Note": note})

    body = _load(tmp_path)[0].raw["body"]

    assert "met at the depot" in body
    assert "Employer: Northwind" in body


def test_labels_and_values_of_unequal_length_do_not_raise(tmp_path: Path) -> None:
    # Real backups contain these. Pairing is by index; a missing label falls
    # back to empty rather than losing the value or raising.
    _write(
        tmp_path,
        "D:ABPerson.abcdp",
        {
            "First": "Pat",
            "Phone": {"labels": ["_$!<Home>!$_"], "values": ["555-0101", "555-0102"]},
        },
    )

    body = _load(tmp_path)[0].raw["body"]

    assert "555-0101" in body
    assert "555-0102" in body


def test_organization_only_contact_is_titled_by_company(tmp_path: Path) -> None:
    _write(tmp_path, "E:ABPerson.abcdp", {"Organization": "Seattle City Light"})

    docs = _load(tmp_path)

    assert docs[0].title == "Seattle City Light"


def test_timestamps_become_iso_strings(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "F:ABPerson.abcdp",
        {
            "First": "Dana",
            "Creation": datetime(2023, 3, 24, 3, 3, 17),
            "Modification": datetime(2023, 3, 25, 4, 4, 18),
        },
    )

    doc = _load(tmp_path)[0]

    assert doc.created_at == "2023-03-24T03:03:17"
    assert doc.updated_at == "2023-03-25T04:04:18"


# --- the failure modes that matter -----------------------------------------


def test_a_contentless_tombstone_is_skipped_not_failed(tmp_path: Path) -> None:
    # A record with only a UID and timestamps. Permanent, so skipped_files —
    # counting it as failed_files would suppress orphan pruning on every run
    # forever, for a condition that never clears.
    _write(tmp_path, "G:ABPerson.abcdp", {"UID": "X:ABPerson", "ABPersonFlags": 0})
    _write(tmp_path, "H:ABPerson.abcdp", {"First": "Real"})
    connector = AbcdpConnector(source_type="contacts", path=tmp_path)

    docs = list(connector.load())

    assert len(docs) == 1
    assert connector.skipped_files == 1
    assert connector.failed_files == 0


def test_unreadable_plist_counts_as_failed(tmp_path: Path) -> None:
    # Transient-shaped: it suppresses pruning rather than letting a run delete
    # contacts it could not read this time.
    src = tmp_path / "Sources" / "U" / "Metadata"
    src.mkdir(parents=True)
    (src / "bad:ABPerson.abcdp").write_bytes(b"not a plist")
    (src / "ok:ABPerson.abcdp").write_bytes(
        plistlib.dumps({"First": "Fine"}, fmt=plistlib.FMT_BINARY)
    )
    connector = AbcdpConnector(source_type="contacts", path=tmp_path)

    docs = list(connector.load())

    assert len(docs) == 1
    assert connector.failed_files == 1


def test_missing_directory_raises(tmp_path: Path) -> None:
    # Must raise, not yield nothing: an empty enumeration makes the ingester
    # treat every indexed contact as an orphan and delete it.
    connector = AbcdpConnector(source_type="contacts", path=tmp_path / "nope")

    with pytest.raises(FileNotFoundError):
        list(connector.load())


def test_source_key_is_stable_across_runs(tmp_path: Path) -> None:
    _write(tmp_path, "I:ABPerson.abcdp", {"First": "Stable"})

    first = _load(tmp_path)[0].source_key
    second = _load(tmp_path)[0].source_key

    assert first == second
    assert first.endswith("I:ABPerson.abcdp")


def test_empty_person_text_helper_returns_placeholder_display() -> None:
    display, body = _person_text({"UID": "X"})

    assert display == "(unnamed contact)"
    assert body == ""
