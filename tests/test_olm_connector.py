"""Tests for `corpus.connectors.olm`.

Fixtures are hand-built rather than sampled from a real archive: every
behaviour under test is a property of Outlook's format, and the real archive
that motivated this connector is a large work mailbox.

The three shapes that matter — a self-mirroring archive, bodies that are
escaped HTML rather than text, and replies that quote their whole history —
were each measured on that archive and each costs money or content if the
connector gets it wrong. See `corpus/connectors/olm.py`'s module docstring.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from corpus.connectors.olm import MESSAGE_ROOT, MIRROR_PREFIX, OlmConnector, trim_quoted_history


def _message(
    subject: str = "Status update",
    body: str = "All done.",
    sender: str = "someone@example.com",
    recipient: str = "dan@example.com",
    sent: str = "2011-07-20T06:02:11",
) -> bytes:
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<emails xml:space="preserve" elementCount="1"><email xml:space="preserve">'
        f"<OPFMessageCopySubject>{subject}</OPFMessageCopySubject>"
        f"<OPFMessageCopyBody>{body}</OPFMessageCopyBody>"
        f"<OPFMessageCopySentTime>{sent}</OPFMessageCopySentTime>"
        "<OPFMessageCopyFromAddresses><emailAddress "
        f'OPFContactEmailAddressAddress="{sender}" OPFContactEmailAddressName="A Sender"/>'
        "</OPFMessageCopyFromAddresses>"
        "<OPFMessageCopyToAddresses><emailAddress "
        f'OPFContactEmailAddressAddress="{recipient}"/>'
        "</OPFMessageCopyToAddresses>"
        "</email></emails>"
    ).encode()


def _archive(tmp_path: Path, members: dict[str, bytes], name: str = "mail.olm") -> Path:
    path = tmp_path / name
    with zipfile.ZipFile(path, "w") as zf:
        for member, payload in members.items():
            zf.writestr(member, payload)
    return path


def _load(tmp_path: Path, **kwargs: object) -> list:
    return list(OlmConnector(source_type="mail", path=tmp_path, **kwargs).load())  # type: ignore[arg-type]


# --- the self-mirroring archive --------------------------------------------


def test_mirror_tree_is_skipped_by_default(tmp_path: Path) -> None:
    # A real archive carries roughly as many mirrored messages as# direct ones. Indexing both doubles the embedding bill for nothing.
    _archive(
        tmp_path,
        {
            f"{MESSAGE_ROOT}archive PST/Inbox/message_00001.xml": _message(),
            f"{MESSAGE_ROOT}{MIRROR_PREFIX}archive PST/Inbox/message_00001.xml": _message(),
        },
    )

    docs = _load(tmp_path)

    assert len(docs) == 1
    assert not docs[0].source_key.startswith(MIRROR_PREFIX)


def test_the_direct_copy_is_the_one_kept(tmp_path: Path) -> None:
    # Not interchangeable: in a few folders of the real archive the direct store
    # holds messages the mirror lacks, and no folder matches on total bytes.
    # Keeping the mirror instead would lose real messages.
    _archive(
        tmp_path,
        {
            f"{MESSAGE_ROOT}archive PST/Inbox/message_1.xml": _message(subject="kept"),
            f"{MESSAGE_ROOT}archive PST/Inbox/message_2.xml": _message(subject="direct only"),
            f"{MESSAGE_ROOT}{MIRROR_PREFIX}archive PST/Inbox/message_1.xml": _message(
                subject="kept"
            ),
        },
    )

    titles = sorted(d.title for d in _load(tmp_path))

    assert titles == ["direct only", "kept"]


def test_mirror_can_be_included_deliberately(tmp_path: Path) -> None:
    """What the flag is for, per the module docstring: the mirror is NOT an
    exact copy, and including it reaches messages the direct tree lacks.

    It yields the UNION, not the sum -- an identical copy is collapsed by the
    near-duplicate skip, so turning the flag off buys coverage without
    paying to embed the same message twice.
    """
    _archive(
        tmp_path,
        {
            f"{MESSAGE_ROOT}archive PST/Inbox/message_1.xml": _message(),
            f"{MESSAGE_ROOT}{MIRROR_PREFIX}archive PST/Inbox/message_1.xml": _message(),
            f"{MESSAGE_ROOT}{MIRROR_PREFIX}archive PST/Inbox/message_2.xml": _message(
                body="Only the mirror has this one."
            ),
        },
    )

    keys = {d.source_key for d in _load(tmp_path, skip_mirror_tree=False)}
    assert len(keys) == 2, "an identical copy was indexed twice"
    assert any("message_2" in k for k in keys), "a mirror-only message was lost"


def test_the_mirror_only_message_is_absent_when_the_mirror_is_skipped(tmp_path: Path) -> None:
    """The negative control for the test above."""
    _archive(
        tmp_path,
        {
            f"{MESSAGE_ROOT}archive PST/Inbox/message_1.xml": _message(),
            f"{MESSAGE_ROOT}{MIRROR_PREFIX}archive PST/Inbox/message_2.xml": _message(
                body="Only the mirror has this one."
            ),
        },
    )
    assert len(_load(tmp_path)) == 1


# --- bodies are escaped HTML, not text -------------------------------------


def test_escaped_html_body_becomes_readable_text(tmp_path: Path) -> None:
    # Markup was 84% of the body across a 2,000-message sample. Without this,
    # a chunk is mostly CSS and Word's font tables.
    body = (
        "&lt;html&gt;&lt;head&gt;&lt;style&gt;p { font-family: Cambria; }&lt;/style&gt;"
        "&lt;/head&gt;&lt;body&gt;&lt;p&gt;Training is complete.&lt;/p&gt;"
        "&lt;p&gt;Thanks&lt;/p&gt;&lt;/body&gt;&lt;/html&gt;"
    )
    _archive(tmp_path, {f"{MESSAGE_ROOT}s/f/message_1.xml": _message(body=body)})

    text = _load(tmp_path)[0].raw["body"]

    assert "Training is complete." in text
    assert "Thanks" in text
    assert "Cambria" not in text  # the stylesheet is gone
    # Check the body specifically: the connector's own "From:" header legitimately
    # contains angle brackets around an address, so the whole-text assertion this
    # replaced was testing the header, not the markup stripping.
    body_only = text.split("\n\n", 1)[1]
    assert "<" not in body_only and "&lt;" not in body_only


def test_adjacent_block_tags_do_not_weld_words_together(tmp_path: Path) -> None:
    # Tags become newlines rather than empty strings; otherwise "</p><p>"
    # produces "oneword" and both words stop being searchable.
    body = "&lt;p&gt;first&lt;/p&gt;&lt;p&gt;second&lt;/p&gt;"
    _archive(tmp_path, {f"{MESSAGE_ROOT}s/f/message_1.xml": _message(body=body)})

    text = _load(tmp_path)[0].raw["body"]

    assert "firstsecond" not in text
    assert "first" in text and "second" in text


def test_plain_text_body_survives_unchanged(tmp_path: Path) -> None:
    # Bodies are only sometimes HTML — the same field is often plain text,
    # frequently Japanese in this archive.
    _archive(
        tmp_path,
        {f"{MESSAGE_ROOT}s/f/message_1.xml": _message(body="ご確認のほどよろしくお願いいたします。")},
    )

    assert "ご確認のほどよろしくお願いいたします。" in _load(tmp_path)[0].raw["body"]


# --- quoted reply history --------------------------------------------------


@pytest.mark.parametrize(
    "marker",
    [
        "From: Someone <s@example.com>",
        "-----Original Message-----",
        "差出人: 山田",
        "On Tuesday, someone wrote:",
        "____________________________",
    ],
)
def test_quoted_history_is_trimmed_at_each_marker(marker: str) -> None:
    text = f"My actual reply.\n\n{marker}\nEverything below is the quoted thread."

    assert trim_quoted_history(text) == "My actual reply."


def test_japanese_markers_matter_as_much_as_english() -> None:
    # This archive is roughly half Japanese correspondence. Matching only the
    # English headers would leave that half of it fully quoted — and quoting
    # is 78% of the text.
    text = "承知しました。\n\n差出人: 田中\n件名: RE: 会議\n\n(quoted thread)"

    assert trim_quoted_history(text) == "承知しました。"


def test_a_bare_forward_keeps_its_quoted_body() -> None:
    # 6.6% of real messages trim to nothing: a forward with no added comment
    # IS its quoted content. Dropping those loses the message entirely.
    text = "From: Someone <s@example.com>\nSubject: FYI\n\nThe actual content."

    assert trim_quoted_history(text) == text


def test_trimming_can_be_turned_off(tmp_path: Path) -> None:
    body = "New text.&#13;&#10;&#13;&#10;From: X&#13;&#10;quoted"
    _archive(tmp_path, {f"{MESSAGE_ROOT}s/f/message_1.xml": _message(body=body)})

    assert "quoted" in _load(tmp_path, trim_quotes=False)[0].raw["body"]


# --- folder scoping --------------------------------------------------------


def test_folders_filter_scopes_to_matching_patterns(tmp_path: Path) -> None:
    # An .olm mixes years and employers in one file; scoping by folder is how
    # you index part of a mailbox without paying to embed all of it.
    _archive(
        tmp_path,
        {
            f"{MESSAGE_ROOT}archive PST/Inbox 2010/message_1.xml": _message(subject="want"),
            f"{MESSAGE_ROOT}archive PST/Inbox 2011/message_1.xml": _message(subject="want too"),
            f"{MESSAGE_ROOT}archive PST/Sent Items 2010/message_1.xml": _message(subject="no"),
        },
    )

    titles = sorted(d.title for d in _load(tmp_path, folders=["archive PST/Inbox*"]))

    assert titles == ["want", "want too"]


def test_no_folder_filter_means_every_folder(tmp_path: Path) -> None:
    _archive(
        tmp_path,
        {
            f"{MESSAGE_ROOT}a/f1/message_1.xml": _message(body="From folder one."),
            f"{MESSAGE_ROOT}b/f2/message_1.xml": _message(body="From folder two."),
        },
    )

    # Distinct bodies on purpose: identical ones are collapsed by the
    # near-duplicate skip, which would make this pass whether or not the
    # second folder was ever walked.
    assert len(_load(tmp_path)) == 2


# --- document shape --------------------------------------------------------


def test_participants_and_date_ride_in_the_body(tmp_path: Path) -> None:
    # The median message is 224 characters once quotes are trimmed. Without a
    # header in the text, most of the archive is unsearchable by participant.
    _archive(tmp_path, {f"{MESSAGE_ROOT}s/f/message_1.xml": _message()})

    text = _load(tmp_path)[0].raw["body"]

    assert "A Sender <someone@example.com>" in text
    assert "dan@example.com" in text
    assert "2011-07-20T06:02:11" in text
    assert "All done." in text


def test_source_key_is_the_member_path_not_the_message_id(tmp_path: Path) -> None:
    # Message-ID is absent from ~18% of real messages and repeats across the
    # mirror; the member path is unique and always present, which is what
    # orphan pruning needs to be stable across runs.
    _archive(tmp_path, {f"{MESSAGE_ROOT}archive PST/Inbox/message_00042.xml": _message()})

    assert _load(tmp_path)[0].source_key == "archive PST/Inbox/message_00042.xml"


def test_recipient_list_is_capped(tmp_path: Path) -> None:
    # A mailing-list blast would otherwise swamp the message's own text.
    addrs = "".join(
        f'<emailAddress OPFContactEmailAddressAddress="p{i}@example.com"/>' for i in range(60)
    )
    payload = (
        "<emails><email><OPFMessageCopySubject>Blast</OPFMessageCopySubject>"
        "<OPFMessageCopyBody>Short.</OPFMessageCopyBody>"
        f"<OPFMessageCopyToAddresses>{addrs}</OPFMessageCopyToAddresses>"
        "</email></emails>"
    ).encode()
    _archive(tmp_path, {f"{MESSAGE_ROOT}s/f/message_1.xml": payload})

    text = _load(tmp_path)[0].raw["body"]

    assert "p0@example.com" in text
    assert "p59@example.com" not in text
    assert "(+48 more)" in text


def test_empty_message_is_skipped_not_failed(tmp_path: Path) -> None:
    # Permanent, not transient: counting it as failed_files would suppress
    # orphan pruning forever, since it fails identically on every run.
    empty = b"<emails><email><OPFMessageCopyBody></OPFMessageCopyBody></email></emails>"
    _archive(
        tmp_path,
        {
            f"{MESSAGE_ROOT}s/f/message_1.xml": empty,
            f"{MESSAGE_ROOT}s/f/message_2.xml": _message(),
        },
    )
    connector = OlmConnector(source_type="mail", path=tmp_path)

    docs = list(connector.load())

    assert len(docs) == 1
    assert connector.failed_files == 0
    assert connector.skipped_files >= 1


# --- enumeration completeness ----------------------------------------------


def test_unopenable_archive_raises_rather_than_yielding_nothing(tmp_path: Path) -> None:
    # A connector that yields zero documents without raising makes the
    # ingester treat every previously-indexed chunk as an orphan and delete
    # it. Raising is what keeps the index intact.
    (tmp_path / "broken.olm").write_bytes(b"not a zip at all")

    with pytest.raises(OSError):
        _load(tmp_path)


def test_missing_directory_raises(tmp_path: Path) -> None:
    connector = OlmConnector(source_type="mail", path=tmp_path / "nope")

    with pytest.raises(FileNotFoundError):
        list(connector.load())


def test_counters_reset_between_runs(tmp_path: Path) -> None:
    # load() is called once per ingest; stale counts from a previous run
    # would suppress pruning on a clean one.
    _archive(
        tmp_path,
        {
            f"{MESSAGE_ROOT}s/f/message_1.xml": _message(),
            f"{MESSAGE_ROOT}{MIRROR_PREFIX}s/f/message_1.xml": _message(),
        },
    )
    connector = OlmConnector(source_type="mail", path=tmp_path)

    list(connector.load())
    first = connector.skipped_files
    list(connector.load())

    assert connector.skipped_files == first
