"""The Outlook connector kept every copy of a message; ten others do not.

`MarkdownConnector`, `TextConnector`, `PdfConnector` and seven more skip a
document whose body fingerprint they have already seen in the same `load()`.
`OlmConnector` did not — and an Outlook archive is the format where the same
message appears most: once in a folder, once in Deleted Items, once in a
mirrored PST export.

Measured on a live archive: 3,087 chunks were losslessly removable across
its filesystem sources, and 2,381 of them — 77% — were duplicate messages
inside ONE Outlook source. Some duplicated a message in a DIFFERENT folder
and some duplicated one in the SAME folder, which is why excluding a folder
could not fix it: the copy to keep was often inside the folder to drop.

The header rides in the body (`From:`/`To:`/`Date:` + text), so the
fingerprint covers participants and date as well as prose: two genuinely
different messages that happen to share a short body are still distinct.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

from corpus.connectors.olm import OlmConnector

MESSAGE = """<?xml version="1.0"?>
<emails>
  <email>
    <OPFMessageCopySubject>{subject}</OPFMessageCopySubject>
    <OPFMessageCopyBody>{body}</OPFMessageCopyBody>
    <OPFMessageCopySentTime>2022-03-04 10:00:00 +0000</OPFMessageCopySentTime>
  </email>
</emails>
"""


def _archive(tmp_path: Path, members: dict[str, tuple[str, str]]) -> Path:
    """members: archive-relative path -> (subject, body)."""
    olm = tmp_path / "mail.olm"
    with zipfile.ZipFile(olm, "w") as z:
        for rel, (subject, body) in members.items():
            z.writestr(
                f"Local/com.microsoft.__Messages/{rel}",
                MESSAGE.format(subject=subject, body=body),
            )
    return olm


def _load(tmp_path: Path) -> list:
    return list(OlmConnector(source_type="mail", path=str(tmp_path)).load())


def test_the_same_message_in_two_folders_is_yielded_once(tmp_path):
    _archive(
        tmp_path,
        {
            "Inbox/message_00001.xml": ("Roof estimate", "The quote came in at 4,200."),
            "Deleted Items/message_01218.xml": (
                "Roof estimate", "The quote came in at 4,200."
            ),
        },
    )
    assert len(_load(tmp_path)) == 1


def test_the_same_message_twice_in_ONE_folder_is_yielded_once(tmp_path):
    """The case a folder-level exclude cannot fix: the copy to keep is inside
    the folder that would be dropped."""
    _archive(
        tmp_path,
        {
            "Archive/message_00081.xml": ("Budget", "Approved, see attached sheet."),
            "Archive/message_01212.xml": ("Budget", "Approved, see attached sheet."),
        },
    )
    assert len(_load(tmp_path)) == 1


def test_two_different_messages_both_survive(tmp_path):
    """The negative control. Without it the tests above pass on a connector
    that yields nothing."""
    _archive(
        tmp_path,
        {
            "Inbox/message_0001.xml": ("Roof estimate", "The quote came in at 4,200."),
            "Inbox/message_0002.xml": ("Fence estimate", "That one was 1,800 including the gate."),
        },
    )
    assert len(_load(tmp_path)) == 2


def test_the_first_copy_seen_is_the_one_kept(tmp_path):
    """Members are walked in sorted order, so the choice is deterministic —
    a run that kept a different copy each time would churn chunk ids and
    re-embed the archive."""
    _archive(
        tmp_path,
        {
            "AAA/message_0001.xml": ("Budget", "Approved, see attached sheet."),
            "ZZZ/message_0001.xml": ("Budget", "Approved, see attached sheet."),
        },
    )
    docs = _load(tmp_path)
    assert len(docs) == 1
    assert docs[0].source_key.startswith("AAA/")


def test_messages_differing_only_in_sender_are_not_merged(tmp_path):
    """The header rides in the body, so the fingerprint covers it. Two
    one-line replies saying "Sounds good" from different people are
    different messages."""
    olm = tmp_path / "mail.olm"
    tmpl = """<?xml version="1.0"?>
<emails><email>
  <OPFMessageCopySubject>Re: plan</OPFMessageCopySubject>
  <OPFMessageCopyBody>Sounds good.</OPFMessageCopyBody>
  <OPFMessageCopyFromAddresses><OPFContactEmailAddress
    OPFContactEmailAddressAddress="{who}"/></OPFMessageCopyFromAddresses>
</email></emails>
"""
    with zipfile.ZipFile(olm, "w") as z:
        z.writestr("Local/com.microsoft.__Messages/Inbox/message_0001.xml", tmpl.format(who="jamie@example.com"))
        z.writestr("Local/com.microsoft.__Messages/Inbox/message_0002.xml", tmpl.format(who="alex@example.com"))
    assert len(_load(tmp_path)) == 2


def test_the_same_template_sent_on_different_dates_is_not_collapsed(tmp_path):
    """The bug the first implementation shipped, caught on real data.

    `corpus.util.dedup.fingerprint` strips dates and URLs before hashing --
    right for a document re-exported with a new timestamp, wrong for mail.
    On a real archive it collapsed 53 salon booking confirmations (same
    template, same sender, bookings across 2014 and 2015, a different
    reservation URL each) into ONE, because the only things telling them
    apart were exactly the two fields it removes. 52 real appointments would
    have left the index.
    """
    olm = tmp_path / "mail.olm"
    tmpl = """<?xml version="1.0"?>
<emails><email>
  <OPFMessageCopySubject>Booking confirmed</OPFMessageCopySubject>
  <OPFMessageCopyBody>Your appointment is confirmed. See {url}</OPFMessageCopyBody>
  <OPFMessageCopySentTime>{sent}</OPFMessageCopySentTime>
</email></emails>
"""
    with zipfile.ZipFile(olm, "w") as z:
        for i, (sent, url) in enumerate(
            [
                ("2014-04-10 05:41:18 +0000", "https://salon.example/r/111"),
                ("2015-01-27 05:07:06 +0000", "https://salon.example/r/222"),
                ("2015-03-27 04:57:01 +0000", "https://salon.example/r/333"),
            ]
        ):
            z.writestr(
                f"Local/com.microsoft.__Messages/Inbox/message_{i:04d}.xml",
                tmpl.format(sent=sent, url=url),
            )
    assert len(_load(tmp_path)) == 3, "distinct bookings were merged"


def test_an_exact_duplicate_of_a_dated_message_is_still_collapsed(tmp_path):
    """The other side: the SAME message stored in two folders has the same
    date, and must still collapse."""
    olm = tmp_path / "mail.olm"
    body = """<?xml version="1.0"?>
<emails><email>
  <OPFMessageCopySubject>Booking confirmed</OPFMessageCopySubject>
  <OPFMessageCopyBody>Your appointment is confirmed.</OPFMessageCopyBody>
  <OPFMessageCopySentTime>2015-03-27 04:57:01 +0000</OPFMessageCopySentTime>
</email></emails>
"""
    with zipfile.ZipFile(olm, "w") as z:
        z.writestr("Local/com.microsoft.__Messages/Inbox/message_0001.xml", body)
        z.writestr("Local/com.microsoft.__Messages/Deleted/message_0999.xml", body)
    assert len(_load(tmp_path)) == 1
