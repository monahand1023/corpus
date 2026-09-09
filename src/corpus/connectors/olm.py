"""Outlook for Mac archive (`.olm`) connector — one document per message.

An `.olm` is a plain zip whose messages live at
`Local/com.microsoft.__Messages/<store>/<folder…>/message_NNNNN.xml`, each
holding one `<email>` element with Outlook's `OPFMessageCopy*` fields. This
connector reads those members directly rather than extracting the archive: a
real one is very large, with millions of members, so the extract-then-
delegate approach `zip.py` uses (fine for a few hundred office documents)
would need that much scratch space to read a mailbox.

Three properties of the format shape everything below. All three were
measured on a real archive; each would silently cost you money or content if
ignored.

**The archive mirrors itself.** Every store also appears under
`Outlook for Mac Archive OLM/On My Computer/<store>/…`. Most messagefolders exist in both trees — roughly as many mirrored messages asdirect ones. Indexing both doubles the embedding bill for nothing, so
`skip_mirror_tree` (default true) reads only the direct stores.

The mirror is NOT an exact copy, which is why the direct tree is the one
kept: in a few folders the direct store holds a few messages the mirror lacks
(`Sent Items 2011`: a handful more than the mirror), and no folder at all matches on
total bytes. Dropping the direct tree instead would lose real messages.

**Message bodies are usually escaped HTML, not text.** `OPFMessageCopyBody`
commonly contains an entire Word-generated HTML document, XML-escaped —
`&lt;html&gt;…` with its stylesheet and font tables inline. Measured across
a sample of messages, markup was 84% of the body. Estimating cost or chunking
without unescaping and stripping it first treats CSS as prose.

**Replies quote their whole history.** After markup is removed, a further
78% of the remaining text is the quoted thread — the same conversation
re-embedded once per reply. `trim_quoted_history` (default true) keeps only
what each message actually added. On the same sample this took the archive
from ~1,073M tokens to ~233M.

Trimming is skipped when it would empty the message: a bare forward with no
added comment IS its quoted content, and 6.6% of messages are that. Those
keep their full text rather than being dropped.
"""

from __future__ import annotations

import html
import logging
import re
import zipfile
from collections.abc import Iterator
from fnmatch import fnmatch
from pathlib import Path
from typing import Any

from corpus.types import SourceDocument

logger = logging.getLogger(__name__)

MESSAGE_ROOT = "Local/com.microsoft.__Messages/"
MIRROR_PREFIX = "Outlook for Mac Archive OLM/On My Computer/"

# Outlook's own field names. Only the ones that earn their place in a chunk
# are read; the archive carries ~30 more per message (read state, priority,
# inference classification) that add nothing to retrieval.
_FIELDS = {
    "subject": "OPFMessageCopySubject",
    "body": "OPFMessageCopyBody",
    "sent": "OPFMessageCopySentTime",
    "received": "OPFMessageCopyReceivedTime",
    "message_id": "OPFMessageCopyMessageID",
    "thread_topic": "OPFMessageCopyThreadTopic",
}
_FIELD_RE = {
    key: re.compile(rf"<{tag}[^>]*>(.*?)</{tag}>".encode(), re.S)
    for key, tag in _FIELDS.items()
}
# Sender/recipient blocks hold <emailAddress OPFContactEmailAddressAddress="…"
# OPFContactEmailAddressName="…"/> children.
_FROM_BLOCK = re.compile(rb"<OPFMessageCopyFromAddresses.*?</OPFMessageCopyFromAddresses>", re.S)
_TO_BLOCK = re.compile(rb"<OPFMessageCopyToAddresses.*?</OPFMessageCopyToAddresses>", re.S)
_ADDRESS = re.compile(rb'OPFContactEmailAddressAddress="([^"]*)"')
_NAME = re.compile(rb'OPFContactEmailAddressName="([^"]*)"')

_SCRIPT_STYLE = re.compile(r"<(script|style|head)\b.*?</\1>", re.S | re.I)
_TAG = re.compile(r"<[^>]+>")
_INLINE_WS = re.compile(r"[ \t\xa0]+")
_BLANK_LINES = re.compile(r"\n\s*\n+")

# Where a reply's quoted history begins. Outlook writes these headers into
# the body verbatim in the composing client's locale, so the Japanese forms
# matter for any archive with Japanese correspondence — this one is roughly
# half Japanese and matching only the English forms would leave that half of
# the archive fully quoted.
_QUOTE_MARKER = re.compile(
    r"^[ \t]*(?:"
    r"-{2,}\s*Original Message\s*-{2,}"
    r"|-{2,}\s*Forwarded Message\s*-{2,}"
    r"|_{10,}"
    r"|From:[ \t]\S"
    r"|Sent:[ \t]\S"
    r"|差出人:[ \t]*\S"
    r"|送信者:[ \t]*\S"
    r"|On .{4,120}? wrote:"
    r")",
    re.M | re.I,
)


def _text(raw: bytes) -> str:
    """Decode one field's bytes to plain text.

    Unescapes twice on purpose: the field arrives XML-escaped, and what that
    reveals is usually HTML whose own attributes and text carry a second
    layer of entities (`&amp;nbsp;` in the source becomes `&nbsp;` after the
    first pass). Tags become newlines rather than empty strings so that
    `</p><p>` doesn't weld two paragraphs into one word.
    """
    text = html.unescape(raw.decode("utf-8", "replace"))
    text = _SCRIPT_STYLE.sub(" ", text)
    text = _TAG.sub("\n", text)
    text = html.unescape(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _INLINE_WS.sub(" ", text)
    return _BLANK_LINES.sub("\n\n", text).strip()


def trim_quoted_history(text: str) -> str:
    """Return only what this message added, or `text` unchanged if that would
    be nothing.

    The fallback is the whole point: a forward with no added comment has no
    "new" content at all, and its quoted body is the entire message. Dropping
    those would silently lose 6.6% of a real archive.
    """
    match = _QUOTE_MARKER.search(text)
    if match is None:
        return text
    trimmed = text[: match.start()].strip()
    return trimmed or text


def _addresses(block_re: re.Pattern[bytes], raw: bytes, limit: int = 12) -> list[str]:
    """Names (falling back to bare addresses) from one participant block.

    Capped because a mailing-list blast can carry hundreds of recipients,
    which would swamp the message's actual text in its own chunk.
    """
    block = block_re.search(raw)
    if block is None:
        return []
    payload = block.group(0)
    names = _NAME.findall(payload)
    addrs = _ADDRESS.findall(payload)
    out: list[str] = []
    for i, addr in enumerate(addrs[:limit]):
        name = names[i].decode("utf-8", "replace").strip() if i < len(names) else ""
        address = addr.decode("utf-8", "replace").strip()
        out.append(f"{name} <{address}>" if name and name != address else address)
    if len(addrs) > limit:
        out.append(f"(+{len(addrs) - limit} more)")
    return out


class OlmConnector:
    """Yields one `SourceDocument` per message in one or more `.olm` files."""

    def __init__(
        self,
        source_type: str,
        path: Path | str,
        glob: str = "**/*.olm",
        folders: list[str] | None = None,
        skip_mirror_tree: bool = True,
        trim_quotes: bool = True,
    ):
        self.source_type = source_type
        self._root = Path(path).expanduser().resolve()
        self._glob = glob
        # fnmatch patterns against the store-relative folder path, e.g.
        # "archive PST/Inbox*". None means every folder.
        self._folders = folders
        self._skip_mirror = skip_mirror_tree
        self._trim = trim_quotes
        self.failed_files = 0
        self.skipped_files = 0

    def _wanted(self, folder: str) -> bool:
        if self._folders is None:
            return True
        return any(fnmatch(folder, pattern) for pattern in self._folders)

    def _members(self, archive: zipfile.ZipFile) -> list[str]:
        """Message members to read, mirror tree and unwanted folders removed.

        Built as a list rather than streamed because the caller needs the
        count up front to log progress over an archive this size, and because
        `namelist()` has already materialized every name anyway.
        """
        members: list[str] = []
        for name in archive.namelist():
            if not name.startswith(MESSAGE_ROOT) or not name.endswith(".xml"):
                continue
            if "/message_" not in name:
                continue
            relative = name[len(MESSAGE_ROOT) :]
            if self._skip_mirror and relative.startswith(MIRROR_PREFIX):
                self.skipped_files += 1
                continue
            folder = relative.rsplit("/", 1)[0]
            if not self._wanted(folder):
                self.skipped_files += 1
                continue
            members.append(name)
        return members

    def _document(self, archive: zipfile.ZipFile, member: str) -> SourceDocument | None:
        raw = archive.read(member)
        fields: dict[str, str] = {}
        for key, pattern in _FIELD_RE.items():
            match = pattern.search(raw)
            if match is not None:
                fields[key] = _text(match.group(1)) if key == "body" else html.unescape(
                    match.group(1).decode("utf-8", "replace")
                ).strip()

        body = fields.get("body", "")
        if self._trim:
            body = trim_quoted_history(body)
        subject = fields.get("subject", "")
        if not body.strip() and not subject.strip():
            # No subject and no body: a calendar stub or an empty draft.
            # Permanent, not transient — skipped_files, so it never blocks
            # orphan pruning (see the ingester's failed/skipped contract).
            self.skipped_files += 1
            return None

        sender = _addresses(_FROM_BLOCK, raw)
        recipients = _addresses(_TO_BLOCK, raw)
        sent = fields.get("sent") or fields.get("received") or None

        header = []
        if sender:
            header.append(f"From: {', '.join(sender)}")
        if recipients:
            header.append(f"To: {', '.join(recipients)}")
        if sent:
            header.append(f"Date: {sent}")
        # The header rides in the body rather than living only in metadata so
        # that a two-line reply still retrieves on who sent it and when — the
        # median message is 224 characters once quotes are trimmed, and
        # without this most of them are unsearchable by participant.
        text = "\n".join(header) + ("\n\n" if header else "") + body

        relative = member[len(MESSAGE_ROOT) :]
        folder = relative.rsplit("/", 1)[0]
        # Keyed by archive member path, not by Message-ID: the path is unique
        # and always present, while OPFMessageCopyMessageID is absent from
        # ~18% of messages and duplicated across a mirrored copy.
        return SourceDocument(
            source_type=self.source_type,
            source_key=relative,
            title=subject or "(no subject)",
            url=None,
            created_at=sent,
            updated_at=sent,
            raw={
                "body": text,
                "folder": folder,
                "message_id": fields.get("message_id"),
                "thread_topic": fields.get("thread_topic"),
            },
        )

    def load(self) -> Iterator[SourceDocument]:
        if not self._root.is_dir():
            raise FileNotFoundError(
                f"Olm source '{self.source_type}': directory not found: {self._root}"
            )
        self.failed_files = 0
        self.skipped_files = 0

        archives = sorted(self._root.glob(self._glob))
        if not archives:
            logger.warning("%s: no .olm archives under %s", self.source_type, self._root)

        for archive_path in archives:
            try:
                archive = zipfile.ZipFile(archive_path)
            except (OSError, zipfile.BadZipFile) as e:
                # ENUMERATION COMPLETENESS: an archive we cannot open at all
                # would silently contribute zero documents, and orphan pruning
                # would then delete every chunk it previously supplied. Raise
                # so the ingester leaves the index alone.
                raise OSError(f"cannot read {archive_path}: {e}") from e

            with archive:
                members = self._members(archive)
                logger.info(
                    "%s: %s — %d messages to read (%d members skipped)",
                    self.source_type,
                    archive_path.name,
                    len(members),
                    self.skipped_files,
                )
                for index, member in enumerate(members, 1):
                    try:
                        document = self._document(archive, member)
                    except (OSError, zipfile.BadZipFile, KeyError) as e:
                        # One unreadable member is transient-shaped (a damaged
                        # region of an otherwise-fine archive), so it counts as
                        # failed_files and suppresses pruning rather than
                        # letting this run delete the messages it couldn't read.
                        logger.debug("%s: cannot read member %s: %s", self.source_type, member, e)
                        self.failed_files += 1
                        continue
                    if document is not None:
                        yield document
                    if index % 20_000 == 0:
                        logger.info(
                            "%s: %d/%d messages", self.source_type, index, len(members)
                        )


def build(cfg: Any) -> OlmConnector:
    """Factory used by `connectors/registry.py`."""
    return OlmConnector(
        source_type=cfg.name,
        path=cfg.path,
        glob=cfg.glob or "**/*.olm",
        folders=cfg.olm_folders,
        skip_mirror_tree=cfg.olm_skip_mirror_tree,
        trim_quotes=cfg.olm_trim_quotes,
    )
