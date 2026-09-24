"""CSV / TSV connector via the stdlib `csv` module — no third-party dependency.

Design decision: summarize, don't dump every row
--------------------------------------------------
A connector that emits one chunk per row (or even one chunk per few hundred
chars of raw rows) makes the index actively *worse* the moment a real export
shows up. A 50,000-row financial report or data export would produce
thousands of chunks that are almost indistinguishable from each other in
embedding space — "Acme,1200.50,2026-01-03", "Globex,980.00,2026-01-04",
... — and because retrieval ranks by similarity, not provenance, those
near-duplicate rows crowd out genuinely distinct prose (emails, notes, docs)
in every subsequent search across the WHOLE corpus, not just within this
file. That's the same failure mode the zip connector's dependency-noise
filter exists to prevent, just triggered by a spreadsheet instead of a
`node_modules` tree.

What actually makes a spreadsheet *findable*, per a query like "quarterly
revenue spreadsheet" or "that CSV with the customer list", is: its filename,
its column names (a schema tells you what the file is about far more
efficiently than any individual row does), how big it is, and a taste of the
actual values so you can distinguish "revenue.csv" from another CSV that also
happens to be called revenue.csv. None of that requires indexing every row.
Once a hit identifies the right file, the user opens it directly to work
with the data — this index's job is discovery, not a query interface over
the data itself (that's a job for a real database, e.g. `sqlite-vec`'s host
DB is emphatically not designed to be pivoted from). So the body built here
is: filename + row/column counts, inferred column names/types, and a bounded
sample — never the whole table, no matter how large the file is.

Two-tier threshold, not a single rule
--------------------------------------
A blanket "always summarize" rule throws away real value for the common case
of a genuinely SMALL CSV — a 20-row roster, a 40-item packing list, a short
config table. Those are exactly the kind of "find the value, not just the
file" documents a personal archive benefits from having fully searchable,
and summarizing them for no reason is a needless loss of fidelity purely to
guard against a problem (chunk flood) that a 20-row file cannot cause.

So: a file with at most `FULL_INDEX_MAX_ROWS` (100) data rows AND at most
`FULL_INDEX_MAX_CHARS` (50,000) characters of rendered row text gets its
**entire** table indexed verbatim (still chunked normally by the shared
markdown chunker afterward — a 100-row file might land as 2-4 chunks, not
one giant one, and that's fine). Both caps have to hold — row count alone
isn't a reliable proxy for chunk-flood risk, since 80 rows of one column
each holding a paragraph of embedded text is not "small" in any sense that
matters here. Above either cap, the file gets the schema + bounded-sample
treatment. This mirrors the brief's own suggested alternative, and is
adopted deliberately: it's a real fidelity win for the common small-file
case at essentially no added complexity, since both code paths already have
to render "a set of rows" — the only difference is which set.

Sample size and shape: fixed-size head + tail, not random, not scaled to
file size
----------------------------------------------------------------------------
Above the threshold, the sample is always `SAMPLE_HEAD_ROWS` (10) from the
start of the file plus `SAMPLE_TAIL_ROWS` (10) from the end — 20 rows total,
whether the file has 200 rows or 2,000,000. Three reasons, not one:

1. **Determinism matters more here than it would in an ordinary sampling
   problem.** This engine content-hashes each chunk and skips re-embedding
   unchanged content on every subsequent `corpus-ingest` run (see
   `util/dedup.py`, `Chunk.content_hash`). A random sample would make the
   rendered body — and therefore its hash — different on every single run
   even when the underlying file hasn't changed at all, forcing a wasted
   re-embed of every CSV/TSV source on every ingest. Head/tail sampling
   is a pure function of the file's own content, so an unchanged file
   produces an unchanged body and costs nothing on a re-run.
2. **Head and tail are usually the two most informative slices of a real
   export**, which tend to be append-ordered or chronological: the head
   shows the file's earliest/baseline rows (and, incidentally, gives a
   human skimming a search result the clearest first impression of the
   file's shape), the tail shows what's most recent — often exactly what
   someone searching for "latest export" or "most recent report" wants
   corroborated. A random sample optimizes for "statistically
   representative," which isn't the same thing as "helps you recognize
   this file."
3. **Fixed size keeps sampling cost, and index cost, constant regardless of
   file size.** A 300-row file and a 300,000-row file produce the same
   ~20-row sample; the summary approach's whole point is that indexing cost
   shouldn't scale with row count once a file is "large" by any definition.

Headerless files
-----------------
`csv.Sniffer().has_header()` is tried first — it's the standard-library tool
built for exactly this and gets it right on any file where the sample
contains at least two distinguishable rows and more than one column.
Sniffer raises `csv.Error` on the cases where it genuinely can't tell (most
often a single-column file, since there's no second column's type profile to
compare against, or a too-small sample) — see `_looks_like_header` for the
fallback. For two OR MORE columns, it counts how many cells in the first row
vs. the second "look numeric," on the reasoning that real column headers are
essentially never bare numeric literals: a first row that is ENTIRELY
numeric-looking is treated as data even in a tie, otherwise the more common
case (a real header) is assumed, since biasing toward "has a header" costs
little when wrong (worst case: the true header row becomes a
slightly-odd-looking first data row, still visible in the sample) while the
reverse mistake (a real data row promoted to a queried column name) is
worse. A headerless file's columns are synthesized as `col_1`, `col_2`, ...
from `max(len(row) for row in rows)`, so every value in the file lands under
some labeled column even if row widths vary. The single-column case gets a
different default — see below.

Single-column files ("a CSV that's really a list of values")
---------------------------------------------------------------
Structurally NOT special-cased — one inferred column, one type, up to 20
sampled values, same machinery as every other width. The one place it DOES
get special treatment is header detection, and for a real reason: a
single-column ambiguous sample is where "guess header" is least reliable,
because `csv.Sniffer` has no second column to compare a type profile
against, and the text-vs-numeric heuristic above can't distinguish a real
header ("fruit") from an ordinary data value ("apple") — both are equally
"non-numeric text." The two possible wrong guesses aren't equally costly,
though: wrongly assuming "no header" on a file that has one costs a single
oddly-placed value in the sample (the real header text is still visible,
just not labeled as a column name); wrongly assuming "header" on a genuine
headerless list — confirmed against a 3-item test fixture — discards one of
what may be very few real values outright, which is a proportionally much
bigger fidelity loss for a short list. So a single-column file with an
ambiguous sample defaults to NO header, the opposite default from the
multi-column case, specifically to avoid silently losing data.

Inconsistent row lengths (ragged files)
-----------------------------------------
Never fatal. Row lengths are collected into a set; if it has more than one
member (or doesn't match the declared header width), a one-line note is
added to the body ("row lengths vary...") so this is visible to a human
without breaking anything. Column-count and type inference use the header
width (or, headerless, the widest row seen) as the "declared" shape; a
shorter row simply contributes no value to its missing trailing columns, a
longer row's extra cells are still rendered in the sample/full-data section
(nothing is silently truncated) even though they fall outside the declared
schema.

Encodings
----------
UTF-8 first; on `UnicodeDecodeError`, latin-1 (which — mapping every byte
0-255 to a codepoint — cannot itself fail to decode, so this is a genuine
fallback, not just "try harder"). No `chardet`/`charset-normalizer`
dependency: UTF-8-with-latin-1-fallback covers the overwhelming majority of
real-world CSVs (ASCII, UTF-8, and the common legacy Windows-1252/Latin-1
exports) without adding a dependency for the long tail of exotic encodings.

Embedded newlines in quoted fields
------------------------------------
Handled correctly by the stdlib `csv` module itself as long as the decoded
text is fed to `csv.reader` as a whole (via `io.StringIO`, not split into
lines first) — verified directly (see the connector's tests). What this
connector adds on top: an embedded `\n`/`\r` inside a cell is flattened to a
single space when a row is RENDERED into the body's tab-joined line format,
so one logical row always corresponds to exactly one line of output text —
otherwise a multi-line cell would make the rendered sample look like extra,
malformed rows.

NUL bytes / binary content under a `.csv` extension
------------------------------------------------------
A file containing a NUL byte is not text no matter what its extension
claims (a misnamed binary file, most likely) — checked directly (`b"\x00"
in raw_bytes`) before any decode/parse is attempted. This is the CSV
analogue of pptx's legacy-`.ppt` detection: a permanent condition, not a
transient one — the file's bytes will not change between runs — so it's
counted in `skipped_files`, not `failed_files`. Every other read/parse
failure (an OS-level read error, or a `csv.Error` from a structurally
broken file — e.g. a quote left unterminated for the rest of the file) is
`failed_files`: possibly transient, and not something this connector can
prove is permanent the way an OLE2 signature or a NUL byte can be.

Empty files
------------
A file with no non-whitespace bytes at all is skipped like an empty
docx/xlsx — logged, not counted as a failure (there is nothing wrong with
having an empty file; there is just nothing to index). A file with a header
row but zero data rows is NOT treated as empty — it's a real, if sparse,
schema worth indexing (a template or an export that ran with no matching
rows) — it just gets a "(no data rows)" body instead of a sample section.

TSV, and any other delimiter
------------------------------
The delimiter is auto-detected per file via `csv.Sniffer().sniff()`
(falling back to `default_delimiter` when the sample is too small/ambiguous
for Sniffer to decide, e.g. a single-column file) rather than assumed from
the file extension — so this one connector class handles comma, tab, or any
other consistent single-character delimiter Sniffer can find, and registry.py
registers it under both `csv` (default glob `**/*.csv`, default delimiter
`,`) and `tsv` (default glob `**/*.tsv`, default delimiter `\t`) — the
"default" only matters as the last-resort fallback when even Sniffer can't
tell.

`pandas` was deliberately not added — the brief's explicit ask, and correct
independently of that: everything this connector does (parse rows, infer a
column's type from a sample of its values, sample head/tail) is a few dozen
lines of stdlib `csv`, and pandas would add a large dependency for zero
capability this connector actually needs (no numeric computation, no
joins/reshaping — just read-and-describe).
"""

from __future__ import annotations

import csv
import io
import logging
import os
import re
from collections.abc import Callable, Iterable
from pathlib import Path

from corpus.connectors.discovery import discover_files
from corpus.types import SourceDocument
from corpus.util.dedup import NearDuplicates

logger = logging.getLogger(__name__)

# See "Two-tier threshold" in the module docstring. Both must hold for the
# full-table path; either being exceeded falls back to schema + sample.
FULL_INDEX_MAX_ROWS = 100
FULL_INDEX_MAX_CHARS = 50_000

# See "Sample size and shape" in the module docstring.
SAMPLE_HEAD_ROWS = 10
SAMPLE_TAIL_ROWS = 10

# Column type inference reads at most this many of a column's leading data
# rows. Bounded so a huge file's type inference stays cheap; the first few
# hundred rows of a real export are overwhelmingly representative of the
# whole column's type (a column that starts as text and becomes numeric
# thousands of rows in is a data-quality problem in the source file, not
# something this findability index is trying to catch).
TYPE_INFERENCE_SAMPLE_ROWS = 500

_SNIFF_SAMPLE_CHARS = 4096
_SNIFF_DELIMITERS = ",\t;|"

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}([ T]\d{2}:\d{2}(:\d{2})?)?$")
_BOOL_VALUES = frozenset({"true", "false", "yes", "no", "y", "n", "t", "f"})

# Every legacy .ppt/.doc/.xls (and nothing valid-as-CSV) starts with this;
# irrelevant here — see pptx.py's use of the same idea for the OLE2 case.
# CSV's equivalent "this will never be text" signal is simpler: any NUL byte
# at all. See "NUL bytes" in the module docstring.


def _looks_numeric(cell: str) -> bool:
    cell = cell.strip()
    if not cell:
        return False
    try:
        float(cell.replace(",", ""))
        return True
    except ValueError:
        return False


def _looks_like_header(rows: list[list[str]]) -> bool:
    """Fallback used only when `csv.Sniffer().has_header()` itself raises
    `csv.Error` (typically a single-column file or too-small sample — see
    the module docstring's "Headerless files" section for the full
    reasoning).

    Single-column files get their own branch because they are exactly where
    this fallback is least reliable: "apple" (a header) and "apple" (a data
    value in a plain list of fruit) are textually indistinguishable, and
    `csv.Sniffer` can't compare a lone column's type profile against
    anything. The two wrong guesses aren't equally costly, so the tie is
    broken deliberately rather than arbitrarily: guessing "no header" on a
    file that actually has one costs a single oddly-placed value in the
    sample (the real header text is still visible, just not labeled as a
    column name); guessing "header" on a genuine headerless list — as
    `csv.Sniffer` itself would for a numeric list, and as a naive default
    would for a text list — discards one of what may be very few real
    values outright (see this file's tests: a 3-item fruit list loses
    "apple" entirely under a "default to header" rule). Bias toward keeping
    the data.
    """
    if len(rows) < 2:
        return True
    first, second = rows[0], rows[1]
    if len(first) == 1:
        return False
    first_numeric = sum(_looks_numeric(c) for c in first)
    if first_numeric == len(first):
        # Every cell in the first row looks like a bare number. Real column
        # headers are essentially never all-numeric literals, so this reads
        # as data — even if the second row is equally numeric (a tie would
        # otherwise default to "header," which is wrong for e.g. a plain
        # headerless list of numbers).
        return False
    second_numeric = sum(_looks_numeric(c) for c in second)
    return first_numeric <= second_numeric


def _detect_delimiter(sample: str, default_delimiter: str) -> str:
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=_SNIFF_DELIMITERS)
        delimiter = dialect.delimiter
        return delimiter if isinstance(delimiter, str) and delimiter else default_delimiter
    except csv.Error:
        return default_delimiter


def _detect_header(rows: list[list[str]], sample: str) -> bool:
    try:
        return csv.Sniffer().has_header(sample)
    except csv.Error:
        return _looks_like_header(rows)


def _infer_column_type(values: list[str]) -> str:
    values = [v for v in values if v.strip()]
    if not values:
        return "empty"
    n = len(values)

    def frac(pred: Callable[[str], bool]) -> float:
        return sum(1 for v in values if pred(v)) / n

    if frac(lambda v: _DATE_RE.match(v.strip()) is not None) >= 0.9:
        return "date"
    if frac(lambda v: v.strip().lower() in _BOOL_VALUES) >= 0.9:
        return "boolean"

    def is_int(v: str) -> bool:
        try:
            int(v.strip().replace(",", ""))
            return True
        except ValueError:
            return False

    if frac(is_int) >= 0.9:
        return "integer"

    def is_float(v: str) -> bool:
        try:
            float(v.strip().replace(",", ""))
            return True
        except ValueError:
            return False

    if frac(is_float) >= 0.9:
        return "number"
    return "text"


def _render_row(row: list[str]) -> str:
    # Flatten embedded newlines so one logical row is always one line of
    # rendered text — see "Embedded newlines" in the module docstring.
    return "\t".join(cell.replace("\r\n", " ").replace("\n", " ").replace("\r", " ").strip() for cell in row)


def _render_body(
    header: list[str],
    types: list[str],
    data_rows: list[list[str]],
    irregular: bool,
) -> str:
    row_count = len(data_rows)
    col_count = len(header)

    lines: list[str] = [f"**Rows:** {row_count}  **Columns:** {col_count}"]
    if irregular:
        lines.append(
            "**Note:** row lengths vary across this file; a short row is "
            "simply missing its trailing columns, a long row's extra cells "
            "still appear in the data below but fall outside the columns "
            "listed here."
        )
    lines.append("")
    lines.append("## Columns")
    for name, type_ in zip(header, types, strict=True):
        lines.append(f"- `{name}` — {type_}")

    if row_count == 0:
        lines.append("")
        lines.append("## Data")
        lines.append("(no data rows)")
        return "\n".join(lines).strip()

    header_line = "\t".join(header)
    use_full = row_count <= FULL_INDEX_MAX_ROWS
    rendered_full: list[str] = []
    if use_full:
        rendered_full = [_render_row(r) for r in data_rows]
        if sum(len(r) for r in rendered_full) > FULL_INDEX_MAX_CHARS:
            use_full = False

    lines.append("")
    if use_full:
        lines.append("## Data")
        lines.append(header_line)
        lines.extend(rendered_full)
    else:
        head = data_rows[:SAMPLE_HEAD_ROWS]
        tail = (
            data_rows[-SAMPLE_TAIL_ROWS:]
            if row_count > SAMPLE_HEAD_ROWS + SAMPLE_TAIL_ROWS
            else []
        )
        label = f"first {len(head)}"
        if tail:
            label += f", last {len(tail)}"
        lines.append(f"## Sample rows ({label} of {row_count})")
        lines.append(header_line)
        lines.extend(_render_row(r) for r in head)
        if tail:
            lines.append("...")
            lines.extend(_render_row(r) for r in tail)

    return "\n".join(lines).strip()


class CsvConnector:
    """Walks `path` for matching files (`.csv` by default; also registered as
    `tsv` with a `.tsv` glob and a tab default — see registry.py). See the
    module docstring for the full indexing design: schema + bounded sample
    above a size threshold, full content below it."""

    def __init__(
        self,
        source_type: str,
        path: Path | str,
        glob: str = "**/*.csv",
        default_delimiter: str = ",",
    ):
        self.source_type = source_type
        self._root = Path(os.path.expanduser(str(path))).resolve()
        self._glob = glob
        self._default_delimiter = default_delimiter
        self.failed_files = 0
        self.skipped_files = 0

    def load(self) -> Iterable[SourceDocument]:
        # Reset per run — see the identical note in every other connector in
        # this package.
        self.failed_files = 0
        self.skipped_files = 0

        if not self._root.is_dir():
            raise FileNotFoundError(
                f"Csv source '{self.source_type}': directory not found: {self._root}"
            )

        dupes = NearDuplicates(self.source_type)
        for path in discover_files(self._root, self._glob):
            doc = self._load_one(path)
            if doc is None:
                continue

            if dupes.seen_before(doc.raw["body"], doc.source_key):
                continue
            yield doc

    def _load_one(self, path: Path) -> SourceDocument | None:
        try:
            raw_bytes = path.read_bytes()
        except OSError as e:
            logger.warning("Csv source '%s': cannot read %s: %s", self.source_type, path, e)
            self.failed_files += 1
            return None

        if not raw_bytes.strip():
            logger.info("%s: skipping '%s' — empty file", self.source_type, path.name)
            return None

        if b"\x00" in raw_bytes:
            # Binary content under a .csv/.tsv extension. Permanent — the
            # bytes will not change between runs — see "NUL bytes" in the
            # module docstring.
            logger.warning(
                "%s: skipping '%s' — contains NUL bytes, not text", self.source_type, path.name
            )
            self.skipped_files += 1
            return None

        try:
            text = raw_bytes.decode("utf-8")
        except UnicodeDecodeError:
            text = raw_bytes.decode("latin-1")  # cannot itself fail

        sample = text[:_SNIFF_SAMPLE_CHARS]
        delimiter = _detect_delimiter(sample, self._default_delimiter)

        try:
            all_rows = [row for row in csv.reader(io.StringIO(text), delimiter=delimiter) if row]
        except csv.Error as e:
            logger.warning("%s: cannot parse '%s': %s", self.source_type, path.name, e)
            self.failed_files += 1
            return None

        if not all_rows:
            logger.info("%s: skipping '%s' — no rows found", self.source_type, path.name)
            return None

        has_header = _detect_header(all_rows, sample)
        if has_header:
            header = [h.strip() or f"col_{i + 1}" for i, h in enumerate(all_rows[0])]
            data_rows = all_rows[1:]
        else:
            width = max(len(r) for r in all_rows)
            header = [f"col_{i + 1}" for i in range(width)]
            data_rows = all_rows

        lengths = {len(r) for r in data_rows}
        irregular = len(lengths) > 1 or (bool(data_rows) and len(header) not in lengths)

        sample_for_types = data_rows[:TYPE_INFERENCE_SAMPLE_ROWS]
        types = [
            _infer_column_type([r[i] for r in sample_for_types if len(r) > i])
            for i in range(len(header))
        ]

        body = _render_body(header, types, data_rows, irregular)

        return SourceDocument(
            source_type=self.source_type,
            source_key=str(path.relative_to(self._root)),
            title=path.stem,
            url=None,
            raw={"body": body, "path": str(path)},
        )
