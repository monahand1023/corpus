"""`corpus-survey`: read-only reconnaissance for deciding what to index.

Answers the mechanical questions that otherwise get re-derived by hand with
throwaway shell pipelines every time a new directory is under consideration:
what's in a tree and can corpus index it (`census`), what's really inside an
archive vs. dependency noise (`archives`), how many hours of audio/video is
here (`media`), and how much of a directory is already indexed somewhere
else (`overlap`).

Hard rule, unlike every other package in this codebase: **nothing here ever
writes.** No database writes, no extraction to a non-temporary location, no
mutation of the tree being surveyed. See each submodule's docstring for how
it enforces that for its specific operation.
"""

from __future__ import annotations
