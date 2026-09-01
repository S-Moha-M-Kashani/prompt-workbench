"""Where artifact identifiers come from.

Injectable rather than a bare ``uuid4()`` call at every call site, so a test can
hand in a counter and assert on an exact snapshot. Real runs get random ids;
tests get ``brief-1``, ``brief-2`` and readable failures.
"""

from collections.abc import Callable
from itertools import count
from uuid import uuid4

IdFactory = Callable[[str], str]

# Long enough that a collision within a session is not worth thinking about,
# short enough to read in a UI label.
_ID_LENGTH = 12


def random_id(prefix: str) -> str:
    """``brief-3f8a1c2d9e04`` — the default factory."""
    return f"{prefix}-{uuid4().hex[:_ID_LENGTH]}"


def sequential_ids() -> IdFactory:
    """A factory numbering each prefix from one; for tests and fixtures."""
    counters: dict[str, count[int]] = {}

    def make(prefix: str) -> str:
        counter = counters.setdefault(prefix, count(1))
        return f"{prefix}-{next(counter)}"

    return make
