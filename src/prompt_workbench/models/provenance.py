"""A pointer to the exact upstream snapshot an artifact came from.

Every generated artifact answers "which inputs made you?" with one of these.
Identity alone is not enough: a brief that has been edited keeps its id but
gains a revision, so a candidate that stores only ``brief-1`` cannot say whether
it predates that edit. Storing the pair makes staleness a comparison rather than
a guess.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class SourceRef:
    """One upstream artifact, pinned to the revision that was read."""

    id: str
    revision: int

    def __str__(self) -> str:
        return f"{self.id}@r{self.revision}"

    def is_stale_against(self, current: "SourceRef") -> bool:
        """Whether ``current`` has moved on from what this reference recorded."""
        return self.id == current.id and self.revision < current.revision
