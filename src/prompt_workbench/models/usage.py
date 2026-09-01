"""What a call cost, in the two directions that behave differently.

Kept apart rather than summed because they are not interchangeable. Input
tokens grow with the prompt and the mocked context and are usually the larger
number by far; output tokens grow with the answer and are usually the more
expensive per token. A single total hides which one an edit just moved — and
"my prompt got longer" is exactly the thing a workbench should make visible.

Zero means the provider reported nothing, not that nothing was spent.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class TokenUsage:
    """Tokens in and out for one call."""

    tokens_in: int = 0
    tokens_out: int = 0

    @property
    def total(self) -> int:
        return self.tokens_in + self.tokens_out

    @property
    def is_reported(self) -> bool:
        """Whether the provider actually told us anything."""
        return self.total > 0

    def __str__(self) -> str:
        return f"{self.tokens_in:,} in · {self.tokens_out:,} out"
