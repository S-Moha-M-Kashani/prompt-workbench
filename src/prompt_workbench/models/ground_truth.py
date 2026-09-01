"""What a good response has to do, in a shape a judge can check.

A reference answer alone only works for tasks with one right shape; a rubric
alone gives a judge nothing to anchor on. So a case always carries *required
criteria* and *forbidden behaviours* — checkable claims about any acceptable
answer — and may additionally carry a reference answer when one genuinely
exists.

Cases are no longer authored: each ready use case builds one from the criteria
that made it worth trying in the first place.
"""

from dataclasses import dataclass
from enum import StrEnum


class CaseCategory(StrEnum):
    """What a case is probing. Generation aims for a spread across all three."""

    NORMAL = "normal"
    EDGE = "edge"
    FAILURE = "failure"

    @property
    def label(self) -> str:
        return {"normal": "Normal", "edge": "Edge", "failure": "Failure"}[self.value]


@dataclass(frozen=True)
class GroundTruthCase:
    """One test message plus what any acceptable reply to it must and must not do."""

    id: str
    test_message: str
    required_criteria: tuple[str, ...] = ()
    forbidden_behaviours: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    reference_answer: str | None = None
    category: CaseCategory = CaseCategory.NORMAL

    def __post_init__(self) -> None:
        if not self.test_message.strip():
            raise ValueError("A test case needs a test message")
        if not any(c.strip() for c in self.required_criteria):
            raise ValueError(
                "A test case needs at least one required criterion — without one "
                "there is nothing for a judge to check the response against"
            )

    @property
    def has_reference(self) -> bool:
        return bool(self.reference_answer and self.reference_answer.strip())

    def as_expectations(self) -> str:
        """The case's demands as text a judge can read."""
        lines = ["Required criteria:"]
        lines += [f"- {c}" for c in self.required_criteria]
        if self.forbidden_behaviours:
            lines.append("Forbidden behaviours:")
            lines += [f"- {b}" for b in self.forbidden_behaviours]
        if self.has_reference:
            lines.append(f"Reference answer:\n{self.reference_answer}")
        return "\n".join(lines)
