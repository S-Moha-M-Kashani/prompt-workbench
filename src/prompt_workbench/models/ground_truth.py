"""The hybrid ground-truth dataset: what a good response has to do.

"Hybrid" is the load-bearing word. A reference answer alone only works for tasks
with one right shape; a rubric alone gives a judge nothing to anchor on. So a
case always carries *required criteria* and *forbidden behaviours* — checkable
claims about any acceptable answer — and may additionally carry a reference
answer when one genuinely exists.

Cases are grouped into a numbered dataset that points back at the brief snapshot
it was generated from.
"""

from collections.abc import Iterator
from dataclasses import dataclass, field, replace
from datetime import datetime
from enum import StrEnum

from prompt_workbench.models.provenance import SourceRef


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


@dataclass(frozen=True)
class GroundTruthDataset:
    """A numbered set of cases, pinned to the brief snapshot that produced it."""

    id: str
    revision: int
    source_brief: SourceRef
    cases: tuple[GroundTruthCase, ...] = ()
    created_at: datetime = field(default_factory=lambda: datetime.min)

    def __post_init__(self) -> None:
        ids = [case.id for case in self.cases]
        if len(ids) != len(set(ids)):
            raise ValueError("Dataset contains duplicate case ids")

    def __len__(self) -> int:
        return len(self.cases)

    def __iter__(self) -> Iterator[GroundTruthCase]:
        return iter(self.cases)

    @property
    def ref(self) -> SourceRef:
        return SourceRef(id=self.id, revision=self.revision)

    def case(self, case_id: str) -> GroundTruthCase:
        for case in self.cases:
            if case.id == case_id:
                return case
        raise KeyError(f"No test case {case_id!r} in this dataset")

    def with_cases(self, cases: tuple[GroundTruthCase, ...]) -> "GroundTruthDataset":
        """A new revision holding ``cases``; the old revision stays intact."""
        return replace(self, cases=cases, revision=self.revision + 1)
