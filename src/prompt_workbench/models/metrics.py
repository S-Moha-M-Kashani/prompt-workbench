"""What a response gets scored on.

A metric is a name, a rubric, a weight, and a way of producing a number between
zero and one. Two kinds exist and the split is deliberate: anything checkable by
code (does this parse as JSON? is it under the length limit?) should not cost a
judge call, because a deterministic check is free, instant, and gives the same
answer every time. Everything else needs a model reading the rubric.

Both kinds return the same result shape, so the grade arithmetic never has to
know which sort of metric produced a score.
"""

from dataclasses import dataclass, replace
from enum import StrEnum

# A rubric shorter than this cannot describe a scoring decision, and a judge
# given three words invents the rest of the criterion itself.
MIN_RUBRIC_CHARACTERS = 20


class MetricKind(StrEnum):
    """How a metric arrives at its number."""

    RUBRIC = "rubric"
    DETERMINISTIC = "deterministic"

    @property
    def label(self) -> str:
        return {"rubric": "Judge", "deterministic": "Deterministic"}[self.value]


@dataclass(frozen=True)
class MetricDefinition:
    """One scoring dimension, with the weight it carries in the grade."""

    id: str
    name: str
    rubric: str
    kind: MetricKind = MetricKind.RUBRIC
    weight: float = 1.0
    enabled: bool = True
    builtin: bool = False
    check: str = ""
    requires_reference: bool = False

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("A metric needs a name")
        if len(self.rubric.strip()) < MIN_RUBRIC_CHARACTERS:
            raise ValueError(
                f"Metric {self.name!r} needs a usable rubric of at least "
                f"{MIN_RUBRIC_CHARACTERS} characters — a judge given less than "
                "that invents the criterion itself"
            )
        if self.weight < 0:
            raise ValueError(f"Metric {self.name!r} has a negative weight")
        if self.kind is MetricKind.DETERMINISTIC and not self.check.strip():
            raise ValueError(
                f"Deterministic metric {self.name!r} must name the check it runs"
            )

    def with_enabled(self, enabled: bool) -> "MetricDefinition":
        return replace(self, enabled=enabled)

    def with_weight(self, weight: float) -> "MetricDefinition":
        return replace(self, weight=weight)
