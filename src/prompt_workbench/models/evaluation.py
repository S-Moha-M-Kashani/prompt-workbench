"""Evaluation results, and the one rule that keeps a grade honest.

A metric score has three distinct outcomes, and collapsing any two of them
corrupts the arithmetic:

- **scored** — a number between zero and one;
- **failed** — the judge errored, timed out, or answered with nothing;
- **not applicable** — the metric genuinely does not apply, as reference
  similarity does not to a case with no reference answer.

A failure read as "no opinion" and softened to 0.5 would move a grade with
nothing on screen to say so. So ``score`` is ``None`` for both non-scored cases,
each carries its own explanation, and the grade arithmetic drops them from the
weighted mean instead of filling them in.
"""

from dataclasses import dataclass
from datetime import datetime

from prompt_workbench.models.ground_truth import GroundTruthCase
from prompt_workbench.models.metrics import MetricDefinition
from prompt_workbench.models.provenance import SourceRef

# Coarse bands, wide on purpose: a judge model's scores are not precise enough
# to justify finer distinctions, and a "B+" would imply they were.
LETTER_BANDS: tuple[tuple[float, str], ...] = (
    (0.85, "A"),
    (0.75, "B"),
    (0.65, "C"),
    (0.55, "D"),
)
NO_GRADE = "-"


@dataclass(frozen=True)
class EvaluationEvidence:
    """Everything a metric is allowed to look at when scoring one response.

    Assembled once per response and passed to every metric, so two metrics
    scoring the same reply are demonstrably reading the same thing. It is a
    value, not a query object: nothing here can reach back into the workspace
    and pick up an artifact that has changed since the run began.
    """

    brief_context: str
    output_format: str
    case: GroundTruthCase
    candidate_prompt: str
    user_message: str
    response: str


@dataclass(frozen=True)
class MetricScore:
    """One metric's verdict on one response."""

    metric_id: str
    metric_name: str
    weight: float
    score: float | None = None
    reason: str = ""
    failed: bool = False
    failure: str = ""
    applicable: bool = True

    def __post_init__(self) -> None:
        if self.score is not None and not 0.0 <= self.score <= 1.0:
            raise ValueError(
                f"Metric {self.metric_name!r} scored {self.score}, but scores must "
                "be normalized to between 0 and 1"
            )

    @classmethod
    def failed_with(
        cls, *, metric_id: str, metric_name: str, weight: float, failure: str
    ) -> "MetricScore":
        return cls(
            metric_id=metric_id,
            metric_name=metric_name,
            weight=weight,
            score=None,
            failed=True,
            failure=failure,
        )

    @classmethod
    def not_applicable(
        cls, *, metric_id: str, metric_name: str, weight: float, reason: str
    ) -> "MetricScore":
        return cls(
            metric_id=metric_id,
            metric_name=metric_name,
            weight=weight,
            score=None,
            reason=reason,
            applicable=False,
        )

    @property
    def counts_towards_grade(self) -> bool:
        return self.score is not None and self.weight > 0


@dataclass(frozen=True)
class Grade:
    """A normalized grade, or the explicit absence of one."""

    value: float | None

    @property
    def percentage(self) -> float | None:
        return None if self.value is None else round(self.value * 100, 1)

    @property
    def letter(self) -> str:
        if self.value is None:
            return NO_GRADE
        for threshold, letter in LETTER_BANDS:
            if self.value >= threshold:
                return letter
        return "F"

    def __str__(self) -> str:
        return NO_GRADE if self.percentage is None else f"{self.percentage}% ({self.letter})"


@dataclass(frozen=True)
class CaseEvaluation:
    """Every metric's verdict on one response, and the weighted result."""

    execution_id: str
    case_id: str
    scores: tuple[MetricScore, ...]
    grade: float | None

    @property
    def failures(self) -> tuple[MetricScore, ...]:
        return tuple(score for score in self.scores if score.failed)


@dataclass(frozen=True)
class EvaluationRun:
    """One completed evaluation. Never edited after it is recorded."""

    id: str
    created_at: datetime
    candidate: SourceRef
    source_brief: SourceRef
    source_dataset: SourceRef
    metrics: tuple[MetricDefinition, ...]
    judge_backend: str
    judge_model: str
    cases: tuple[CaseEvaluation, ...]
    overall: Grade

    @property
    def failures(self) -> tuple[MetricScore, ...]:
        return tuple(score for case in self.cases for score in case.failures)

    @property
    def has_failures(self) -> bool:
        return bool(self.failures)
