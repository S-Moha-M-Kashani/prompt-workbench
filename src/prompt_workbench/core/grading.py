"""Turning metric scores into a grade, visibly enough to check by hand.

Two levels of arithmetic, both deliberately simple:

1. A **case grade** is the weighted mean of the metric scores for one response.
2. The **overall grade** is the plain arithmetic mean of the case grades — not a
   second weighted mean, because cases are not more or less important than each
   other; if one matters more, it should appear more than once in the dataset.

The rule that does the real work is what gets left out. A metric that failed or
did not apply carries no number, so it is dropped from the mean entirely rather
than counted as zero (which would punish a candidate for a judge timing out) or
as a neutral half (which would move the grade with nothing on screen to explain
why). Zero-weight metrics drop out for the same reason: they were switched off.
"""

from collections.abc import Sequence

from prompt_workbench.models.evaluation import Grade, MetricScore


def grade_case(scores: Sequence[MetricScore]) -> float | None:
    """The weighted mean of the usable scores, or ``None`` if there are none."""
    usable = [score for score in scores if score.counts_towards_grade]
    total_weight = sum(score.weight for score in usable)
    if total_weight <= 0:
        return None
    weighted = sum(score.weight * (score.score or 0.0) for score in usable)
    return weighted / total_weight


def grade_run(case_grades: Sequence[float | None]) -> Grade:
    """The arithmetic mean of the graded cases, ignoring the ungradeable ones."""
    graded = [grade for grade in case_grades if grade is not None]
    if not graded:
        return Grade(value=None)
    return Grade(value=sum(graded) / len(graded))
