"""Weighted means, letter bands, and what a missing score must not do."""

import pytest

from prompt_workbench.core.grading import grade_case, grade_run
from prompt_workbench.models import Grade, MetricScore


def score(name: str, value: float | None, weight: float = 1.0) -> MetricScore:
    return MetricScore(metric_id=name, metric_name=name, weight=weight, score=value)


def test_a_case_grade_is_the_weighted_mean_of_its_scores() -> None:
    scores = (score("a", 1.0, weight=3.0), score("b", 0.0, weight=1.0))
    assert grade_case(scores) == 0.75


def test_equal_weights_give_a_plain_average() -> None:
    assert grade_case((score("a", 1.0), score("b", 0.5))) == 0.75


def test_a_failed_metric_is_excluded_rather_than_scored_as_neutral() -> None:
    failed = MetricScore.failed_with(
        metric_id="b", metric_name="b", weight=1.0, failure="judge timed out"
    )
    # A neutral 0.5 would give 0.75; excluding the failure gives 1.0.
    assert grade_case((score("a", 1.0), failed)) == 1.0


def test_a_not_applicable_metric_is_excluded_too() -> None:
    skipped = MetricScore.not_applicable(
        metric_id="b", metric_name="b", weight=1.0, reason="no reference answer"
    )
    assert grade_case((score("a", 0.4), skipped)) == 0.4


def test_a_zero_weight_metric_does_not_move_the_grade() -> None:
    assert grade_case((score("a", 1.0, weight=1.0), score("b", 0.0, weight=0.0))) == 1.0


def test_a_case_with_no_usable_scores_has_no_grade() -> None:
    assert grade_case(()) is None
    assert grade_case((score("a", 1.0, weight=0.0),)) is None


def test_the_overall_grade_is_the_plain_mean_of_the_case_grades() -> None:
    assert grade_run((1.0, 0.5, 0.0)) == Grade(value=0.5)


def test_cases_without_a_grade_are_left_out_of_the_overall_mean() -> None:
    assert grade_run((1.0, None, 0.0)) == Grade(value=0.5)


def test_a_run_where_nothing_could_be_graded_reports_no_grade() -> None:
    assert grade_run((None, None)) == Grade(value=None)
    assert grade_run(()) == Grade(value=None)


def test_a_displayed_grade_can_be_reproduced_by_hand() -> None:
    scores = (score("a", 0.9, weight=2.0), score("b", 0.6, weight=1.0))
    # (0.9*2 + 0.6*1) / 3 == 0.8, up to IEEE-754 noise.
    case = grade_case(scores)
    assert case == pytest.approx(0.8)
    assert grade_run((case,)).percentage == 80.0
    assert Grade(value=0.8).letter == "B"
