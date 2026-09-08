"""The bridge to deepeval: preflight, per-metric outcomes, aggregation."""

from datetime import UTC, datetime

import pytest

from prompt_workbench.core import scoring
from prompt_workbench.core.scoring import MetricOutcome
from prompt_workbench.models.case import CaseBrief, EvalCase
from prompt_workbench.services import deepeval_metrics

WHEN = datetime(2026, 3, 1, tzinfo=UTC)


def a_brief(**overrides) -> CaseBrief:  # type: ignore[no-untyped-def]
    fields = {
        "description": "Sort tickets",
        "task_type_key": "classification",
        "created_at": WHEN,
        "cases": (EvalCase(id="c1", input="charged twice", expected_output="billing"),),
    }
    fields.update(overrides)
    return CaseBrief(**fields)  # type: ignore[arg-type]


def outcome(score: float | None, threshold: float, *, higher: bool = True,
            failed: bool = False) -> MetricOutcome:
    return MetricOutcome(
        metric_key="m", metric_label="M", score=score, threshold=threshold,
        higher_is_better=higher, failed=failed,
    )


# --- passing --------------------------------------------------------------


def test_a_higher_is_better_metric_passes_at_or_above_its_threshold() -> None:
    assert outcome(0.9, 0.8).passed
    assert outcome(0.8, 0.8).passed
    assert not outcome(0.79, 0.8).passed


def test_a_lower_is_better_metric_passes_at_or_below_its_threshold() -> None:
    """Reading a hallucination score the wrong way round would invert the verdict."""
    assert outcome(0.1, 0.2, higher=False).passed
    assert not outcome(0.5, 0.2, higher=False).passed


def test_a_failed_metric_never_counts_as_passed() -> None:
    assert not outcome(None, 0.8, failed=True).passed


# --- preflight ------------------------------------------------------------


def test_a_metric_whose_fields_the_cases_lack_is_refused_before_running() -> None:
    problems = scoring.preflight(
        brief=a_brief(), choices=deepeval_metrics.suggested_for(("faithfulness",))
    )
    assert any("retrieval_context" in p for p in problems)


def test_a_metric_whose_fields_are_present_passes_preflight() -> None:
    brief = a_brief(
        cases=(EvalCase(id="c1", input="q", retrieval_context=("a doc",)),)
    )
    assert scoring.preflight(
        brief=brief, choices=deepeval_metrics.suggested_for(("faithfulness",))
    ) == ()


def test_an_unconfigured_metric_is_refused_before_running() -> None:
    problems = scoring.preflight(
        brief=a_brief(), choices=deepeval_metrics.suggested_for(("misuse",))
    )
    assert any("domain" in p for p in problems)


def test_no_cases_and_no_metrics_are_both_reported() -> None:
    problems = scoring.preflight(brief=a_brief(cases=()), choices=())
    assert any("no test cases" in p.lower() for p in problems)
    assert any("no metric" in p.lower() for p in problems)


def test_a_disabled_metric_is_not_preflighted() -> None:
    from dataclasses import replace

    choice = replace(deepeval_metrics.suggested_for(("faithfulness",))[0], enabled=False)
    problems = scoring.preflight(brief=a_brief(), choices=(choice,))
    assert not any("retrieval_context" in p for p in problems)


# --- aggregation ----------------------------------------------------------


def test_directions_are_normalised_before_averaging() -> None:
    """Faithfulness 0.9 and hallucination 0.1 are both good. Averaging them raw
    would give 0.5 and read as mediocre."""
    combined = scoring.aggregate((outcome(0.9, 0.8), outcome(0.1, 0.2, higher=False)))
    assert combined == pytest.approx(0.9)


def test_failed_metrics_are_excluded_rather_than_scored_as_zero() -> None:
    combined = scoring.aggregate((outcome(1.0, 0.8), outcome(None, 0.8, failed=True)))
    assert combined == pytest.approx(1.0)


def test_nothing_scoreable_yields_no_aggregate() -> None:
    assert scoring.aggregate(()) is None
    assert scoring.aggregate((outcome(None, 0.8, failed=True),)) is None


def test_every_threshold_met_requires_every_metric_to_pass() -> None:
    assert scoring.met_every_threshold((outcome(0.9, 0.8), outcome(0.1, 0.2, higher=False)))
    assert not scoring.met_every_threshold((outcome(0.9, 0.8), outcome(0.5, 0.2, higher=False)))


def test_a_configuration_whose_measurement_broke_has_not_been_shown_to_work() -> None:
    assert not scoring.met_every_threshold((outcome(1.0, 0.8), outcome(None, 0.8, failed=True)))
    assert not scoring.met_every_threshold(())


# --- the deepeval bridge --------------------------------------------------


def test_a_case_becomes_the_object_deepeval_scores() -> None:
    pytest.importorskip("deepeval")
    case = EvalCase(id="c1", input="q", expected_output="a", retrieval_context=("doc",))
    llm_case = scoring.as_llm_test_case(case, "the response")
    assert llm_case.input == "q"
    assert llm_case.actual_output == "the response"
    assert llm_case.expected_output == "a"
    assert llm_case.retrieval_context == ["doc"]


def test_absent_context_is_none_rather_than_an_empty_list() -> None:
    """deepeval distinguishes 'no context' from 'empty context'."""
    pytest.importorskip("deepeval")
    llm_case = scoring.as_llm_test_case(EvalCase(id="c1", input="q"), "r")
    assert llm_case.retrieval_context is None
    assert llm_case.context is None


def test_a_metric_that_raises_becomes_a_recorded_failure_not_an_exception() -> None:
    pytest.importorskip("deepeval")

    class BrokenJudge:
        pass

    result = scoring.score_one(
        case=EvalCase(id="c1", input="q"),
        response="r",
        choice=deepeval_metrics.suggested_for(("faithfulness",))[0],
        judge=BrokenJudge(),
    )
    assert result.failed
    assert result.score is None
    assert result.failure
