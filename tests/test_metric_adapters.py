"""The two ways a metric produces a number, and how each one fails."""

import json

import pytest

from prompt_workbench.models.evaluation import EvaluationEvidence
from prompt_workbench.models.ground_truth import GroundTruthCase
from prompt_workbench.models.metrics import MetricDefinition, MetricKind
from prompt_workbench.services import metric_adapters, metric_catalog


def evidence(response: str, *, output_format: str = "", **case_overrides: object) -> EvaluationEvidence:
    fields: dict[str, object] = {
        "id": "case-1",
        "test_message": "My order never arrived.",
        "required_criteria": ("acknowledges the delay", "offers a next step"),
        "forbidden_behaviours": ("promises a refund",),
    }
    fields.update(case_overrides)
    return EvaluationEvidence(
        situation="You answer support tickets from an unhappy customer.",
        output_format=output_format,
        case=GroundTruthCase(**fields),  # type: ignore[arg-type]
        system_prompt="You are a support assistant.",
        user_message="My order never arrived.",
        response=response,
    )


# --- the built-in catalog -------------------------------------------------


def test_the_catalog_ships_the_four_agreed_metrics() -> None:
    names = {metric.check or metric.id for metric in metric_catalog.built_in_metrics()}
    assert {
        "criteria_coverage",
        "forbidden_behaviour",
        "format_compliance",
        "reference_similarity",
    } <= names


def test_every_built_in_metric_is_valid_and_marked_as_built_in() -> None:
    for metric in metric_catalog.built_in_metrics():
        assert metric.builtin
        assert metric.rubric.strip()
        assert metric.weight >= 0


def test_format_compliance_is_the_only_metric_that_needs_no_judge() -> None:
    deterministic = [
        m for m in metric_catalog.built_in_metrics() if m.kind is MetricKind.DETERMINISTIC
    ]
    assert [m.id for m in deterministic] == ["format_compliance"]


def test_reference_similarity_only_applies_to_cases_that_have_a_reference() -> None:
    metric = metric_catalog.get("reference_similarity")
    assert metric.requires_reference


def test_every_deterministic_metric_names_a_check_that_exists() -> None:
    for metric in metric_catalog.built_in_metrics():
        if metric.kind is MetricKind.DETERMINISTIC:
            assert metric.check in metric_adapters.DETERMINISTIC_CHECKS


# --- the deterministic path -----------------------------------------------


def test_a_json_output_format_is_checked_by_parsing_the_response() -> None:
    good = metric_adapters.format_compliance(evidence('{"reply": "ok"}', output_format="JSON object"))
    bad = metric_adapters.format_compliance(evidence("Sorry about that!", output_format="JSON object"))
    assert good.score == 1.0
    assert bad.score == 0.0
    assert "json" in bad.reason.lower()


def test_a_word_limit_in_the_output_format_is_enforced() -> None:
    within = metric_adapters.format_compliance(
        evidence("one two three", output_format="At most 5 words.")
    )
    over = metric_adapters.format_compliance(
        evidence("one two three four five six", output_format="At most 5 words.")
    )
    assert within.score == 1.0
    assert over.score == 0.0


def test_a_format_with_nothing_checkable_reports_not_applicable_not_a_free_pass() -> None:
    score = metric_adapters.format_compliance(
        evidence("anything at all", output_format="A warm, friendly tone.")
    )
    assert score.score is None
    assert not score.applicable
    assert not score.failed


def test_a_brief_with_no_stated_output_format_is_also_not_applicable() -> None:
    score = metric_adapters.format_compliance(evidence("anything", output_format=""))
    assert not score.applicable


def test_several_checkable_rules_are_scored_as_a_fraction() -> None:
    # Parses as JSON, but is over the word limit: one of two rules met.
    score = metric_adapters.format_compliance(
        evidence('{"a": "one two three four five six"}', output_format="JSON, at most 3 words.")
    )
    assert score.score == 0.5


def test_the_deterministic_path_makes_no_judge_call() -> None:
    def explode(messages: list[dict[str, str]]) -> str:
        raise AssertionError("a deterministic metric must never call a judge")

    metric_adapters.score_metric(
        metric_catalog.get("format_compliance"), evidence("{}", output_format="JSON"), judge=explode
    )


# --- the judge path -------------------------------------------------------


def a_judge(reply: str) -> metric_adapters.JudgeFn:
    def judge(messages: list[dict[str, str]]) -> str:
        return reply

    return judge


def rubric_metric() -> MetricDefinition:
    return metric_catalog.get("criteria_coverage")


def test_a_judge_reply_becomes_a_score_and_a_reason() -> None:
    score = metric_adapters.score_metric(
        rubric_metric(),
        evidence("I am sorry for the delay; I have opened a trace."),
        judge=a_judge(json.dumps({"score": 0.8, "reason": "both criteria met"})),
    )
    assert score.score == 0.8
    assert score.reason == "both criteria met"
    assert not score.failed


def test_a_judge_reply_wrapped_in_prose_still_yields_its_json() -> None:
    score = metric_adapters.score_metric(
        rubric_metric(),
        evidence("ok"),
        judge=a_judge('Here is my verdict:\n{"score": 0.5, "reason": "partial"}\nThanks!'),
    )
    assert score.score == 0.5


def test_a_judge_score_outside_the_range_is_a_failure_not_a_clamp() -> None:
    score = metric_adapters.score_metric(
        rubric_metric(), evidence("ok"), judge=a_judge('{"score": 4, "reason": "great"}')
    )
    assert score.failed
    assert score.score is None


def test_malformed_judge_output_is_a_visible_failure() -> None:
    score = metric_adapters.score_metric(
        rubric_metric(), evidence("ok"), judge=a_judge("I would rather not say.")
    )
    assert score.failed
    assert score.score is None
    assert "could not" in score.failure.lower() or "no score" in score.failure.lower()


def test_a_judge_that_raises_becomes_a_failure_carrying_its_message() -> None:
    def broken(messages: list[dict[str, str]]) -> str:
        raise RuntimeError("codex exited 1: not logged in")

    score = metric_adapters.score_metric(rubric_metric(), evidence("ok"), judge=broken)
    assert score.failed
    assert "not logged in" in score.failure


def test_the_judge_prompt_carries_the_rubric_the_case_and_the_response() -> None:
    seen: list[dict[str, str]] = []

    def judge(messages: list[dict[str, str]]) -> str:
        seen.extend(messages)
        return '{"score": 1, "reason": "ok"}'

    metric = rubric_metric()
    metric_adapters.score_metric(metric, evidence("my response text"), judge=judge)
    prompt = "\n".join(m["content"] for m in seen)
    assert metric.rubric in prompt
    assert "acknowledges the delay" in prompt
    assert "my response text" in prompt
    assert "You are a support assistant." in prompt


def test_a_reference_metric_is_skipped_when_the_case_has_no_reference() -> None:
    score = metric_adapters.score_metric(
        metric_catalog.get("reference_similarity"),
        evidence("anything"),
        judge=a_judge('{"score": 1, "reason": "x"}'),
    )
    assert not score.applicable
    assert score.score is None
    assert "reference" in score.reason.lower()


def test_a_reference_metric_runs_when_the_case_does_have_a_reference() -> None:
    score = metric_adapters.score_metric(
        metric_catalog.get("reference_similarity"),
        evidence("close enough", reference_answer="the ideal answer"),
        judge=a_judge('{"score": 0.9, "reason": "close"}'),
    )
    assert score.score == 0.9


def test_a_custom_metric_scores_through_the_same_path_as_a_built_in() -> None:
    custom = MetricDefinition(
        id="m-custom",
        name="Warmth",
        rubric="Score how warm and human the reply sounds to an upset customer.",
        weight=0.5,
    )
    score = metric_adapters.score_metric(
        custom, evidence("ok"), judge=a_judge('{"score": 0.3, "reason": "curt"}')
    )
    assert (score.metric_name, score.weight, score.score) == ("Warmth", 0.5, 0.3)


def test_an_unknown_deterministic_check_fails_loudly() -> None:
    broken = MetricDefinition(
        id="m-x",
        name="Nonsense",
        rubric="This names a check that does not exist anywhere.",
        kind=MetricKind.DETERMINISTIC,
        check="no_such_check",
    )
    score = metric_adapters.score_metric(broken, evidence("ok"), judge=a_judge("{}"))
    assert score.failed
    assert "no_such_check" in score.failure
