"""On-demand evaluation against a use case's own expectations."""

import json
from datetime import UTC, datetime

import pytest

from prompt_workbench.core import evaluation, metric_config
from prompt_workbench.core.evaluation import EvaluationBlocked
from prompt_workbench.models import MetricDefinition, PromptRun, sequential_ids
from prompt_workbench.services import metric_catalog, use_case_catalog

WHEN = datetime(2026, 3, 1, tzinfo=UTC)


def a_use_case():  # type: ignore[no-untyped-def]
    return use_case_catalog.get("grounded_briefing")


def a_run(run_id: str = "run-1", response: str = "There is no previous history.") -> PromptRun:
    return PromptRun(
        id=run_id,
        use_case_key="grounded_briefing",
        prompt_revision=1,
        system_prompt="Use only the history given. Return at most 120 words.",
        model_id="openai/gpt-4o-mini",
        user_message="What should I know about this learner?",
        response=response,
        created_at=WHEN,
    )


def judge_scoring(value: float):  # type: ignore[no-untyped-def]
    def judge(messages: list[dict[str, str]]) -> str:
        return json.dumps({"score": value, "reason": "because"})

    return judge


def metrics() -> tuple[MetricDefinition, ...]:
    return (metric_catalog.get("criteria_coverage"), metric_catalog.get("forbidden_behaviour"))


# --- metric configuration -------------------------------------------------


def test_a_configuration_with_no_enabled_metric_is_rejected() -> None:
    problems = metric_config.validate(
        tuple(m.with_enabled(False) for m in metric_catalog.built_in_metrics())
    )
    assert any("no metric" in p.lower() for p in problems)


def test_a_configuration_whose_weights_are_all_zero_is_rejected() -> None:
    problems = metric_config.validate(tuple(m.with_weight(0.0) for m in metrics()))
    assert any("weight" in p.lower() for p in problems)


def test_a_valid_configuration_reports_no_problems() -> None:
    assert metric_config.validate(metrics()) == ()


# --- preflight ------------------------------------------------------------


def test_preflight_rejects_an_empty_selection() -> None:
    problems = evaluation.preflight(runs=(), metrics=metrics())
    assert any("no response" in p.lower() for p in problems)


def test_preflight_rejects_a_run_with_an_empty_response() -> None:
    problems = evaluation.preflight(runs=(a_run(response="   "),), metrics=metrics())
    assert any("empty response" in p for p in problems)


def test_preflight_passes_a_complete_selection() -> None:
    assert evaluation.preflight(runs=(a_run(),), metrics=metrics()) == ()


# --- the run --------------------------------------------------------------


def run_with(**overrides):  # type: ignore[no-untyped-def]
    kwargs = dict(
        runs=(a_run(),),
        use_case=a_use_case(),
        metrics=metrics(),
        judge=judge_scoring(0.5),
        judge_backend="codex",
        judge_model="gpt-5.6-luna",
        new_id=sequential_ids(),
        clock=lambda: WHEN,
    )
    kwargs.update(overrides)
    return evaluation.run(**kwargs)  # type: ignore[arg-type]


def test_the_use_case_supplies_the_ground_truth_so_none_is_authored() -> None:
    seen: list[str] = []

    def judge(messages: list[dict[str, str]]) -> str:
        seen.append("\n".join(m["content"] for m in messages))
        return json.dumps({"score": 1.0, "reason": "ok"})

    run_with(judge=judge)
    prompt = "\n".join(seen)
    # The criteria written into the use case reach the judge verbatim.
    assert "states plainly that there is no previous history" in prompt
    assert "invents a past session" in prompt


def test_a_run_records_the_use_case_and_prompt_revision_it_judged() -> None:
    result = run_with()
    assert result.use_case_key == "grounded_briefing"
    assert result.prompt_revision == 1
    assert (result.judge_backend, result.judge_model) == ("codex", "gpt-5.6-luna")


def test_the_grade_follows_from_the_scores_on_screen() -> None:
    result = run_with(judge=judge_scoring(0.5))
    assert result.overall.percentage == 50.0


def test_a_judge_failure_is_visible_and_does_not_stop_the_run() -> None:
    calls = {"n": 0}

    def flaky(messages: list[dict[str, str]]) -> str:
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("codex exited 1: not logged in")
        return json.dumps({"score": 1.0, "reason": "fine"})

    result = run_with(judge=flaky)
    assert result.has_failures
    assert "not logged in" in result.failures[0].failure
    assert result.overall.percentage == 100.0


def test_a_run_refuses_an_invalid_configuration_before_any_judge_call() -> None:
    def explode(messages: list[dict[str, str]]) -> str:
        raise AssertionError("no judge call once the configuration is invalid")

    with pytest.raises(EvaluationBlocked):
        run_with(metrics=tuple(m.with_weight(0.0) for m in metrics()), judge=explode)


def test_a_run_refuses_when_there_is_nothing_to_score() -> None:
    def explode(messages: list[dict[str, str]]) -> str:
        raise AssertionError("no judge call with nothing to score")

    with pytest.raises(EvaluationBlocked, match="no response"):
        run_with(runs=(), judge=explode)


def test_the_output_format_is_read_back_out_of_the_prompt_itself() -> None:
    """With no brief to consult, the deterministic format check reads what the
    prompt actually asked for."""
    declared = evaluation._declared_output_format(
        "You are a judge.\nReturn ONLY valid JSON.\nAt most 40 words.\nBe kind."
    )
    assert "JSON" in declared
    assert "40 words" in declared
    assert "Be kind" not in declared


def test_scoring_a_json_use_case_exercises_the_deterministic_metric() -> None:
    """Format compliance needs no judge, so it must work with a judge that
    refuses to answer at all."""
    def refuse(messages: list[dict[str, str]]) -> str:
        raise RuntimeError("judge unavailable")

    result = run_with(
        runs=(
            PromptRun(
                id="run-1",
                use_case_key="closed_set_classification",
                prompt_revision=1,
                system_prompt="Return ONLY valid JSON in the requested shape.",
                model_id="m",
                user_message="file this",
                response='{"tags": []}',
                created_at=WHEN,
            ),
        ),
        use_case=use_case_catalog.get("closed_set_classification"),
        metrics=(metric_catalog.get("format_compliance"),),
        judge=refuse,
    )
    assert result.overall.percentage == 100.0
    assert not result.has_failures
