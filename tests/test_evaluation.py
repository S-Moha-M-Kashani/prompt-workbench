"""On-demand evaluation: the command boundary, preflight, and the run itself."""

import json
from datetime import UTC, datetime

import pytest

from prompt_workbench.core import evaluation, metric_config
from prompt_workbench.core.evaluation import EvaluationBlocked
from prompt_workbench.models import (
    BriefSnapshot,
    CandidatePrompt,
    ExecutionRecord,
    GroundTruthCase,
    GroundTruthDataset,
    MetricDefinition,
    ModelSettings,
    PromptBrief,
    PromptTechnique,
    SourceRef,
    sequential_ids,
)
from prompt_workbench.services import metric_catalog

WHEN = datetime(2026, 3, 1, tzinfo=UTC)


def a_brief() -> BriefSnapshot:
    return BriefSnapshot(
        id="brief-1",
        revision=1,
        brief=PromptBrief(purpose="Answer support tickets", output_format="Plain prose"),
        created_at=WHEN,
    )


def a_candidate() -> CandidatePrompt:
    return CandidatePrompt(
        id="cand-1",
        technique=PromptTechnique.DIRECT,
        system_prompt="You are a support assistant.",
        revision=1,
        source_brief=SourceRef(id="brief-1", revision=1),
        created_at=WHEN,
    )


def a_dataset(n: int = 2) -> GroundTruthDataset:
    return GroundTruthDataset(
        id="ds-1",
        revision=1,
        source_brief=SourceRef(id="brief-1", revision=1),
        cases=tuple(
            GroundTruthCase(
                id=f"case-{i}",
                test_message=f"message {i}",
                required_criteria=("is specific",),
            )
            for i in range(1, n + 1)
        ),
        created_at=WHEN,
    )


def an_execution(case_id: str | None = "case-1", response: str = "a reply") -> ExecutionRecord:
    return ExecutionRecord(
        id=f"exec-{case_id}",
        thread_id="t-1",
        candidate=SourceRef(id="cand-1", revision=1),
        source_brief=SourceRef(id="brief-1", revision=1),
        case_id=case_id,
        model_id="m",
        settings=ModelSettings(),
        user_message="message 1",
        response=response,
        created_at=WHEN,
    )


def judge_scoring(value: float):  # type: ignore[no-untyped-def]
    def judge(messages: list[dict[str, str]]) -> str:
        return json.dumps({"score": value, "reason": "because"})

    return judge


def metrics() -> tuple[MetricDefinition, ...]:
    return (metric_catalog.get("criteria_coverage"), metric_catalog.get("forbidden_behaviour"))


# --- metric configuration validation --------------------------------------


def test_a_configuration_with_no_enabled_metric_is_rejected() -> None:
    problems = metric_config.validate(
        tuple(m.with_enabled(False) for m in metric_catalog.built_in_metrics())
    )
    assert any("no metric" in p.lower() for p in problems)


def test_a_configuration_whose_enabled_weights_are_all_zero_is_rejected() -> None:
    problems = metric_config.validate(tuple(m.with_weight(0.0) for m in metrics()))
    assert any("weight" in p.lower() for p in problems)


def test_two_metrics_with_the_same_name_are_rejected() -> None:
    duplicate = MetricDefinition(
        id="m-2", name="Criteria coverage", rubric="A different rubric entirely, but same name."
    )
    problems = metric_config.validate((metric_catalog.get("criteria_coverage"), duplicate))
    assert any("unique" in p.lower() or "duplicate" in p.lower() for p in problems)


def test_a_valid_configuration_reports_no_problems() -> None:
    assert metric_config.validate(metrics()) == ()


def test_only_enabled_metrics_are_selected_for_a_run() -> None:
    chosen = metric_config.selected(
        (metric_catalog.get("criteria_coverage"), metric_catalog.get("forbidden_behaviour").with_enabled(False))
    )
    assert [m.id for m in chosen] == ["criteria_coverage"]


# --- preflight ------------------------------------------------------------


def test_preflight_rejects_a_response_with_no_test_case() -> None:
    problems = evaluation.preflight(
        executions=(an_execution(case_id=None),), dataset=a_dataset(), metrics=metrics()
    )
    assert any("test case" in p for p in problems)


def test_preflight_rejects_a_response_whose_case_is_no_longer_in_the_dataset() -> None:
    problems = evaluation.preflight(
        executions=(an_execution(case_id="case-99"),), dataset=a_dataset(), metrics=metrics()
    )
    assert any("case-99" in p for p in problems)


def test_preflight_rejects_an_empty_selection() -> None:
    problems = evaluation.preflight(executions=(), dataset=a_dataset(), metrics=metrics())
    assert any("no response" in p.lower() for p in problems)


def test_preflight_passes_a_complete_selection() -> None:
    assert evaluation.preflight(
        executions=(an_execution(),), dataset=a_dataset(), metrics=metrics()
    ) == ()


# --- the run --------------------------------------------------------------


def run_with(**overrides):  # type: ignore[no-untyped-def]
    kwargs = dict(
        executions=(an_execution(),),
        dataset=a_dataset(),
        brief=a_brief(),
        candidate=a_candidate(),
        metrics=metrics(),
        judge=judge_scoring(0.5),
        judge_backend="codex",
        judge_model="gpt-5.6-luna",
        new_id=sequential_ids(),
        clock=lambda: WHEN,
    )
    kwargs.update(overrides)
    return evaluation.run(**kwargs)  # type: ignore[arg-type]


def test_a_run_scores_every_selected_response_against_every_metric() -> None:
    result = run_with(executions=(an_execution("case-1"), an_execution("case-2")))
    assert len(result.cases) == 2
    assert all(len(case.scores) == 2 for case in result.cases)


def test_a_run_records_which_judge_produced_it() -> None:
    result = run_with()
    assert (result.judge_backend, result.judge_model) == ("codex", "gpt-5.6-luna")


def test_a_run_records_every_source_snapshot() -> None:
    result = run_with()
    assert result.candidate == SourceRef(id="cand-1", revision=1)
    assert result.source_brief == SourceRef(id="brief-1", revision=1)
    assert result.source_dataset == SourceRef(id="ds-1", revision=1)


def test_the_overall_grade_follows_from_the_scores_on_screen() -> None:
    result = run_with(judge=judge_scoring(0.5))
    assert result.overall.percentage == 50.0
    assert result.overall.letter == "F"


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
    # The surviving metric still produced a grade.
    assert result.overall.percentage == 100.0


def test_a_run_refuses_a_selection_that_fails_preflight() -> None:
    with pytest.raises(EvaluationBlocked) as raised:
        run_with(executions=(an_execution(case_id=None),))
    assert "test case" in str(raised.value)


def test_a_run_refuses_an_invalid_metric_configuration_before_any_judge_call() -> None:
    def explode(messages: list[dict[str, str]]) -> str:
        raise AssertionError("no judge call may happen once the configuration is invalid")

    with pytest.raises(EvaluationBlocked):
        run_with(metrics=tuple(m.with_weight(0.0) for m in metrics()), judge=explode)


def test_no_generation_or_testing_module_can_reach_the_evaluator() -> None:
    """The command boundary, checked in the import graph rather than in prose.

    If none of these modules imports the evaluator, then editing, generating, or
    manually testing something cannot possibly start a scoring call."""
    import ast
    import inspect

    from prompt_workbench.core import candidate_generation, dataset_generation, discovery, testing

    forbidden = {"prompt_workbench.core.evaluation", "prompt_workbench.services.metric_adapters"}
    for module in (discovery, dataset_generation, candidate_generation, testing):
        tree = ast.parse(inspect.getsource(module))
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)
                imported.update(f"{node.module}.{alias.name}" for alias in node.names)
        assert not (imported & forbidden), (
            f"{module.__name__} imports {imported & forbidden}, so it could start an "
            "evaluation without the user asking for one"
        )
