"""The only door into scoring.

Evaluation is a decision, not a side effect. Nothing scores anything except
through ``run``, and ``run`` is reached from exactly one place: the explicit
evaluate action. Putting it alone in a module with its own error type makes the
boundary visible in the import graph, so "did editing that prompt just cost me a
judge call?" is answerable by reading the code rather than by watching a bill.

What changed with ready use cases is where ground truth comes from. There is no
dataset to author: a use case's criteria and forbidden behaviours are the same
ones that made it worth trying, so the thing being measured against is the thing
that was explained in the chat.
"""

from collections.abc import Callable, Sequence
from datetime import datetime

from prompt_workbench.core import metric_config
from prompt_workbench.core.grading import grade_case, grade_run
from prompt_workbench.models.evaluation import (
    CaseEvaluation,
    EvaluationEvidence,
    EvaluationRun,
)
from prompt_workbench.models.identifiers import IdFactory
from prompt_workbench.models.metrics import MetricDefinition
from prompt_workbench.models.runs import PromptRun
from prompt_workbench.models.use_case import UseCase
from prompt_workbench.services.metric_adapters import JudgeFn, score_metric


class EvaluationBlocked(RuntimeError):
    """The run was refused before any judge was called, and why."""


def preflight(
    *, runs: Sequence[PromptRun], metrics: Sequence[MetricDefinition]
) -> tuple[str, ...]:
    """Everything that would make this scoring meaningless, checked offline."""
    problems: list[str] = list(metric_config.validate(metrics))

    if not runs:
        problems.append(
            "There is no response to score yet. Switch to end-user mode, send a "
            "message, and evaluate the response that comes back."
        )
    for record in runs:
        if not record.is_scoreable:
            problems.append(f"Run {record.id} recorded an empty response.")
    return tuple(problems)


def run(
    *,
    runs: Sequence[PromptRun],
    use_case: UseCase,
    metrics: Sequence[MetricDefinition],
    judge: JudgeFn,
    judge_backend: str,
    judge_model: str,
    new_id: IdFactory,
    clock: Callable[[], datetime],
) -> EvaluationRun:
    """Score the selected responses against this use case's own expectations.

    A metric that fails is recorded as a failure and the run continues. Stopping
    at the first failure throws away scores already paid for; substituting a
    number for it would be worse.
    """
    problems = preflight(runs=runs, metrics=metrics)
    if problems:
        raise EvaluationBlocked("\n".join(problems))

    chosen = metric_config.selected(metrics)
    cases: list[CaseEvaluation] = []

    for record in runs:
        case = use_case.as_ground_truth_case(
            f"{use_case.key}:{record.id}", user_message=record.user_message
        )
        evidence = EvaluationEvidence(
            situation=use_case.situation,
            output_format=_declared_output_format(record.system_prompt),
            case=case,
            system_prompt=record.system_prompt,
            user_message=record.user_message,
            response=record.response,
        )
        scores = tuple(score_metric(metric, evidence, judge=judge) for metric in chosen)
        cases.append(
            CaseEvaluation(
                run_id=record.id, case_id=case.id, scores=scores, grade=grade_case(scores)
            )
        )

    return EvaluationRun(
        id=new_id("evaluation"),
        created_at=clock(),
        use_case_key=use_case.key,
        prompt_revision=runs[0].prompt_revision,
        metrics=tuple(chosen),
        judge_backend=judge_backend,
        judge_model=judge_model,
        cases=tuple(cases),
        overall=grade_run(tuple(case.grade for case in cases)),
    )


def _declared_output_format(system_prompt: str) -> str:
    """The lines of the prompt that state an output shape.

    The deterministic format check needs to know what shape was asked for. With
    no brief to read it from, it is read back out of the prompt itself — which is
    the honest source anyway: a prompt is only owed the format it actually asks
    for.
    """
    wanted = ("json", "word", "character", "sentence", "format", "shape", "return only")
    return "\n".join(
        line for line in system_prompt.splitlines() if any(w in line.lower() for w in wanted)
    )
