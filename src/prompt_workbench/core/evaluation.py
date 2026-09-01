"""The only door into scoring.

Evaluation is a decision, not a side effect. Nothing in this application scores
anything except through ``run``, and ``run`` is called from exactly one place:
the Evaluate button. That is the whole point of putting it in its own module
with its own error type — the boundary is visible in the import graph, so
"did editing a candidate just cost me a judge call?" is answerable by reading
the code rather than by watching a bill.

Two gates come before any judge call. ``preflight`` checks that every selected
response can actually be mapped back to a test case, and ``metric_config``
checks the metrics are usable. Both report every problem at once.
"""

from collections.abc import Callable, Sequence
from datetime import datetime

from prompt_workbench.core import metric_config
from prompt_workbench.core.grading import grade_case, grade_run
from prompt_workbench.models.brief import BriefSnapshot
from prompt_workbench.models.candidates import CandidatePrompt
from prompt_workbench.models.evaluation import (
    CaseEvaluation,
    EvaluationEvidence,
    EvaluationRun,
)
from prompt_workbench.models.execution import ExecutionRecord
from prompt_workbench.models.ground_truth import GroundTruthDataset
from prompt_workbench.models.identifiers import IdFactory
from prompt_workbench.models.metrics import MetricDefinition
from prompt_workbench.services.metric_adapters import JudgeFn, score_metric


class EvaluationBlocked(RuntimeError):
    """The run was refused before any judge was called, and why."""


def preflight(
    *,
    executions: Sequence[ExecutionRecord],
    dataset: GroundTruthDataset,
    metrics: Sequence[MetricDefinition],
) -> tuple[str, ...]:
    """Everything that would make this run unscoreable, checked without calling out."""
    problems: list[str] = list(metric_config.validate(metrics))

    if not executions:
        problems.append(
            "No response is selected. Run a candidate in the Test area first, then "
            "choose which responses to score."
        )

    for record in executions:
        for reason in record.missing_evidence():
            problems.append(f"Response {record.id} cannot be scored: {reason}.")
        if record.case_id:
            try:
                dataset.case(record.case_id)
            except KeyError:
                problems.append(
                    f"Response {record.id} refers to test case {record.case_id}, which "
                    "is no longer in the dataset. Re-run it against a current case."
                )
    return tuple(problems)


def run(
    *,
    executions: Sequence[ExecutionRecord],
    dataset: GroundTruthDataset,
    brief: BriefSnapshot,
    candidate: CandidatePrompt,
    metrics: Sequence[MetricDefinition],
    judge: JudgeFn,
    judge_backend: str,
    judge_model: str,
    new_id: IdFactory,
    clock: Callable[[], datetime],
) -> EvaluationRun:
    """Score every selected response against every enabled metric.

    A metric that fails is recorded as a failure and the run continues. Stopping
    on the first failure would throw away the scores already paid for, and
    silently substituting a number for it would be worse still.
    """
    problems = preflight(executions=executions, dataset=dataset, metrics=metrics)
    if problems:
        raise EvaluationBlocked("\n".join(problems))

    chosen = metric_config.selected(metrics)
    output_format = brief.brief.output_format
    brief_context = brief.as_context()

    cases: list[CaseEvaluation] = []
    for record in executions:
        # Preflight has already proved the case id resolves.
        case = dataset.case(str(record.case_id))
        evidence = EvaluationEvidence(
            brief_context=brief_context,
            output_format=output_format,
            case=case,
            candidate_prompt=candidate.system_prompt,
            user_message=record.user_message,
            response=record.response,
        )
        scores = tuple(score_metric(metric, evidence, judge=judge) for metric in chosen)
        cases.append(
            CaseEvaluation(
                execution_id=record.id,
                case_id=case.id,
                scores=scores,
                grade=grade_case(scores),
            )
        )

    return EvaluationRun(
        id=new_id("evaluation"),
        created_at=clock(),
        candidate=candidate.ref,
        source_brief=brief.ref,
        source_dataset=dataset.ref,
        metrics=tuple(chosen),
        judge_backend=judge_backend,
        judge_model=judge_model,
        cases=tuple(cases),
        overall=grade_run(tuple(case.grade for case in cases)),
    )
