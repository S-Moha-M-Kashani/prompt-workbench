"""The whole workflow, offline, with fake providers.

One test walks brief → confirmation → dataset → candidates → manual run →
custom metric → explicit evaluation, and the rest cover what happens when
something upstream changes underneath an artifact that was already built.

No provider or CLI is reached. Every model call is a function defined here, so a
failure in this file is a failure in the workbench, never in someone's network.
"""

import json
from datetime import UTC, datetime

import pytest

from prompt_workbench.core import (
    candidate_generation,
    dataset_generation,
    discovery,
    evaluation,
    testing,
)
from prompt_workbench.core.chat_memory import ThreadStore
from prompt_workbench.core.evaluation import EvaluationBlocked
from prompt_workbench.core.workspace import Workspace
from prompt_workbench.models import (
    ModelSettings,
    PromptBrief,
    PromptTechnique,
    sequential_ids,
)

WHEN = datetime(2026, 3, 1, tzinfo=UTC)
PLATFORM = "You are a prompt engineer."
MODEL = "openai/gpt-4o-mini"


def a_workspace() -> tuple[Workspace, ThreadStore]:
    ids = sequential_ids()
    return Workspace(new_id=ids, clock=lambda: WHEN), ThreadStore(
        new_id=ids, clock=lambda: WHEN
    )


DISCOVERY_REPLY = json.dumps(
    {
        "reply": "Who reads the summaries?",
        "brief": {
            "purpose": "Summarize incoming support tickets",
            "audience": "Team leads",
            "inputs": "The raw ticket text",
            "desired_behaviour": "State the problem and what the customer wants",
            "constraints": "Never invent order details",
            "output_format": "JSON with a summary field",
            "examples": "A late delivery ticket",
            "failure_cases": "Inventing a refund policy",
        },
        "ready": True,
    }
)

DATASET_REPLY = json.dumps(
    {
        "cases": [
            {
                "test_message": "Where is my order?",
                "required_criteria": ["names the missing order"],
                "forbidden_behaviours": ["promises a refund"],
                "tags": ["delivery"],
                "reference_answer": None,
                "category": "normal",
            },
            {
                "test_message": "!!!",
                "required_criteria": ["asks for more detail"],
                "forbidden_behaviours": ["guesses the problem"],
                "tags": ["edge"],
                "reference_answer": None,
                "category": "edge",
            },
        ]
    }
)


def scripted(*replies: str):  # type: ignore[no-untyped-def]
    """A completion that returns each scripted reply in turn."""
    queue = list(replies)

    def complete(messages, *, model=None, settings=None, response_format=None):  # type: ignore[no-untyped-def]
        return queue.pop(0) if queue else replies[-1]

    return complete


def judge_scoring(value: float):  # type: ignore[no-untyped-def]
    def judge(messages: list[dict[str, str]]) -> str:
        return json.dumps({"score": value, "reason": "as scored by the fake judge"})

    return judge


def test_the_whole_workflow_runs_offline_from_brief_to_grade() -> None:
    workspace, store = a_workspace()

    # 1. Discovery fills the brief in, and the user confirms it.
    thread_id = store.open("discovery")
    result = discovery.clarify(
        store=store,
        thread_id=thread_id,
        user_message="I need to summarize support tickets",
        brief=PromptBrief(),
        platform_instruction=PLATFORM,
        complete=scripted(DISCOVERY_REPLY),
        model=MODEL,
        settings=ModelSettings(),
    )
    assert result.ready
    workspace.update_draft(result.brief)
    brief = workspace.confirm_brief()
    assert brief.revision == 1

    # 2. Ground truth comes from the confirmed snapshot only.
    workspace.set_dataset(
        dataset_generation.generate(
            brief=brief,
            count=2,
            platform_instruction=PLATFORM,
            complete=scripted(DATASET_REPLY),
            model=MODEL,
            settings=ModelSettings(),
            new_id=workspace.new_id,
            clock=workspace.clock,
        )
    )
    assert workspace.dataset is not None and len(workspace.dataset) == 2

    # 3. Candidates: one per technique, few-shot fed by the visible dataset.
    workspace.set_candidates(
        candidate_generation.generate(
            brief=brief,
            dataset=workspace.dataset,
            techniques=(PromptTechnique.DIRECT, PromptTechnique.FEW_SHOT),
            platform_instruction=PLATFORM,
            complete=scripted("You are a ticket summarizer."),
            model=MODEL,
            settings=ModelSettings(),
            new_id=workspace.new_id,
            clock=workspace.clock,
        )
    )
    assert len(workspace.candidates) == 2

    # 4. The user edits one candidate by hand; only that one changes.
    candidate = workspace.candidates[0]
    workspace.edit_candidate(candidate.id, "You are a careful ticket summarizer.")
    assert workspace.candidate(candidate.id).revision == 2
    assert workspace.candidates[1].revision == 1

    # 5. A manual run against a real test case, in its own pinned thread.
    candidate = workspace.candidate(candidate.id)
    case = workspace.dataset.cases[0]
    test_thread = testing.open_test_thread(
        store, candidate=candidate, model=MODEL, settings=ModelSettings()
    )
    record = testing.send_test_message(
        store=store,
        thread_id=test_thread,
        candidate=candidate,
        brief=brief,
        case=case,
        user_message=case.test_message,
        model=MODEL,
        settings=ModelSettings(),
        complete=scripted('{"summary": "The order has not arrived."}'),
        new_id=workspace.new_id,
        clock=workspace.clock,
    )
    workspace.record_execution(record)
    assert record.is_attributable

    # 6. A custom metric joins the built-in ones.
    workspace.add_metric(
        name="Brevity", rubric="Score how concise the summary is for a busy reader."
    )

    # 7. Nothing has been scored yet — evaluation only happens when asked for.
    assert workspace.evaluations == ()

    run = evaluation.run(
        executions=workspace.executions,
        dataset=workspace.dataset,
        brief=brief,
        candidate=candidate,
        metrics=workspace.metrics,
        judge=judge_scoring(0.8),
        judge_backend="codex",
        judge_model="gpt-5.6-luna",
        new_id=workspace.new_id,
        clock=workspace.clock,
    )
    workspace.record_evaluation(run)

    # 8. The grade is reproducible by hand from what is on screen.
    #
    # Four metrics were enabled. The fake judge scored 0.8 for each of the three
    # rubric metrics it was asked about; format compliance needed no judge at all
    # and scored 1.0, because the brief asks for JSON and the response is JSON.
    # Reference similarity was skipped — this case has no reference answer — and
    # is excluded from the mean rather than counted as zero.
    #   (0.8 + 0.8 + 0.8 + 1.0) / 4 == 0.85
    scores = {s.metric_name: s for s in run.cases[0].scores}
    assert scores["Format compliance"].score == 1.0
    assert scores["Reference similarity"].applicable is False
    assert scores["Reference similarity"].score is None
    assert run.overall.percentage == 85.0
    assert run.overall.letter == "A"
    assert run.candidate == candidate.ref
    assert run.source_brief == brief.ref
    assert run.source_dataset == workspace.dataset.ref


def test_editing_the_brief_marks_the_downstream_artifacts_stale() -> None:
    workspace, _ = a_workspace()
    workspace.update_draft(PromptBrief(purpose="first"))
    brief = workspace.confirm_brief()
    workspace.set_dataset(
        dataset_generation.generate(
            brief=brief,
            count=2,
            platform_instruction=PLATFORM,
            complete=scripted(DATASET_REPLY),
            model=MODEL,
            settings=ModelSettings(),
            new_id=workspace.new_id,
            clock=workspace.clock,
        )
    )
    assert workspace.stale_artifacts() == ()

    workspace.update_draft(PromptBrief(purpose="completely different"))
    workspace.confirm_brief()

    stale = workspace.stale_artifacts()
    assert stale, "an artifact built from an older brief revision must be flagged"
    # Nothing was deleted to make the point.
    assert workspace.dataset is not None and len(workspace.dataset) == 2


def test_removing_a_case_blocks_evaluation_of_a_response_that_used_it() -> None:
    workspace, store = a_workspace()
    workspace.update_draft(PromptBrief(purpose="x"))
    brief = workspace.confirm_brief()
    workspace.set_dataset(
        dataset_generation.generate(
            brief=brief,
            count=2,
            platform_instruction=PLATFORM,
            complete=scripted(DATASET_REPLY),
            model=MODEL,
            settings=ModelSettings(),
            new_id=workspace.new_id,
            clock=workspace.clock,
        )
    )
    workspace.set_candidates(
        candidate_generation.generate(
            brief=brief,
            dataset=workspace.dataset,
            techniques=(PromptTechnique.DIRECT,),
            platform_instruction=PLATFORM,
            complete=scripted("You are a summarizer."),
            model=MODEL,
            settings=ModelSettings(),
            new_id=workspace.new_id,
            clock=workspace.clock,
        )
    )
    candidate = workspace.candidates[0]
    case = workspace.dataset.cases[0]  # type: ignore[union-attr]
    thread = testing.open_test_thread(
        store, candidate=candidate, model=MODEL, settings=ModelSettings()
    )
    workspace.record_execution(
        testing.send_test_message(
            store=store,
            thread_id=thread,
            candidate=candidate,
            brief=brief,
            case=case,
            user_message=case.test_message,
            model=MODEL,
            settings=ModelSettings(),
            complete=scripted("a reply"),
            new_id=workspace.new_id,
            clock=workspace.clock,
        )
    )
    workspace.remove_case(case.id)

    with pytest.raises(EvaluationBlocked, match="no longer in the dataset"):
        evaluation.run(
            executions=workspace.executions,
            dataset=workspace.dataset,  # type: ignore[arg-type]
            brief=brief,
            candidate=candidate,
            metrics=workspace.metrics,
            judge=judge_scoring(1.0),
            judge_backend="codex",
            judge_model="m",
            new_id=workspace.new_id,
            clock=workspace.clock,
        )


def test_a_provider_failure_mid_workflow_destroys_nothing() -> None:
    workspace, _ = a_workspace()
    workspace.update_draft(PromptBrief(purpose="keep me"))
    brief = workspace.confirm_brief()
    workspace.set_dataset(
        dataset_generation.generate(
            brief=brief,
            count=2,
            platform_instruction=PLATFORM,
            complete=scripted(DATASET_REPLY),
            model=MODEL,
            settings=ModelSettings(),
            new_id=workspace.new_id,
            clock=workspace.clock,
        )
    )

    def failing(messages, *, model=None, settings=None, response_format=None):  # type: ignore[no-untyped-def]
        raise RuntimeError("provider is unreachable")

    with pytest.raises(RuntimeError):
        candidate_generation.generate(
            brief=brief,
            dataset=workspace.dataset,
            techniques=(PromptTechnique.DIRECT,),
            platform_instruction=PLATFORM,
            complete=failing,
            model=MODEL,
            settings=ModelSettings(),
            new_id=workspace.new_id,
            clock=workspace.clock,
        )

    assert workspace.confirmed_brief == brief
    assert workspace.dataset is not None and len(workspace.dataset) == 2
    assert workspace.candidates == ()


def test_a_partial_judge_failure_still_produces_a_grade_and_shows_the_gap() -> None:
    workspace, store = a_workspace()
    workspace.update_draft(PromptBrief(purpose="x"))
    brief = workspace.confirm_brief()
    workspace.set_dataset(
        dataset_generation.generate(
            brief=brief,
            count=2,
            platform_instruction=PLATFORM,
            complete=scripted(DATASET_REPLY),
            model=MODEL,
            settings=ModelSettings(),
            new_id=workspace.new_id,
            clock=workspace.clock,
        )
    )
    workspace.set_candidates(
        candidate_generation.generate(
            brief=brief,
            dataset=workspace.dataset,
            techniques=(PromptTechnique.DIRECT,),
            platform_instruction=PLATFORM,
            complete=scripted("You are a summarizer."),
            model=MODEL,
            settings=ModelSettings(),
            new_id=workspace.new_id,
            clock=workspace.clock,
        )
    )
    candidate = workspace.candidates[0]
    case = workspace.dataset.cases[0]  # type: ignore[union-attr]
    thread = testing.open_test_thread(
        store, candidate=candidate, model=MODEL, settings=ModelSettings()
    )
    workspace.record_execution(
        testing.send_test_message(
            store=store,
            thread_id=thread,
            candidate=candidate,
            brief=brief,
            case=case,
            user_message=case.test_message,
            model=MODEL,
            settings=ModelSettings(),
            complete=scripted("a reply"),
            new_id=workspace.new_id,
            clock=workspace.clock,
        )
    )

    calls = {"n": 0}

    def flaky(messages: list[dict[str, str]]) -> str:
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("codex did not answer within 600s")
        return json.dumps({"score": 1.0, "reason": "fine"})

    run = evaluation.run(
        executions=workspace.executions,
        dataset=workspace.dataset,  # type: ignore[arg-type]
        brief=brief,
        candidate=candidate,
        metrics=workspace.metrics,
        judge=flaky,
        judge_backend="codex",
        judge_model="m",
        new_id=workspace.new_id,
        clock=workspace.clock,
    )

    assert run.has_failures
    assert "did not answer" in run.failures[0].failure
    # The failure was excluded from the mean, not counted as zero or as a half.
    assert run.overall.percentage == 100.0


def test_a_second_evaluation_never_rewrites_the_first() -> None:
    workspace, _ = a_workspace()
    assert workspace.evaluations == ()
    # Recording two runs keeps both; a completed run is immutable.
    from prompt_workbench.models import Grade, EvaluationRun, SourceRef

    def a_run(run_id: str, value: float) -> EvaluationRun:
        return EvaluationRun(
            id=run_id,
            created_at=WHEN,
            candidate=SourceRef(id="c", revision=1),
            source_brief=SourceRef(id="b", revision=1),
            source_dataset=SourceRef(id="d", revision=1),
            metrics=(),
            judge_backend="codex",
            judge_model="m",
            cases=(),
            overall=Grade(value=value),
        )

    workspace.record_evaluation(a_run("eval-1", 0.5))
    workspace.record_evaluation(a_run("eval-2", 0.9))
    assert [run.id for run in workspace.evaluations] == ["eval-1", "eval-2"]
    assert workspace.evaluations[0].overall.value == 0.5
    assert workspace.latest_evaluation is not None
    assert workspace.latest_evaluation.id == "eval-2"
