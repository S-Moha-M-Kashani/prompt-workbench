"""Ground-truth cases, candidate prompts, metrics, threads, executions, grades.

Every artifact records the identifier *and* revision of the snapshot it came
from, so a result can always be traced back to the exact inputs that made it.
"""

from datetime import UTC, datetime

import pytest

from prompt_workbench.models.candidates import CandidatePrompt, PromptTechnique
from prompt_workbench.models.chat import ChatMessage, ChatThread, ThreadConfig
from prompt_workbench.models.evaluation import (
    CaseEvaluation,
    EvaluationRun,
    Grade,
    MetricScore,
)
from prompt_workbench.models.execution import ExecutionRecord
from prompt_workbench.models.ground_truth import (
    CaseCategory,
    GroundTruthCase,
    GroundTruthDataset,
)
from prompt_workbench.models.metrics import MetricDefinition, MetricKind
from prompt_workbench.models.model_settings import ModelSettings
from prompt_workbench.models.provenance import SourceRef

WHEN = datetime(2026, 3, 1, tzinfo=UTC)


def a_case(case_id: str = "case-1", **overrides: object) -> GroundTruthCase:
    fields: dict[str, object] = {
        "id": case_id,
        "test_message": "My order never arrived.",
        "required_criteria": ("acknowledges the delay",),
        "forbidden_behaviours": ("promises a refund",),
        "tags": ("support",),
    }
    fields.update(overrides)
    return GroundTruthCase(**fields)  # type: ignore[arg-type]


# --- ground truth ---------------------------------------------------------


def test_a_case_needs_a_message_and_at_least_one_criterion() -> None:
    with pytest.raises(ValueError, match="test message"):
        a_case(test_message="  ")
    with pytest.raises(ValueError, match="required criterion"):
        a_case(required_criteria=())


def test_a_case_may_omit_a_reference_answer() -> None:
    assert a_case().reference_answer is None
    assert a_case(reference_answer="Sorry to hear that.").reference_answer is not None


def test_a_case_defaults_to_the_normal_category() -> None:
    assert a_case().category is CaseCategory.NORMAL
    assert a_case(category=CaseCategory.EDGE).category is CaseCategory.EDGE


def test_a_dataset_reports_its_size_and_its_source_brief() -> None:
    source = SourceRef(id="brief-1", revision=2)
    dataset = GroundTruthDataset(
        id="ds-1", revision=1, source_brief=source, cases=(a_case(),), created_at=WHEN
    )
    assert len(dataset) == 1
    assert dataset.source_brief == source


def test_a_dataset_rejects_two_cases_with_the_same_id() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        GroundTruthDataset(
            id="ds-1",
            revision=1,
            source_brief=SourceRef(id="brief-1", revision=1),
            cases=(a_case("case-1"), a_case("case-1")),
            created_at=WHEN,
        )


def test_a_dataset_can_find_one_case_by_id() -> None:
    dataset = GroundTruthDataset(
        id="ds-1",
        revision=1,
        source_brief=SourceRef(id="brief-1", revision=1),
        cases=(a_case("case-1"), a_case("case-2")),
        created_at=WHEN,
    )
    assert dataset.case("case-2").id == "case-2"
    with pytest.raises(KeyError):
        dataset.case("case-9")


# --- candidates -----------------------------------------------------------


def test_every_technique_has_a_readable_label_and_description() -> None:
    assert len(PromptTechnique) == 5
    for technique in PromptTechnique:
        assert technique.label
        assert technique.description


def test_a_candidate_records_technique_and_both_source_snapshots() -> None:
    candidate = CandidatePrompt(
        id="cand-1",
        technique=PromptTechnique.FEW_SHOT,
        system_prompt="You are ...",
        revision=1,
        source_brief=SourceRef(id="brief-1", revision=2),
        source_dataset=SourceRef(id="ds-1", revision=1),
        created_at=WHEN,
    )
    assert candidate.technique is PromptTechnique.FEW_SHOT
    assert candidate.source_dataset == SourceRef(id="ds-1", revision=1)
    assert candidate.edited is False


def test_a_candidate_rejects_an_empty_system_prompt() -> None:
    with pytest.raises(ValueError, match="system prompt"):
        CandidatePrompt(
            id="cand-1",
            technique=PromptTechnique.DIRECT,
            system_prompt="   ",
            revision=1,
            source_brief=SourceRef(id="brief-1", revision=1),
            created_at=WHEN,
        )


def test_editing_a_candidate_makes_a_new_revision_marked_as_edited() -> None:
    original = CandidatePrompt(
        id="cand-1",
        technique=PromptTechnique.DIRECT,
        system_prompt="first",
        revision=1,
        source_brief=SourceRef(id="brief-1", revision=1),
        created_at=WHEN,
    )
    edited = original.with_text("second", edited_at=WHEN)
    assert (edited.id, edited.revision, edited.edited) == ("cand-1", 2, True)
    assert edited.system_prompt == "second"
    assert original.revision == 1 and original.system_prompt == "first"


# --- metrics --------------------------------------------------------------


def a_metric(**overrides: object) -> MetricDefinition:
    fields: dict[str, object] = {
        "id": "m-1",
        "name": "Criteria coverage",
        "rubric": "Score how many required criteria the response satisfies.",
        "kind": MetricKind.RUBRIC,
        "weight": 1.0,
    }
    fields.update(overrides)
    return MetricDefinition(**fields)  # type: ignore[arg-type]


def test_a_metric_needs_a_name_and_a_usable_rubric() -> None:
    with pytest.raises(ValueError, match="name"):
        a_metric(name="  ")
    with pytest.raises(ValueError, match="rubric"):
        a_metric(rubric="too short")


def test_a_metric_weight_may_be_zero_but_never_negative() -> None:
    assert a_metric(weight=0.0).weight == 0.0
    with pytest.raises(ValueError, match="negative"):
        a_metric(weight=-0.5)


def test_a_deterministic_metric_names_the_check_it_runs() -> None:
    metric = a_metric(kind=MetricKind.DETERMINISTIC, check="format_compliance")
    assert metric.kind is MetricKind.DETERMINISTIC
    assert metric.check == "format_compliance"


def test_a_deterministic_metric_without_a_check_is_rejected() -> None:
    with pytest.raises(ValueError, match="check"):
        a_metric(kind=MetricKind.DETERMINISTIC, check="")


def test_a_metric_is_enabled_by_default_and_can_be_toggled() -> None:
    assert a_metric().enabled is True
    assert a_metric().with_enabled(False).enabled is False


# --- chat threads ---------------------------------------------------------


def test_a_thread_config_compares_by_value_so_a_change_is_detectable() -> None:
    one = ThreadConfig(candidate=SourceRef(id="c", revision=1), model_id="m", settings=ModelSettings())
    two = ThreadConfig(candidate=SourceRef(id="c", revision=1), model_id="m", settings=ModelSettings())
    three = ThreadConfig(candidate=SourceRef(id="c", revision=2), model_id="m", settings=ModelSettings())
    assert one == two
    assert one != three


def test_appending_to_a_thread_returns_a_new_thread_and_keeps_order() -> None:
    thread = ChatThread(id="t-1", purpose="discovery", created_at=WHEN)
    grown = thread.appended(ChatMessage("user", "hi", WHEN)).appended(
        ChatMessage("assistant", "hello", WHEN)
    )
    assert [m.content for m in grown.messages] == ["hi", "hello"]
    assert thread.messages == ()


def test_a_thread_renders_provider_messages_without_timestamps() -> None:
    thread = ChatThread(id="t-1", purpose="discovery", created_at=WHEN).appended(
        ChatMessage("user", "hi", WHEN)
    )
    assert thread.as_provider_messages() == [{"role": "user", "content": "hi"}]


def test_a_chat_message_rejects_an_unknown_role() -> None:
    with pytest.raises(ValueError, match="role"):
        ChatMessage("narrator", "hi", WHEN)


# --- executions -----------------------------------------------------------


def test_an_execution_carries_every_identifier_needed_for_attribution() -> None:
    record = ExecutionRecord(
        id="run-1",
        thread_id="t-1",
        candidate=SourceRef(id="cand-1", revision=2),
        source_brief=SourceRef(id="brief-1", revision=1),
        case_id="case-1",
        model_id="openai/gpt-4o-mini",
        settings=ModelSettings(temperature=0.2),
        user_message="My order never arrived.",
        response="I am sorry about the delay.",
        created_at=WHEN,
    )
    assert record.is_attributable
    assert record.case_id == "case-1"
    assert record.settings.temperature == 0.2


def test_an_execution_without_a_test_case_is_not_attributable() -> None:
    record = ExecutionRecord(
        id="run-1",
        thread_id="t-1",
        candidate=SourceRef(id="cand-1", revision=2),
        source_brief=SourceRef(id="brief-1", revision=1),
        case_id=None,
        model_id="m",
        settings=ModelSettings(),
        user_message="free text",
        response="reply",
        created_at=WHEN,
    )
    assert not record.is_attributable
    assert "test case" in record.missing_evidence()[0]


def test_an_execution_with_an_empty_response_is_not_attributable() -> None:
    record = ExecutionRecord(
        id="run-1",
        thread_id="t-1",
        candidate=SourceRef(id="cand-1", revision=2),
        source_brief=SourceRef(id="brief-1", revision=1),
        case_id="case-1",
        model_id="m",
        settings=ModelSettings(),
        user_message="hi",
        response="   ",
        created_at=WHEN,
    )
    assert not record.is_attributable


# --- evaluation results ---------------------------------------------------


def test_a_metric_score_must_be_normalized() -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        MetricScore(metric_id="m", metric_name="n", weight=1.0, score=1.5)


def test_a_failed_metric_score_holds_no_number_and_says_why() -> None:
    score = MetricScore.failed_with(
        metric_id="m", metric_name="n", weight=1.0, failure="judge returned no text"
    )
    assert score.score is None
    assert score.failed
    assert "no text" in score.failure


def test_a_not_applicable_metric_score_is_neither_failed_nor_scored() -> None:
    score = MetricScore.not_applicable(
        metric_id="m", metric_name="n", weight=1.0, reason="case has no reference answer"
    )
    assert score.score is None
    assert not score.failed
    assert not score.applicable


def test_an_evaluation_run_is_immutable_once_built() -> None:
    run = EvaluationRun(
        id="eval-1",
        created_at=WHEN,
        candidate=SourceRef(id="cand-1", revision=1),
        source_brief=SourceRef(id="brief-1", revision=1),
        source_dataset=SourceRef(id="ds-1", revision=1),
        metrics=(a_metric(),),
        judge_backend="codex",
        judge_model="gpt-5.6-luna",
        cases=(CaseEvaluation(execution_id="run-1", case_id="case-1", scores=(), grade=None),),
        overall=Grade(value=None),
    )
    with pytest.raises(Exception):
        run.id = "eval-2"  # type: ignore[misc]


def test_a_grade_reports_a_percentage_and_a_letter_band() -> None:
    assert Grade(value=0.9).letter == "A"
    assert Grade(value=0.9).percentage == 90.0
    assert Grade(value=0.8).letter == "B"
    assert Grade(value=0.7).letter == "C"
    assert Grade(value=0.6).letter == "D"
    assert Grade(value=0.2).letter == "F"


def test_a_grade_with_no_value_reports_no_letter() -> None:
    grade = Grade(value=None)
    assert grade.letter == "-"
    assert grade.percentage is None
