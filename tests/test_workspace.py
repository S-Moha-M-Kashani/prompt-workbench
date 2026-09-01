"""The session workspace: one place that holds the artifacts and their state."""

from datetime import UTC, datetime

import pytest

from prompt_workbench.core.workspace import Workspace
from prompt_workbench.models import (
    GroundTruthCase,
    PromptBrief,
    PromptTechnique,
    sequential_ids,
)

WHEN = datetime(2026, 3, 1, tzinfo=UTC)


def a_workspace() -> Workspace:
    return Workspace(new_id=sequential_ids(), clock=lambda: WHEN)


def a_case(case_id: str = "case-1", message: str = "hi") -> GroundTruthCase:
    return GroundTruthCase(
        id=case_id, test_message=message, required_criteria=("is specific",)
    )


# --- the brief ------------------------------------------------------------


def test_a_new_workspace_has_a_draft_brief_and_no_confirmed_one() -> None:
    workspace = a_workspace()
    assert workspace.draft_brief == PromptBrief()
    assert workspace.confirmed_brief is None
    assert not workspace.can_generate


def test_confirming_records_a_snapshot_at_revision_one() -> None:
    workspace = a_workspace()
    workspace.update_draft(PromptBrief(purpose="Summarize tickets"))
    snapshot = workspace.confirm_brief()
    assert snapshot.revision == 1
    assert workspace.confirmed_brief == snapshot
    assert workspace.can_generate


def test_confirming_again_after_an_edit_makes_revision_two_with_the_same_id() -> None:
    workspace = a_workspace()
    workspace.update_draft(PromptBrief(purpose="first"))
    first = workspace.confirm_brief()
    workspace.update_draft(PromptBrief(purpose="second"))
    second = workspace.confirm_brief()
    assert second.id == first.id
    assert second.revision == 2


def test_an_empty_brief_cannot_be_confirmed() -> None:
    with pytest.raises(ValueError, match="empty"):
        a_workspace().confirm_brief()


def test_editing_the_draft_after_confirming_marks_the_brief_as_dirty() -> None:
    workspace = a_workspace()
    workspace.update_draft(PromptBrief(purpose="first"))
    workspace.confirm_brief()
    assert not workspace.brief_is_dirty
    workspace.update_draft(PromptBrief(purpose="first", audience="new"))
    assert workspace.brief_is_dirty


# --- ground truth ---------------------------------------------------------


def confirmed_workspace() -> Workspace:
    workspace = a_workspace()
    workspace.update_draft(PromptBrief(purpose="Summarize tickets"))
    workspace.confirm_brief()
    return workspace


def test_adding_a_case_by_hand_creates_a_dataset_pinned_to_the_brief() -> None:
    workspace = confirmed_workspace()
    workspace.add_case(a_case())
    dataset = workspace.dataset
    assert dataset is not None
    assert len(dataset) == 1
    assert dataset.source_brief == workspace.confirmed_brief.ref  # type: ignore[union-attr]


def test_editing_a_case_makes_a_new_dataset_revision() -> None:
    workspace = confirmed_workspace()
    workspace.add_case(a_case(message="before"))
    first_revision = workspace.dataset.revision  # type: ignore[union-attr]
    stored = workspace.dataset.cases[0]  # type: ignore[union-attr]
    workspace.replace_case(stored.id, a_case(stored.id, message="after"))
    assert workspace.dataset.revision == first_revision + 1  # type: ignore[union-attr]
    assert workspace.dataset.cases[0].test_message == "after"  # type: ignore[union-attr]


def test_duplicating_a_case_gives_the_copy_its_own_id() -> None:
    workspace = confirmed_workspace()
    workspace.add_case(a_case())
    original = workspace.dataset.cases[0]  # type: ignore[union-attr]
    workspace.duplicate_case(original.id)
    cases = workspace.dataset.cases  # type: ignore[union-attr]
    assert len(cases) == 2
    assert cases[0].id != cases[1].id
    assert cases[0].test_message == cases[1].test_message


def test_removing_a_case_leaves_the_others_alone() -> None:
    workspace = confirmed_workspace()
    workspace.add_case(a_case(message="keep"))
    workspace.add_case(a_case(message="drop"))
    to_drop = workspace.dataset.cases[1]  # type: ignore[union-attr]
    workspace.remove_case(to_drop.id)
    assert [c.test_message for c in workspace.dataset] == ["keep"]  # type: ignore[union-attr]


# --- candidates -----------------------------------------------------------


def test_editing_a_candidate_never_touches_the_others() -> None:
    workspace = confirmed_workspace()
    workspace.add_case(a_case())
    workspace.set_candidates(_two_candidates(workspace))
    first, second = workspace.candidates
    workspace.edit_candidate(first.id, "rewritten by hand")
    edited, untouched = workspace.candidates
    assert edited.system_prompt == "rewritten by hand"
    assert edited.edited and edited.revision == 2
    assert untouched == second


def _two_candidates(workspace: Workspace):  # type: ignore[no-untyped-def]
    from prompt_workbench.models import CandidatePrompt

    brief = workspace.confirmed_brief
    assert brief is not None
    return tuple(
        CandidatePrompt(
            id=f"cand-{i}",
            technique=technique,
            system_prompt=f"prompt {i}",
            revision=1,
            source_brief=brief.ref,
            created_at=WHEN,
        )
        for i, technique in enumerate((PromptTechnique.DIRECT, PromptTechnique.ROLE_BASED), 1)
    )


def test_generating_candidates_again_replaces_the_set_only_when_told_to() -> None:
    workspace = confirmed_workspace()
    workspace.set_candidates(_two_candidates(workspace))
    workspace.edit_candidate("cand-1", "hand written")
    assert workspace.has_edited_candidates
    with pytest.raises(ValueError, match="edited"):
        workspace.set_candidates(_two_candidates(workspace))
    workspace.set_candidates(_two_candidates(workspace), overwrite=True)
    assert not workspace.has_edited_candidates


# --- staleness ------------------------------------------------------------


def test_artifacts_generated_before_a_brief_edit_are_reported_as_stale() -> None:
    workspace = confirmed_workspace()
    workspace.add_case(a_case())
    workspace.set_candidates(_two_candidates(workspace))
    assert workspace.stale_artifacts() == ()
    workspace.update_draft(PromptBrief(purpose="something quite different"))
    workspace.confirm_brief()
    stale = workspace.stale_artifacts()
    assert any("ground truth" in s.lower() for s in stale)
    assert any("candidate" in s.lower() for s in stale)


# --- metrics --------------------------------------------------------------


def test_a_new_workspace_starts_with_the_built_in_metrics_enabled() -> None:
    workspace = a_workspace()
    assert workspace.metrics
    assert all(metric.builtin and metric.enabled for metric in workspace.metrics)


def test_a_custom_metric_can_be_added_edited_and_removed() -> None:
    workspace = a_workspace()
    added = workspace.add_metric(name="Warmth", rubric="Score how warm the reply sounds overall.")
    assert not added.builtin
    workspace.set_metric_weight(added.id, 2.0)
    assert workspace.metric(added.id).weight == 2.0
    workspace.remove_metric(added.id)
    with pytest.raises(KeyError):
        workspace.metric(added.id)


def test_a_built_in_metric_can_be_disabled_but_not_removed() -> None:
    workspace = a_workspace()
    builtin = workspace.metrics[0]
    workspace.set_metric_enabled(builtin.id, False)
    assert not workspace.metric(builtin.id).enabled
    with pytest.raises(ValueError, match="built-in"):
        workspace.remove_metric(builtin.id)


# --- executions and runs --------------------------------------------------


def test_recorded_responses_and_evaluations_accumulate_in_order() -> None:
    workspace = a_workspace()
    assert workspace.executions == ()
    assert workspace.evaluations == ()
    assert workspace.latest_evaluation is None
