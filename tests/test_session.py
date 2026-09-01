"""One session: a selected use case, the prompt being worked on, what it did."""

from datetime import UTC, datetime

import pytest

from prompt_workbench.core.session import ENGINEER_MODE, MODES, USER_MODE, Session
from prompt_workbench.models import PromptRun, sequential_ids
from prompt_workbench.services import use_case_catalog

WHEN = datetime(2026, 3, 1, tzinfo=UTC)


def a_session() -> Session:
    return Session(new_id=sequential_ids(), clock=lambda: WHEN)


def a_run(use_case_key: str = "grounded_briefing") -> PromptRun:
    return PromptRun(
        id="run-1",
        use_case_key=use_case_key,
        prompt_revision=1,
        system_prompt="p",
        model_id="m",
        user_message="hi",
        response="there",
        created_at=WHEN,
    )


def test_a_new_session_has_nothing_selected_and_offers_both_modes() -> None:
    session = a_session()
    assert session.use_case is None
    assert not session.is_ready
    assert session.mode == ENGINEER_MODE
    assert set(MODES) == {ENGINEER_MODE, USER_MODE}


def test_selecting_a_use_case_loads_its_starting_prompt() -> None:
    session = a_session()
    case = use_case_catalog.get("injection_resistance")
    session.select(case)
    assert session.is_ready
    assert session.prompt_under_test == case.system_prompt
    assert session.prompt_revision == 1
    assert not session.prompt_is_modified


def test_editing_the_prompt_makes_a_new_revision() -> None:
    session = a_session()
    session.select(use_case_catalog.get("grounded_briefing"))
    assert session.update_prompt("a different prompt entirely {history}")
    assert session.prompt_revision == 2
    assert session.prompt_is_modified


def test_an_unchanged_edit_does_not_bump_the_revision() -> None:
    session = a_session()
    case = use_case_catalog.get("grounded_briefing")
    session.select(case)
    assert not session.update_prompt(case.system_prompt)
    assert session.prompt_revision == 1


def test_an_empty_prompt_is_refused() -> None:
    session = a_session()
    session.select(use_case_catalog.get("grounded_briefing"))
    with pytest.raises(ValueError, match="empty"):
        session.update_prompt("   ")


def test_the_original_prompt_can_be_restored() -> None:
    session = a_session()
    case = use_case_catalog.get("grounded_briefing")
    session.select(case)
    session.update_prompt("something else {history}")
    session.reset_prompt()
    assert session.prompt_under_test == case.system_prompt
    assert not session.prompt_is_modified
    # Restoring is itself a revision, so history is never rewritten.
    assert session.prompt_revision == 3


def test_switching_use_case_clears_the_runs_of_the_previous_one() -> None:
    """A response scored against one situation's criteria must not survive into
    another, where it was never measured."""
    session = a_session()
    session.select(use_case_catalog.get("grounded_briefing"))
    session.record_run(a_run())
    assert session.runs

    session.select(use_case_catalog.get("scope_guardrail"))
    assert session.runs == ()
    assert session.evaluations == ()
    assert session.prompt_revision == 1


def test_runs_and_evaluations_accumulate_in_order() -> None:
    session = a_session()
    session.select(use_case_catalog.get("grounded_briefing"))
    assert session.latest_run is None
    session.record_run(a_run())
    assert session.latest_run is not None
    assert session.latest_evaluation is None


def test_only_runs_with_a_response_are_offered_for_scoring() -> None:
    session = a_session()
    session.select(use_case_catalog.get("grounded_briefing"))
    session.record_run(a_run())
    empty = PromptRun(
        id="run-2",
        use_case_key="grounded_briefing",
        prompt_revision=1,
        system_prompt="p",
        model_id="m",
        user_message="hi",
        response="",
        created_at=WHEN,
    )
    session.record_run(empty)
    assert [r.id for r in session.scoreable_runs()] == ["run-1"]


def test_a_session_starts_with_the_built_in_metrics_enabled() -> None:
    session = a_session()
    assert session.metrics
    assert all(m.builtin and m.enabled for m in session.metrics)
