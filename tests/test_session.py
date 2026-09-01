"""The session: a sequence of steps, and what a change to one costs the others."""

from dataclasses import replace
from datetime import UTC, datetime

import pytest

from prompt_workbench.core.session import Session
from prompt_workbench.core.sweep import SweepCell
from prompt_workbench.models.case import EvalCase
from prompt_workbench.models.identifiers import sequential_ids
from prompt_workbench.models.model_settings import ModelSettings
from prompt_workbench.models.usage import TokenUsage
from prompt_workbench.models.variant import PromptVariant
from prompt_workbench.services import model_registry, task_catalog

WHEN = datetime(2026, 3, 1, tzinfo=UTC)

SAMPLE = {"data": [{"id": "cheap/model", "pricing": {"prompt": "0.0000001", "completion": "0.0000004"}}]}


def a_session() -> Session:
    return Session(
        new_id=sequential_ids(),
        clock=lambda: WHEN,
        registry=model_registry.ModelRegistry(fetch=lambda: SAMPLE),
    )


def a_variant(variant_id: str = "v1", approach: str = "enumerate") -> PromptVariant:
    return PromptVariant(
        id=variant_id, approach_key=approach, approach_label="Strict enumeration",
        task_type_key="classification", system_prompt="You are a classifier.",
        revision=1, created_at=WHEN,
    )


def prepared() -> Session:
    session = a_session()
    session.describe("Sort tickets into six queues")
    session.set_task_type(task_catalog.get("classification"))
    session.set_cases((EvalCase(id="c1", input="charged twice", expected_output="billing"),))
    session.set_variants((a_variant(),))
    return session


# --- describing -----------------------------------------------------------


def test_describing_a_case_proposes_a_task_type() -> None:
    session = a_session()
    proposed = session.describe("I need to classify support tickets into queues")
    assert proposed is not None and proposed.key in {"classification", "routing"}
    assert session.description


def test_an_unrecognisable_description_proposes_nothing() -> None:
    assert a_session().describe("asdf qwerty") is None


def test_adopting_a_task_type_brings_its_metrics_and_settings() -> None:
    session = a_session()
    session.set_task_type(task_catalog.get("classification"))
    assert session.metrics, "a type must arrive with its suggested metrics"
    assert session.settings.temperature == 0.0


# --- what a change costs --------------------------------------------------


def test_changing_the_task_type_discards_what_the_old_one_shaped() -> None:
    """A classifier's prompt judged by a summarizer's metrics would otherwise
    look like a measurement."""
    session = prepared()
    assert session.variants
    session.set_task_type(task_catalog.get("summarization"))
    assert session.variants == ()
    assert {c.key for c in session.metrics} != set()
    assert session.settings.temperature != 0.0


def test_reselecting_the_same_task_type_changes_nothing() -> None:
    session = prepared()
    session.metrics = session.metrics[:1]
    kept = session.metrics
    session.set_task_type(task_catalog.get("classification"))
    assert session.variants, "an idempotent reselect must not wipe the work"
    assert session.metrics == kept


def test_the_cases_survive_a_task_type_change() -> None:
    """They describe the user's problem, not the workbench's guess about it."""
    session = prepared()
    session.set_task_type(task_catalog.get("summarization"))
    assert len(session.cases) == 1


# --- stages ---------------------------------------------------------------


def test_the_stage_advances_as_each_step_is_completed() -> None:
    session = a_session()
    assert session.stage == "describe"
    session.describe("classify tickets")
    session.set_task_type(task_catalog.get("classification"))
    assert session.stage == "cases"
    session.set_cases((EvalCase(id="c1", input="x"),))
    assert session.stage == "variants"
    session.set_variants((a_variant(),))
    assert session.stage == "sweep"


def test_is_past_reports_how_far_the_session_has_got() -> None:
    session = prepared()
    assert session.is_past("cases")
    assert not a_session().is_past("cases")


# --- metrics --------------------------------------------------------------


def test_only_judged_metrics_count_toward_the_cost_driver() -> None:
    session = prepared()
    session.metrics = ()
    session.add_metric("exact_match")
    assert session.judged_metric_count == 0
    session.add_metric("faithfulness")
    assert session.judged_metric_count == 1


def test_a_metric_can_be_added_rethresholded_and_removed() -> None:
    from dataclasses import replace

    session = prepared()
    session.metrics = ()
    session.add_metric("faithfulness")
    session.add_metric("faithfulness")  # twice is once
    assert len(session.metrics) == 1
    session.update_metric("faithfulness", replace(session.metric("faithfulness"), threshold=0.95))
    assert session.metric("faithfulness").threshold == 0.95
    session.remove_metric("faithfulness")
    with pytest.raises(KeyError):
        session.metric("faithfulness")


def test_a_disabled_metric_is_not_counted_as_enabled() -> None:
    from dataclasses import replace

    session = prepared()
    session.metrics = ()
    session.add_metric("faithfulness")
    session.update_metric("faithfulness", replace(session.metric("faithfulness"), enabled=False))
    assert session.enabled_metrics == ()


# --- variants -------------------------------------------------------------


def test_editing_a_variant_bumps_its_revision_and_leaves_the_rest() -> None:
    session = prepared()
    session.set_variants((a_variant("v1"), a_variant("v2", approach="schema")))
    session.edit_variant("v1", "rewritten by hand")
    assert session.variant("v1").revision == 2
    assert session.variant("v1").edited
    assert session.variant("v2").revision == 1
    assert session.has_edited_variants


# --- writing things by hand -----------------------------------------------


def test_a_case_can_be_added_by_hand() -> None:
    """Generation needs a provider key. Writing one down does not, and a
    workbench you cannot use until you have paid for a key is not usable."""
    session = a_session()
    session.describe("classify tickets")
    session.set_task_type(task_catalog.get("classification"))
    case = session.add_case(input="My card was charged twice", expected_output="billing")
    assert session.cases == (case,)
    assert case.id


def test_a_hand_written_case_needs_an_input() -> None:
    session = prepared()
    with pytest.raises(ValueError, match="input"):
        session.add_case(input="   ")


def test_your_existing_prompt_can_be_added_as_a_variant() -> None:
    """The obvious thing to want from a prompt workbench: bring the prompt you
    already have and find out whether anything beats it."""
    session = prepared()
    session.set_variants(())
    variant = session.add_variant("You are a ticket classifier. Reply with the queue name.")
    assert session.variants == (variant,)
    assert variant.approach_key == "your_own"
    assert "Your own" in variant.approach_label


def test_a_hand_written_variant_joins_the_generated_ones() -> None:
    """It becomes the baseline the sweep compares against."""
    session = prepared()
    assert len(session.variants) == 1
    session.add_variant("my existing prompt")
    assert len(session.variants) == 2
    assert {v.approach_key for v in session.variants} == {"enumerate", "your_own"}


def test_an_empty_hand_written_variant_is_refused() -> None:
    session = prepared()
    with pytest.raises(ValueError, match="empty"):
        session.add_variant("   ")


def test_generating_variants_does_not_discard_your_own() -> None:
    """Your prompt is the thing being compared against; losing it to a
    regeneration would defeat the comparison."""
    session = prepared()
    session.set_variants(())
    mine = session.add_variant("my existing prompt")
    session.set_variants((a_variant("v9"),), keep_hand_written=True)
    assert mine in session.variants
    assert len(session.variants) == 2


# --- sweep results stay tied to the work that produced them ---------------


def a_cell(variant_key: str = "v1", model_id: str = "cheap/model") -> SweepCell:
    return SweepCell(
        variant_key=variant_key,
        model_id=model_id,
        settings=ModelSettings(),
        score=0.9,
        usage=TokenUsage(tokens_in=100, tokens_out=20),
        met_every_threshold=True,
    )


def swept() -> Session:
    session = prepared()
    session.sweep_results = (a_cell(),)
    return session


def test_a_sweep_keeps_its_results_until_something_changes() -> None:
    session = swept()
    assert session.sweep_results


@pytest.mark.parametrize(
    "change",
    [
        pytest.param(lambda s: s.add_case(input="refund never arrived"), id="a case is added"),
        pytest.param(lambda s: s.remove_case("c1"), id="a case is removed"),
        pytest.param(
            lambda s: s.replace_case("c1", EvalCase(id="c1", input="edited")),
            id="a case is edited",
        ),
        pytest.param(lambda s: s.add_variant("You are a classifier."), id="a variant is added"),
        pytest.param(lambda s: s.edit_variant("v1", "Rewritten."), id="a variant is edited"),
        pytest.param(lambda s: s.set_variants(()), id="the variants are regenerated"),
        pytest.param(lambda s: s.remove_metric(s.metrics[0].key), id="a metric is dropped"),
        pytest.param(lambda s: s.add_metric("bias"), id="a metric is added"),
        pytest.param(
            lambda s: s.set_task_type(task_catalog.get("summarization")),
            id="the kind of job changes",
        ),
        pytest.param(
            lambda s: setattr(s, "settings", ModelSettings(temperature=1.5)),
            id="the settings change",
        ),
    ],
)
def test_changing_the_work_discards_the_results_it_produced(change) -> None:  # type: ignore[no-untyped-def]
    """A score belongs to the exact prompt, cases, metrics and settings that
    produced it. Leaving it on screen after any of those move invites reading a
    number as if it described the current state."""
    session = swept()
    change(session)
    assert session.sweep_results == ()


def test_rewriting_the_settings_with_the_same_values_keeps_the_results() -> None:
    """The sidebar reassigns the settings on every rerun, so an unchanged
    assignment must not count as a change."""
    session = swept()
    session.settings = replace(session.settings)
    assert session.sweep_results
