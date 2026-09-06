"""Task types: the one choice that decides everything downstream."""

import pytest

from prompt_workbench.models.task_type import FineTuneVerdict, TaskType
from prompt_workbench.services import task_catalog


def test_the_catalogue_ships_at_least_nine_types() -> None:
    """Nine, not ten: a tool call is a property of a round, so it stopped being
    a kind of job. The docstring of `task_catalog` says why."""
    assert len(task_catalog.all_task_types()) >= 9


def test_every_type_is_fully_specified() -> None:
    for task in task_catalog.all_task_types():
        assert task.label, task.key
        assert task.description, task.key
        assert task.variants, f"{task.key} offers no prompt approaches"
        assert task.metric_keys, f"{task.key} suggests no metrics"
        assert task.settings_note, task.key


def test_each_type_offers_several_distinct_prompt_approaches() -> None:
    """The old design applied five fixed techniques to every job. Approaches are
    now chosen per type, so classification and drafting share none of them."""
    classification = task_catalog.get("classification")
    drafting = task_catalog.get("generation")
    assert len(classification.variants) >= 3
    shared = {v.key for v in classification.variants} & {v.key for v in drafting.variants}
    assert not shared, f"a variant should not suit both jobs: {shared}"


def test_deterministic_tasks_ask_for_zero_temperature() -> None:
    for key in ("classification", "extraction", "routing"):
        assert task_catalog.get(key).suggested_settings.temperature == 0.0


def test_open_ended_generation_allows_sampling() -> None:
    temperature = task_catalog.get("generation").suggested_settings.temperature
    assert temperature is not None and temperature > 0.3


def test_high_volume_stable_output_tasks_recommend_a_fine_tune() -> None:
    """The cheapest thing that works may not be a prompt at all."""
    for key in ("classification", "extraction", "routing"):
        task = task_catalog.get(key)
        assert task.fine_tune is FineTuneVerdict.LIKELY, key
        assert task.fine_tune_note, key


def test_judging_refuses_to_recommend_the_cheapest_models() -> None:
    """A cheap judge is a noisy judge, and noise in the judge is indistinguishable
    from a worse prompt."""
    judging = task_catalog.get("judging")
    assert judging.needs_strong_model
    assert judging.fine_tune is not FineTuneVerdict.LIKELY


def test_a_cheap_task_does_not_demand_a_strong_model() -> None:
    assert not task_catalog.get("classification").needs_strong_model


def test_types_are_proposed_from_a_free_text_case() -> None:
    proposed = task_catalog.propose(
        "I need to sort incoming support tickets into one of six queues"
    )
    assert proposed is not None
    assert proposed.key in {"classification", "routing"}


def test_a_case_about_pulling_fields_out_of_documents_proposes_extraction() -> None:
    proposed = task_catalog.propose(
        "Extract the invoice number, total and due date from a PDF's text"
    )
    assert proposed is not None and proposed.key == "extraction"


def test_a_case_about_answering_from_documents_proposes_grounded_qa() -> None:
    proposed = task_catalog.propose(
        "Answer customer questions using only our retrieved help-centre articles"
    )
    assert proposed is not None and proposed.key == "grounded_qa"


def test_an_unrecognisable_case_proposes_nothing_rather_than_guessing() -> None:
    """A wrong type silently picks the wrong metrics, so no proposal beats a
    coin flip."""
    assert task_catalog.propose("asdf qwerty zxcv") is None


def test_an_empty_description_proposes_nothing() -> None:
    assert task_catalog.propose("   ") is None


def test_every_suggested_metric_key_is_one_the_metric_layer_knows() -> None:
    from prompt_workbench.services import deepeval_metrics

    known = set(deepeval_metrics.METRIC_SPECS)
    for task in task_catalog.all_task_types():
        unknown = set(task.metric_keys) - known
        assert not unknown, f"{task.key} suggests unknown metrics: {unknown}"


def test_a_task_type_is_immutable() -> None:
    task = task_catalog.get("classification")
    with pytest.raises(Exception):
        task.label = "something else"  # type: ignore[misc]


def test_an_unknown_key_raises() -> None:
    with pytest.raises(KeyError):
        task_catalog.get("no_such_task")


# --- nine kinds of job, each carrying a starting kit ----------------------


def test_a_tool_call_is_a_property_of_a_round_not_a_kind_of_job() -> None:
    """`agentic` was the odd one out only because it sent tools, and tools are
    now a per-round switch. Nine of the ten were mechanically the same call."""
    keys = {task.key for task in task_catalog.all_task_types()}
    assert "agentic" not in keys
    assert len(keys) == 9


def test_nothing_still_points_at_the_removed_kind_of_job() -> None:
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "src" / "prompt_workbench"
    offenders = [
        path.relative_to(root)
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix in {".py", ".yaml", ".json"}
        and "task_type: agentic" in path.read_text()
    ]
    assert offenders == []


def test_every_kind_of_job_names_the_preset_that_fills_it_in() -> None:
    for task in task_catalog.all_task_types():
        assert task.preset_key, f"{task.key} carries no preset"
