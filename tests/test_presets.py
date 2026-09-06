"""Presets: a kind of job as an editable starting kit, not as the engine."""

import pytest

from prompt_workbench.core import presets
from prompt_workbench.services import deepeval_metrics, task_catalog


@pytest.mark.parametrize(
    "task", task_catalog.all_task_types(), ids=lambda t: t.key
)
def test_every_kind_of_job_has_a_loadable_kit(task) -> None:
    kit = presets.load(task.preset_key)
    assert kit.system_prompt.strip()
    assert kit.user_prompt.strip()
    assert kit.temperature is not None
    assert kit.approach_labels, "a kit names the approaches it offers"


@pytest.mark.parametrize(
    "task", task_catalog.all_task_types(), ids=lambda t: t.key
)
def test_a_kits_metrics_are_real_metrics_with_thresholds(task) -> None:
    kit = presets.load(task.preset_key)
    assert kit.metrics, f"{task.key} suggests no metric"
    for key, threshold in kit.metrics.items():
        assert key in deepeval_metrics.METRIC_SPECS, key
        assert 0.0 < threshold <= 1.0, (key, threshold)


def test_a_kit_that_wants_tools_defines_them_by_name_and_description() -> None:
    with_tools = [
        presets.load(t.preset_key) for t in task_catalog.all_task_types()
    ]
    defined = [tool for kit in with_tools for tool in kit.tools]
    assert defined, "at least one kind of job starts with a tool set"
    for tool in defined:
        assert tool.name.strip() and tool.description.strip()


def test_an_unknown_kit_is_refused_by_name() -> None:
    with pytest.raises(KeyError, match="nonsense"):
        presets.load("nonsense")


def test_a_kit_is_a_starting_point_and_says_so() -> None:
    kit = presets.load(task_catalog.get("classification").preset_key)
    assert kit.note.strip(), "a kit explains that it is a starting point"
