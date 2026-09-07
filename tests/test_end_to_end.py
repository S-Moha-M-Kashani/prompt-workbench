"""The whole page, walked, with every model call mocked.

These are the tests the unit suite cannot be: each one starts at a blank page,
types a case into it, and stops when the sweep has named a configuration. What
they cover is the wiring — that the description reaches the task type, that the
task type reaches the metrics, that the variant's prompt and the case's input
are the two halves of every request, and that a result names the framework that
produced it.

No model is ever called. One scripted adapter stands in per framework and
records what it was asked for, which is also how these tests assert on the
request rather than only on the screen.

The metric is deliberately ``exact_match``: it is the only one that calls no
model at all, so a walk that ends in a green result ends there because the
plumbing worked, not because a judge was in a good mood.
"""

from __future__ import annotations

from typing import Any

import pytest

from prompt_workbench.services import deepeval_metrics
from tests_support_app import ScriptedRunner, rendered, start

pytestmark = pytest.mark.skipif(
    not deepeval_metrics.is_available(),
    reason="the sweep is gated on the deepeval extra",
)

DESCRIPTION = "Sort incoming support tickets into one of six queues."
CASE_INPUT = "I was charged twice this month."
EXPECTED = "billing"
VARIANT = "You are a classifier. Answer with the queue name only."

# The one metric that reaches no model, so a walk is deterministic end to end.
KEPT_METRIC = "exact_match"


# --- driving the page -----------------------------------------------------


def _widget(at: Any, group: str, key: str) -> Any:
    for element in getattr(at, group):
        if element.key == key:
            return element
    raise AssertionError(f"no {group} with key {key!r} on the page")


def _button(at: Any, label: str) -> Any:
    for button in at.button:
        if button.label == label:
            return button
    raise AssertionError(f"no button labelled {label!r} on the page")


def _only_metric(at: Any, keep: str) -> Any:
    """Leave one metric enabled, so the score means one thing."""
    for box in [b for b in at.checkbox if b.key and b.key.startswith("me_")]:
        wanted = box.key == f"me_{keep}"
        if box.value != wanted:
            at = _widget(at, "checkbox", box.key).set_value(wanted).run()
    return at


def walk_to_the_sweep(
    monkeypatch,
    *,
    expected: str = EXPECTED,
    keep_metric: str = KEPT_METRIC,
    **runners: ScriptedRunner,
) -> Any:
    """From a blank page to one case, one variant and one metric."""
    at = start(monkeypatch, **runners)
    at = _widget(at, "text_area", "case_description").set_value(DESCRIPTION).run()
    at = _widget(at, "selectbox", "task_type_pick").set_value("classification").run()

    _widget(at, "text_area", "new_case_input").set_value(CASE_INPUT)
    _widget(at, "text_input", "new_case_expected").set_value(expected)
    at = at.run()
    at = _button(at, "Add this case").click().run()

    _widget(at, "text_area", "new_variant_prompt").set_value(VARIANT)
    at = at.run()
    at = _button(at, "Add this prompt").click().run()

    return _only_metric(at, keep_metric)


def run_the_sweep(at: Any) -> Any:
    button = _button(at, "Run the sweep")
    assert not button.disabled, f"the sweep was refused: {button.help}"
    return button.click().run()


# --- a case that passes ---------------------------------------------------


def test_a_described_case_walks_all_the_way_to_a_named_configuration(monkeypatch):
    """The whole point of the page, in one test: describe a job, and be told
    which framework, prompt and model to ship."""
    runner = ScriptedRunner("openai", answer=EXPECTED)
    at = run_the_sweep(walk_to_the_sweep(monkeypatch, openai=runner))

    assert not at.exception
    text = rendered(at)
    assert "Cheapest configuration that passed" in text
    assert "via `openai`" in text


def test_the_configuration_is_shown_as_code_to_carry_into_another_project(monkeypatch):
    """Nothing is written to disk, so the deliverable is code on screen."""
    at = run_the_sweep(walk_to_the_sweep(monkeypatch, openai=ScriptedRunner("openai")))
    code = " ".join(block.value for block in at.code)
    assert "# framework: openai" in code
    assert "ExactMatchMetric" in code


# --- what the round actually sent -----------------------------------------


def test_every_request_pairs_the_variants_prompt_with_the_cases_input(monkeypatch):
    """The sweep substitutes exactly two things per cell. If it substituted the
    round's own prompts instead, every cell would measure the same call."""
    runner = ScriptedRunner("openai")
    run_the_sweep(walk_to_the_sweep(monkeypatch, openai=runner))

    assert runner.requests
    for request in runner.requests:
        assert request.system_prompt == VARIANT
        assert request.user_prompt == CASE_INPUT


def test_one_request_is_made_per_framework_variant_model_and_case(monkeypatch):
    """The count is the sweep's contract, and the number the estimate is built
    on. A cell that quietly ran twice would double a bill nobody approved."""
    runner = ScriptedRunner("openai")
    at = run_the_sweep(walk_to_the_sweep(monkeypatch, openai=runner))

    models = next(box for box in at.multiselect if box.label == "Models to try")
    assert len(runner.requests) == len(models.value)  # one variant, one case


def test_a_tool_surface_is_carried_into_every_call_the_sweep_makes(monkeypatch):
    """Tools are a property of the round, so they belong to every cell of the
    sweep — not only to the single round run in the lab."""
    at = walk_to_the_sweep(monkeypatch, openai=(runner := ScriptedRunner("openai")))
    at = _widget(at, "checkbox", "round_tools_on").set_value(True).run()
    _widget(at, "text_input", "tool_name_0").set_value("lookup_customer")
    at = at.run()

    assert "2 model calls" in rendered(at) or "2 model calls rather than one" in rendered(at)
    run_the_sweep(at)
    assert runner.requests
    for request in runner.requests:
        assert request.tool_names == ("lookup_customer",)


# --- a case that does not pass --------------------------------------------


def test_a_wrong_answer_clears_nothing_and_the_page_says_so(monkeypatch):
    """A sweep that found no answer must say so, not pick a least-bad winner."""
    runner = ScriptedRunner("openai", answer="shipping")
    at = run_the_sweep(walk_to_the_sweep(monkeypatch, openai=runner))

    text = rendered(at)
    assert "Nothing cleared every threshold" in text
    assert "Cheapest configuration that passed" not in text


def test_one_model_failing_does_not_cost_the_user_the_rest_of_the_sweep(monkeypatch):
    """The other cells have already been paid for by the time one breaks."""
    runner = ScriptedRunner("openai")
    at = walk_to_the_sweep(monkeypatch, openai=runner)
    models = next(box for box in at.multiselect if box.label == "Models to try")
    assert len(models.value) > 1, "this test needs a second model to survive the first"
    runner.fail_on(models.value[0])

    at = run_the_sweep(at)
    text = rendered(at)
    assert "the provider refused" in text
    assert "Cheapest configuration that passed" in text


# --- results describe something that still exists -------------------------


def test_changing_a_metric_after_a_sweep_discards_the_result(monkeypatch):
    """A score is a statement about one exact configuration. Leaving it on
    screen after the configuration moved is how a stale number gets shipped."""
    at = run_the_sweep(walk_to_the_sweep(monkeypatch, openai=ScriptedRunner("openai")))
    assert "Cheapest configuration that passed" in rendered(at)

    at = _widget(at, "checkbox", f"me_{KEPT_METRIC}").set_value(False).run()
    assert "Cheapest configuration that passed" not in rendered(at)


def test_two_frameworks_are_measured_apart_and_each_result_names_its_own(monkeypatch):
    """The framework is part of the configuration, not a detail of how it ran.
    Two frameworks over one variant and one model are two answers to "what
    should we ship", and a result that did not name its own would be
    unreproducible."""
    from prompt_workbench.llm_call import registry as frameworks

    second = next(
        (
            key
            for key in frameworks.available_keys()
            if key != "openai" and not frameworks.get(key).reaches_own_provider
        ),
        None,
    )
    if second is None:
        pytest.skip("no second framework extra is installed")

    runners = {"openai": ScriptedRunner("openai"), second: ScriptedRunner(second)}
    at = walk_to_the_sweep(monkeypatch, **runners)
    at = _widget(at, "multiselect", "round_frameworks").set_value(["openai", second]).run()
    at = run_the_sweep(at)

    text = rendered(at)
    assert "via `openai`" in text
    assert f"via `{second}`" in text
    assert runners["openai"].requests and runners[second].requests
