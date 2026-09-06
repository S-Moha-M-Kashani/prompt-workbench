"""The screen, exercised headlessly.

Every test renders the real app with no credential and no network: the model
registry is pinned to its committed snapshot, so prices are the bundled ones and
nothing reaches out.
"""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import prompt_workbench
from prompt_workbench.services import deepeval_metrics, model_registry, openrouter_client
from prompt_workbench.services.openrouter_client import ProviderConfig

APP = Path(__file__).resolve().parents[1] / "src" / "prompt_workbench" / "app.py"

_FAKE_API_KEY = "test-fake-key"
SETUP_NOTICE = "Add a provider API key in the sidebar"


@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    """No test reaches the provider's model list; the snapshot is the catalogue.

    Both the raw fetcher and the UI's cached wrapper are stubbed, and the
    Streamlit cache is cleared so a result from another test cannot leak in.
    """
    from prompt_workbench.ui import session as ui_session

    def fail() -> dict:
        raise RuntimeError("tests never fetch the live model list")

    monkeypatch.setattr(model_registry, "fetch_live", fail)
    ui_session._cached_model_list.clear()
    monkeypatch.setattr(ui_session, "_cached_model_list", fail)


def _without_credentials(monkeypatch):
    monkeypatch.setattr(openrouter_client, "config_from_env", lambda **_: ProviderConfig())


def _with_credentials(monkeypatch):
    monkeypatch.setattr(
        openrouter_client, "config_from_env", lambda **_: ProviderConfig(api_key=_FAKE_API_KEY)
    )


def _rendered(at) -> str:
    """Every string the page put on screen, including metric labels."""
    texts = [
        element.value
        for group in (at.title, at.caption, at.markdown, at.subheader, at.info,
                      at.warning, at.success, at.error)
        for element in group
        if isinstance(element.value, str)
    ]
    for tile in at.metric:
        texts.extend(
            part for part in (tile.label, tile.value, tile.delta) if isinstance(part, str)
        )
    return " ".join(texts)


def _run(monkeypatch, *, credentials: bool = False):
    (_with_credentials if credentials else _without_credentials)(monkeypatch)
    return AppTest.from_file(APP).run()


class _FakeRunner:
    """A framework that answers without a network, for the lab's own tests."""

    def __init__(self, result):
        self._result = result
        self.requests = []

    def run(self, request):
        self.requests.append(request)
        return self._result

    def describe(self) -> str:
        return "fake"


def _with_fake_runner(monkeypatch, *, tool_calls=()):
    """Inject a CallRunner, so no test ever runs a real round."""
    from prompt_workbench.models.call import CallResult
    from prompt_workbench.models.usage import TokenUsage
    from prompt_workbench.ui import lab

    runner = _FakeRunner(
        CallResult(
            answer="Ada Lovelace.",
            framework="openai",
            model_id="openai/gpt-4o-mini",
            usage=TokenUsage(tokens_in=120, tokens_out=18),
            latency_ms=412.0,
            tool_calls=tool_calls,
            model_calls=2 if tool_calls else 1,
        )
    )
    monkeypatch.setattr(lab.session, "call_runner", lambda key: runner)
    return runner


# --- it renders -----------------------------------------------------------


def test_the_app_runs_without_exception(monkeypatch):
    assert not _run(monkeypatch).exception


def test_the_title_and_version_are_shown(monkeypatch):
    at = _run(monkeypatch)
    assert at.title[0].value == "Prompt Workbench"
    assert prompt_workbench.__version__ in _rendered(at)


def test_all_six_steps_are_on_one_page(monkeypatch):
    """Sections rather than tabs: each step only means anything once the one
    before it exists."""
    rendered = _rendered(_run(monkeypatch))
    for step in ("1 · Your case", "2 · Test cases", "3 · Prompt variants",
                 "4 · The round", "5 · Metrics", "6 · Sweep"):
        assert step in rendered, step


def test_later_steps_say_what_they_are_waiting_for(monkeypatch):
    rendered = _rendered(_run(monkeypatch))
    assert "Pick the kind of job above" in rendered
    assert "Pick the kind of job first" in rendered


# --- the task type drives the page ---------------------------------------


def test_every_task_type_is_offered(monkeypatch):
    from prompt_workbench.services import task_catalog

    at = _run(monkeypatch)
    picker = next(box for box in at.selectbox if box.label == "Kind of job")
    assert len(picker.options) == len(task_catalog.all_task_types())


def test_picking_a_job_type_shows_its_approaches(monkeypatch):
    at = _run(monkeypatch)
    at = next(b for b in at.selectbox if b.label == "Kind of job").set_value("classification").run()
    labels = {box.label for box in at.checkbox}
    assert "Strict enumeration" in labels
    assert "JSON schema" in labels


def test_a_different_job_type_shows_different_approaches(monkeypatch):
    """The fix to the earlier design, visible on screen: approaches are chosen
    for the job, not a fixed list applied to everything."""
    at = _run(monkeypatch)
    at = next(b for b in at.selectbox if b.label == "Kind of job").set_value("generation").run()
    labels = {box.label for box in at.checkbox}
    assert "Outline, then write" in labels
    assert "Strict enumeration" not in labels


def test_a_high_volume_job_says_a_fine_tune_may_beat_prompting(monkeypatch):
    """A prompt workbench that never mentions this is selling something."""
    at = _run(monkeypatch)
    at = next(b for b in at.selectbox if b.label == "Kind of job").set_value("classification").run()
    assert "fine-tuned small model" in _rendered(at)


def test_picking_a_job_type_adopts_its_settings(monkeypatch):
    at = _run(monkeypatch)
    at = next(b for b in at.selectbox if b.label == "Kind of job").set_value("classification").run()
    assert "same input should get the same label" in _rendered(at)


# --- prices ---------------------------------------------------------------


def test_models_are_offered_cheapest_first_with_real_prices(monkeypatch):
    at = _run(monkeypatch)
    authoring = next(box for box in at.sidebar.selectbox if box.label == "Authoring model")
    registry = model_registry.ModelRegistry(fetch=lambda: (_ for _ in ()).throw(RuntimeError()))
    cheapest = registry.cheapest_first()[0].id
    assert authoring.options[0] == cheapest
    assert "per 1M tokens" in " ".join(c.value for c in at.sidebar.caption)


def test_the_snapshot_is_declared_as_possibly_stale(monkeypatch):
    at = _run(monkeypatch)
    assert any("snapshot" in w.value for w in at.sidebar.warning)


# --- settings -------------------------------------------------------------


def test_settings_are_disabled_for_a_model_that_ignores_them(monkeypatch):
    at = _run(monkeypatch)
    at = next(b for b in at.sidebar.selectbox if b.label == "Authoring model").set_value(
        "openai/gpt-5-mini"
    ).run()
    sampling = [s for s in at.sidebar.slider if s.label in {"temperature", "top_p"}]
    assert sampling and all(s.disabled for s in sampling)
    cap = next(box for box in at.sidebar.number_input if box.label == "max_tokens")
    assert not cap.disabled


def test_settings_are_active_for_a_tunable_model(monkeypatch):
    at = _run(monkeypatch)
    at = next(b for b in at.sidebar.selectbox if b.label == "Authoring model").set_value(
        "openai/gpt-4o-mini"
    ).run()
    assert not any(s.disabled for s in at.sidebar.slider)


def test_the_settings_live_in_the_sidebar_not_the_page(monkeypatch):
    at = _run(monkeypatch)
    assert len(at.slider) == len(at.sidebar.slider)


# --- metrics --------------------------------------------------------------


def test_the_metric_layer_states_its_install_command_when_absent(monkeypatch):
    monkeypatch.setattr(deepeval_metrics, "is_available", lambda: False)
    at = _run(monkeypatch)
    rendered = _rendered(at) + " ".join(w.value for w in at.sidebar.warning)
    assert deepeval_metrics.INSTALL_HINT in rendered


def test_the_rest_of_the_page_works_without_the_metric_extra(monkeypatch):
    monkeypatch.setattr(deepeval_metrics, "is_available", lambda: False)
    at = _run(monkeypatch)
    assert not at.exception
    assert "1 · Your case" in _rendered(at)


def test_a_job_type_brings_its_suggested_metrics(monkeypatch):
    """Grounded QA proposes faithfulness; classification does not."""
    if not deepeval_metrics.is_available():
        pytest.skip("needs the deepeval extra")
    at = _run(monkeypatch)
    at = next(b for b in at.selectbox if b.label == "Kind of job").set_value("grounded_qa").run()
    # Metric names sit in expander labels; their purpose is rendered as a caption.
    purposes = " ".join(caption.value for caption in at.caption)
    assert deepeval_metrics.get("faithfulness").purpose in purposes


def test_a_different_job_type_brings_different_metrics(monkeypatch):
    if not deepeval_metrics.is_available():
        pytest.skip("needs the deepeval extra")
    at = _run(monkeypatch)
    at = next(b for b in at.selectbox if b.label == "Kind of job").set_value("routing").run()
    purposes = " ".join(caption.value for caption in at.caption)
    assert deepeval_metrics.get("exact_match").purpose in purposes
    assert deepeval_metrics.get("faithfulness").purpose not in purposes


# --- nothing runs on its own ---------------------------------------------


def test_rendering_makes_no_provider_call(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("rendering must not build a provider client")

    monkeypatch.setattr(openrouter_client, "build_client", fail)
    assert not _run(monkeypatch, credentials=True).exception


def test_rendering_never_spawns_the_local_judge(monkeypatch):
    from prompt_workbench.services import codex_cli_client

    def fail(*args, **kwargs):
        raise AssertionError("rendering must not run the judge CLI")

    monkeypatch.setattr(codex_cli_client, "complete", fail)
    assert not _run(monkeypatch, credentials=True).exception


def test_with_a_key_the_setup_notice_disappears(monkeypatch):
    at = _run(monkeypatch, credentials=True)
    assert not any(SETUP_NOTICE in message.value for message in at.info)


def test_a_worked_example_can_be_loaded_as_a_starting_point(monkeypatch):
    """The examples show what a well-shaped case looks like. They are not the
    thing the workbench is for, so they sit in an expander, not the front page."""
    from prompt_workbench.services import use_case_catalog

    at = _run(monkeypatch)
    picker = next(box for box in at.selectbox if box.label == "Worked situations")
    assert len(picker.options) == len(use_case_catalog.all_use_cases())


def test_every_example_names_the_kind_of_job_it_is(monkeypatch):
    """Loading one settles the task type too, or it would drop the user back
    into the choice it exists to demonstrate."""
    from prompt_workbench.services import task_catalog, use_case_catalog

    known = {task.key for task in task_catalog.all_task_types()}
    for example in use_case_catalog.all_use_cases():
        assert example.task_type in known, example.key


def test_a_case_can_be_written_without_a_provider_key(monkeypatch):
    """Generation needs a key; writing a case down does not. A workbench that is
    inert until someone has paid for a key is not usable."""
    at = _run(monkeypatch)
    at = next(b for b in at.selectbox if b.label == "Kind of job").set_value("classification").run()

    add = next(button for button in at.button if button.label == "Add this case")
    assert not add.disabled

    inputs = {area.label for area in at.text_area}
    assert "Input" in inputs


def test_an_existing_prompt_can_be_pasted_in_without_a_key(monkeypatch):
    """The obvious thing to want: bring the prompt you already have and find out
    whether any generated approach beats it."""
    at = _run(monkeypatch)
    at = next(b for b in at.selectbox if b.label == "Kind of job").set_value("classification").run()

    add = next(button for button in at.button if button.label == "Add this prompt")
    assert not add.disabled


def test_the_front_page_does_not_overstate_what_works_without_a_key(monkeypatch):
    at = _run(monkeypatch)
    notice = " ".join(message.value for message in at.info)
    assert "generate" in notice.lower()
    assert "paste" in notice.lower() or "write" in notice.lower()


# --- a loaded example lands where the user is looking ----------------------


def _load_first_example(monkeypatch):
    from prompt_workbench.services import use_case_catalog

    example = use_case_catalog.all_use_cases()[0]
    at = _run(monkeypatch)
    at = next(b for b in at.selectbox if b.label == "Worked situations").set_value(
        example.key
    ).run()
    at = next(button for button in at.button if button.label == "Load it").click().run()
    return example, at


def _description_box(at):
    return next(area for area in at.text_area if area.label == "What must this prompt do?")


def test_loading_an_example_fills_in_the_description_box(monkeypatch):
    """It has to land in the widget, not only in the session behind it — an
    example that silently fills nothing in reads as a broken button."""
    example, at = _load_first_example(monkeypatch)
    assert example.situation[:40] in _description_box(at).value


def test_loading_an_example_settles_the_job_picker(monkeypatch):
    example, at = _load_first_example(monkeypatch)
    picker = next(box for box in at.selectbox if box.label == "Kind of job")
    assert picker.value == example.task_type


def test_a_loaded_example_survives_the_next_rerun(monkeypatch):
    """The description box is read back on the following run; if the load never
    reached it, that read wipes the description it just wrote."""
    _, at = _load_first_example(monkeypatch)
    assert _description_box(at.run()).value.strip()


def test_a_described_case_settles_the_job_picker_too(monkeypatch):
    """The proposal is only useful if the control that owns the choice shows it."""
    at = _run(monkeypatch)
    at = _description_box(at).set_value(
        "Sort incoming support tickets into one of six queues."
    ).run()
    picker = next(box for box in at.selectbox if box.label == "Kind of job")
    assert picker.value == "classification"


# --- the sweep refuses out loud -------------------------------------------


def _ready_to_sweep(monkeypatch, *, credentials: bool = False, description: str = ""):
    """Walk the page to the point where a sweep could run.

    The sweep section is gated on the metric layer, so without that optional
    extra there is nothing here to assert on — an uninstalled extra is a
    supported state, not a failure.
    """
    if not deepeval_metrics.is_available():
        pytest.skip("needs the deepeval extra")
    at = _run(monkeypatch, credentials=credentials)
    if description:
        at = _description_box(at).set_value(description).run()
    at = next(box for box in at.selectbox if box.label == "Kind of job").set_value(
        "classification"
    ).run()
    at = next(area for area in at.text_area if area.label == "Input").set_value(
        "charged twice this month"
    ).run()
    at = next(button for button in at.button if button.label == "Add this case").click().run()
    at = next(area for area in at.text_area if area.label == "System prompt").set_value(
        "You are a classifier."
    ).run()
    at = next(button for button in at.button if button.label == "Add this prompt").click().run()
    return at


def _sweep_button(at):
    return next(button for button in at.button if button.label == "Run the sweep")


def test_a_sweep_with_nothing_described_refuses_out_loud(monkeypatch):
    """The run used to return silently when there was no case brief, so the
    button looked live and did nothing."""
    at = _ready_to_sweep(monkeypatch, credentials=True)
    assert _sweep_button(at).disabled
    assert "Describe the case" in _rendered(at)


def test_a_sweep_without_a_key_says_that_is_why(monkeypatch):
    at = _ready_to_sweep(monkeypatch, description="Sort tickets into one of six queues.")
    button = _sweep_button(at)
    assert button.disabled
    assert button.help and "API key" in button.help


def test_a_sweep_whose_metrics_are_unconfigured_points_at_the_metric_step(monkeypatch):
    """Every task type suggests metrics that need something from the case before
    they can score it, so this is the state a new user actually lands in."""
    at = _ready_to_sweep(
        monkeypatch, credentials=True, description="Sort tickets into one of six queues."
    )
    button = _sweep_button(at)
    assert button.disabled
    assert button.help and "step 5" in button.help


def test_a_sweep_with_no_metric_left_on_refuses(monkeypatch):
    """A sweep that measures nothing ranks nothing; it must not look runnable."""
    at = _ready_to_sweep(
        monkeypatch, credentials=True, description="Sort tickets into one of six queues."
    )
    for box in [b for b in at.checkbox if b.key and b.key.startswith("me_")]:
        at = box.set_value(False).run()
    button = _sweep_button(at)
    assert button.disabled
    assert button.help and "at least one metric" in button.help


def test_a_sweep_is_offered_once_its_metrics_can_score_the_cases(monkeypatch):
    at = _ready_to_sweep(
        monkeypatch, credentials=True, description="Sort tickets into one of six queues."
    )
    # Exact match and JSON correctness want fields these cases do not carry;
    # the criteria metric can score them as they are.
    for key in ("me_exact_match", "me_json_correctness"):
        at = next(box for box in at.checkbox if box.key == key).set_value(False).run()
    assert not _sweep_button(at).disabled


# --- the round: the shape, enforced or asked for --------------------------


def _lab_shape_on(at):
    return next(box for box in at.checkbox if box.key == "round_shape_on")


def test_the_round_offers_an_optional_answer_shape(monkeypatch):
    at = _run(monkeypatch, credentials=True)
    assert _lab_shape_on(at).value is False
    assert "nothing is claimed of it" in _rendered(at)


def test_switching_the_shape_on_states_which_of_the_two_is_in_force(monkeypatch):
    at = _run(monkeypatch, credentials=True)
    at = _lab_shape_on(at).set_value(True).run()
    text = _rendered(at)
    assert "enforce" in text.lower()


def test_an_unenforced_shape_names_the_metric_that_would_check_it(monkeypatch):
    """The wording for the two states must differ, and the weaker one must say
    what to do about it — otherwise a claim reads as a guarantee."""
    from prompt_workbench.core import output_structure as shapes

    asked = shapes.enforcement_note(False)
    assert shapes.SHAPE_METRIC_KEY in asked
    assert asked != shapes.enforcement_note(True)


# --- the model panel and the further parameters ---------------------------


def _round_model_picker(at):
    return next(box for box in at.selectbox if box.key == "round_model_pick")


def test_the_round_offers_the_whole_catalogue_not_only_the_shortlist(monkeypatch):
    at = _run(monkeypatch, credentials=True)
    offered = set(_round_model_picker(at).options)
    from prompt_workbench.services.model_registry import ModelRegistry

    catalogue = {e.id for e in ModelRegistry(fetch=lambda: _snapshot()).selectable()}
    assert offered == catalogue


def _snapshot() -> dict:
    import json

    from prompt_workbench.services.model_registry import SNAPSHOT_PATH

    return json.loads(SNAPSHOT_PATH.read_text())


def test_the_selected_model_shows_what_it_costs_and_what_it_accepts(monkeypatch):
    at = _run(monkeypatch, credentials=True)
    text = _rendered(at)
    assert "per 1M tokens" in text
    assert "context" in text.lower()
    assert "per 1,000 calls" in text
    assert "tool" in text.lower()


def test_the_model_panel_lists_the_parameters_the_model_publishes(monkeypatch):
    at = _run(monkeypatch, credentials=True)
    assert "response_format" in _rendered(at)


def test_the_model_panel_says_when_its_information_is_from_the_snapshot(monkeypatch):
    at = _run(monkeypatch, credentials=True)
    assert "stale" in _rendered(at).lower() or "snapshot" in _rendered(at).lower()


def test_a_further_parameter_can_be_added_from_the_models_own_list(monkeypatch):
    at = _run(monkeypatch, credentials=True)
    picker = next(box for box in at.selectbox if box.key == "extra_param_pick")
    assert picker.options, "the model publishes parameters beyond the five sliders"
    assert "temperature" not in picker.options, "already a slider"


# --- the single-round lab -------------------------------------------------


def _framework_picker(at):
    return next(box for box in at.multiselect if box.key == "round_frameworks")


def test_the_lab_lists_every_framework_with_its_availability(monkeypatch):
    from prompt_workbench.llm_call import registry as frameworks

    at = _run(monkeypatch, credentials=True)
    offered = " ".join(_framework_picker(at).options)
    for entry in frameworks.all_frameworks():
        assert entry.label in offered, entry.key


def test_an_unavailable_framework_is_named_with_its_install_command(monkeypatch):
    from prompt_workbench.llm_call import registry as frameworks

    def nothing_optional(name, *args, **kwargs):
        return None if name != "openai" else object()

    monkeypatch.setattr(frameworks.importlib.util, "find_spec", nothing_optional)
    at = _run(monkeypatch, credentials=True)
    assert "uv sync --extra langchain" in _rendered(at)


def test_the_round_requires_both_prompts(monkeypatch):
    at = _run(monkeypatch, credentials=True)
    labels = {area.label for area in at.text_area}
    assert {"Round system prompt", "Round user prompt"} <= labels
    button = _run_round_button(at)
    assert button.disabled
    assert button.help and "system prompt" in button.help


def _run_round_button(at):
    return next(button for button in at.button if button.label == "Run the round")


def test_tools_are_switchable_with_a_name_and_a_description(monkeypatch):
    at = _run(monkeypatch, credentials=True)
    switch = next(box for box in at.checkbox if box.key == "round_tools_on")
    assert switch.value is False
    at = switch.set_value(True).run()
    labels = {field.label for field in at.text_input}
    assert "Tool name" in labels
    labels = {area.label for area in at.text_area}
    assert "Tool description" in labels


def test_the_run_is_disabled_without_a_credential(monkeypatch):
    at = _run(monkeypatch, credentials=False)
    button = _run_round_button(at)
    assert button.disabled
    assert button.help and "API key" in button.help


def test_the_result_panel_reports_what_the_round_actually_cost(monkeypatch):
    """Answer, latency, tokens each way, model calls, cost or unknown, trace."""
    _with_fake_runner(monkeypatch)
    at = _run(monkeypatch, credentials=True)
    at = _prepared_round(at)
    at = _run_round_button(at).click().run()
    text = _rendered(at)
    for expected in ("Latency", "Tokens in", "Tokens out", "Model calls"):
        assert expected in text, expected
    assert "Ada Lovelace" in text


def _prepared_round(at):
    at = next(a for a in at.text_area if a.label == "Round system prompt").set_value(
        "Be brief."
    ).run()
    return next(a for a in at.text_area if a.label == "Round user prompt").set_value(
        "Who is customer 7?"
    ).run()


def test_the_tool_trace_is_shown_in_the_order_it_happened(monkeypatch):
    from prompt_workbench.models.call import ToolInvocation

    _with_fake_runner(
        monkeypatch,
        tool_calls=(
            ToolInvocation(name="lookup", arguments={"id": 7}, order=0),
            ToolInvocation(name="notify", arguments={}, order=1),
        ),
    )
    at = _run(monkeypatch, credentials=True)
    at = next(box for box in at.checkbox if box.key == "round_tools_on").set_value(True).run()
    at = _prepared_round(at)
    at = _run_round_button(at).click().run()
    assert "tool trace" in _rendered(at).lower()


# --- the call as code -----------------------------------------------------


def test_the_sketch_opens_on_request_and_is_labelled_as_a_representation(monkeypatch):
    at = _run(monkeypatch, credentials=True)
    at = _prepared_round(at)
    at = next(box for box in at.checkbox if box.key == "show_sketch").set_value(True).run()
    text = _rendered(at)
    assert "chat.completions.create" in " ".join(block.value for block in at.code)
    assert "not the workbench's own code" in text


def test_the_sketch_is_not_offered_as_a_file(monkeypatch):
    at = _run(monkeypatch, credentials=True)
    at = _prepared_round(at)
    at = next(box for box in at.checkbox if box.key == "show_sketch").set_value(True).run()
    assert not list(at.get("download_button")), "a sketch is a picture, not an export"


# --- the limits stay next to the numbers ----------------------------------


def test_every_required_statement_reaches_the_page_showing_the_numbers(monkeypatch):
    from prompt_workbench.core import considerations

    _with_fake_runner(monkeypatch)
    at = _run(monkeypatch, credentials=True)
    at = _prepared_round(at)
    at = _run_round_button(at).click().run()
    text = " ".join(_rendered(at).split())
    for item in considerations.all_considerations():
        assert " ".join(item.statement.split()) in text, item.key


def test_the_statements_come_from_the_same_source_as_the_document(monkeypatch):
    """Not retyped into the UI: the page reads the module the doc quotes."""
    from prompt_workbench.ui import lab

    assert lab.considerations.all_considerations()


# --- the second provider stays a second catalogue -------------------------


def test_anthropic_needs_its_own_key_and_says_so(monkeypatch):
    pytest.importorskip("anthropic", reason="needs the anthropic extra")
    at = _run(monkeypatch, credentials=True)
    labels = {field.label for field in at.sidebar.text_input}
    assert "Anthropic API key" in labels


def test_the_anthropic_models_never_appear_in_the_provider_picker(monkeypatch):
    """A Claude model priced from the workbench's provider list would be a
    fabricated number wearing the look of a measured one."""
    from prompt_workbench.services import anthropic_catalog

    at = _run(monkeypatch, credentials=True)
    offered = set(_round_model_picker(at).options)
    for entry in anthropic_catalog.all_models():
        assert entry.id not in offered, entry.id


def test_choosing_anthropic_shows_its_own_catalogue_and_its_staleness(monkeypatch):
    pytest.importorskip("anthropic", reason="needs the anthropic extra")
    from prompt_workbench.services import anthropic_catalog

    at = _run(monkeypatch, credentials=True)
    at = _framework_picker(at).set_value(["Anthropic SDK (separate provider)"]).run()
    text = _rendered(at)
    assert "claude-opus-5" in text or "Claude Opus 5" in text
    assert anthropic_catalog.AS_OF in text
