"""The Streamlit surface, exercised headlessly.

Every one of these renders the real app with a fake or absent credential. The
recurring assertion is the important one: rendering a page must never reach the
provider. Only an explicit action may.
"""

from pathlib import Path

from streamlit.testing.v1 import AppTest

import prompt_workbench
from prompt_workbench.services import (
    codex_cli_client,
    model_catalog,
    openrouter_client,
    use_case_catalog,
)
from prompt_workbench.services.openrouter_client import ProviderConfig

APP = Path(__file__).resolve().parents[1] / "src" / "prompt_workbench" / "app.py"

_FAKE_API_KEY = "test-fake-key"
SETUP_NOTICE = "Add a provider API key in the sidebar"


def _without_credentials(monkeypatch):
    """Pin the app to an empty config so the tests never read a developer's
    .env or environment — the app must behave identically on a clean clone
    with no key available at all."""
    monkeypatch.setattr(openrouter_client, "config_from_env", lambda **_: ProviderConfig())


def _with_credentials(monkeypatch):
    monkeypatch.setattr(
        openrouter_client,
        "config_from_env",
        lambda **_: ProviderConfig(api_key=_FAKE_API_KEY),
    )


def _rendered(at) -> str:
    """Everything textual the app put on screen."""
    return " ".join(
        element.value
        for group in (at.title, at.caption, at.markdown, at.subheader, at.info, at.warning)
        for element in group
        if isinstance(element.value, str)
    )


def test_app_runs_without_exception(monkeypatch):
    _without_credentials(monkeypatch)

    at = AppTest.from_file(APP).run()

    assert not at.exception


def test_app_shows_title_and_version(monkeypatch):
    _without_credentials(monkeypatch)

    at = AppTest.from_file(APP).run()

    assert at.title[0].value == "Prompt Workbench"
    assert prompt_workbench.__version__ in _rendered(at)


def test_sidebar_offers_a_key_field_and_a_catalog_model(monkeypatch):
    _without_credentials(monkeypatch)

    at = AppTest.from_file(APP).run()

    assert at.sidebar.text_input[0].label == "API key"
    assert at.sidebar.selectbox[0].value in {m.id for m in model_catalog.all_models()}


def test_without_a_key_the_app_explains_what_is_disabled(monkeypatch):
    _without_credentials(monkeypatch)

    at = AppTest.from_file(APP).run()

    assert any(SETUP_NOTICE in message.value for message in at.info)


def test_with_a_key_the_setup_notice_disappears(monkeypatch):
    _with_credentials(monkeypatch)

    at = AppTest.from_file(APP).run()

    assert not any(SETUP_NOTICE in message.value for message in at.info)


def test_rendering_makes_no_provider_call(monkeypatch):
    """Rendering must never reach the provider; only explicit actions may."""
    _with_credentials(monkeypatch)

    def fail(*args, **kwargs):
        raise AssertionError("rendering must not build a provider client")

    monkeypatch.setattr(openrouter_client, "build_client", fail)

    at = AppTest.from_file(APP).run()

    assert not at.exception


def test_rendering_never_spawns_the_local_judge(monkeypatch):
    """The CLI judge costs a process spawn per call, so a page render must not
    start one — only pressing Evaluate may."""
    _with_credentials(monkeypatch)

    def fail(*args, **kwargs):
        raise AssertionError("rendering must not run the judge CLI")

    monkeypatch.setattr(codex_cli_client, "complete", fail)

    at = AppTest.from_file(APP).run()

    assert not at.exception


def test_a_missing_local_judge_is_reported_rather_than_assumed(monkeypatch):
    _without_credentials(monkeypatch)
    monkeypatch.setattr(codex_cli_client, "is_available", lambda **_: False)

    at = AppTest.from_file(APP).run()

    assert not at.exception
    assert any("not installed" in caption.value for caption in at.sidebar.caption)


def test_the_use_case_dropdown_offers_every_shipped_situation(monkeypatch):
    _without_credentials(monkeypatch)

    at = AppTest.from_file(APP).run()

    # AppTest reports the formatted labels, so compare against those.
    picker = next(box for box in at.selectbox if box.label == "Use case")
    shipped = use_case_catalog.all_use_cases()
    assert len(picker.options) == len(shipped)
    for case in shipped:
        assert any(case.label in option for option in picker.options), case.key


def test_nothing_is_selected_until_the_user_picks(monkeypatch):
    """The screen must not quietly load a situation the user did not choose."""
    _without_credentials(monkeypatch)

    at = AppTest.from_file(APP).run()

    assert any("Pick a use case" in message.value for message in at.info)


def test_picking_a_use_case_writes_its_situation_into_the_chat(monkeypatch):
    _without_credentials(monkeypatch)

    at = AppTest.from_file(APP).run()
    picker = next(box for box in at.selectbox if box.label == "Use case")
    at = picker.set_value("injection_resistance").run()

    rendered = _rendered(at)
    assert "Prompt injection" in rendered
    assert "Obeying it" in rendered, "the trap must be stated, not just the situation"


def test_picking_a_use_case_loads_its_prompt_for_editing(monkeypatch):
    _without_credentials(monkeypatch)

    at = AppTest.from_file(APP).run()
    picker = next(box for box in at.selectbox if box.label == "Use case")
    at = picker.set_value("grounded_briefing").run()

    prompt_box = next(area for area in at.text_area if area.label == "System prompt")
    assert "{history}" in prompt_box.value, "placeholders stay visible while editing"


def test_the_chat_offers_both_modes(monkeypatch):
    _without_credentials(monkeypatch)

    at = AppTest.from_file(APP).run()
    picker = next(box for box in at.selectbox if box.label == "Use case")
    at = picker.set_value("grounded_briefing").run()

    mode = next(radio for radio in at.radio if radio.label == "Mode")
    assert set(mode.options) == {"Prompt engineer", "End user (one-shot)"}


def test_there_is_exactly_one_chat_input(monkeypatch):
    """One window. The five-area workspace is gone."""
    _without_credentials(monkeypatch)

    at = AppTest.from_file(APP).run()
    picker = next(box for box in at.selectbox if box.label == "Use case")
    at = picker.set_value("grounded_briefing").run()

    assert len(at.chat_input) == 1


def test_the_main_screen_carries_no_knobs(monkeypatch):
    """Settings belong in the sidebar. The screen itself stays a use case, a
    prompt and a conversation."""
    _without_credentials(monkeypatch)

    at = AppTest.from_file(APP).run()
    picker = next(box for box in at.selectbox if box.label == "Use case")
    at = picker.set_value("grounded_briefing").run()

    sidebar_sliders = {slider.label for slider in at.sidebar.slider}
    assert {"temperature", "top_p"} <= sidebar_sliders
    assert len(at.slider) == len(at.sidebar.slider), "no slider outside the sidebar"


def test_model_settings_are_offered_for_a_model_that_honours_them(monkeypatch):
    _without_credentials(monkeypatch)

    at = AppTest.from_file(APP).run()
    model_pick = next(box for box in at.sidebar.selectbox if box.label == "Model under test")
    at = model_pick.set_value("openai/gpt-4o-mini").run()

    labels = {slider.label for slider in at.sidebar.slider}
    assert {"temperature", "top_p"} <= labels
    assert not any(slider.disabled for slider in at.sidebar.slider)


def test_model_settings_stay_visible_but_disabled_for_a_reasoning_model(monkeypatch):
    """They are shown rather than hidden so the reason is legible: this model
    drops them, it is not that the workbench forgot to offer them."""
    _without_credentials(monkeypatch)

    at = AppTest.from_file(APP).run()
    model_pick = next(box for box in at.sidebar.selectbox if box.label == "Model under test")
    at = model_pick.set_value("openai/gpt-5-mini").run()

    sampling = [s for s in at.sidebar.slider if s.label in {"temperature", "top_p"}]
    assert sampling, "sampling settings must stay on screen"
    assert all(s.disabled for s in sampling)


def test_the_output_cap_stays_active_for_a_reasoning_model(monkeypatch):
    """max_tokens is the one knob a reasoning model does honour."""
    _without_credentials(monkeypatch)

    at = AppTest.from_file(APP).run()
    model_pick = next(box for box in at.sidebar.selectbox if box.label == "Model under test")
    at = model_pick.set_value("openai/gpt-5-mini").run()

    cap = next(box for box in at.sidebar.number_input if box.label == "max_tokens")
    assert not cap.disabled


def test_the_sidebar_says_why_settings_are_disabled(monkeypatch):
    _without_credentials(monkeypatch)

    at = AppTest.from_file(APP).run()
    model_pick = next(box for box in at.sidebar.selectbox if box.label == "Model under test")
    at = model_pick.set_value("openai/gpt-5-mini").run()

    captions = " ".join(caption.value for caption in at.sidebar.caption)
    assert "ignore" in captions.lower() or "drop" in captions.lower()
