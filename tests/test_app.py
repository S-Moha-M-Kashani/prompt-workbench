"""The Streamlit surface, exercised headlessly.

Every one of these renders the real app with a fake or absent credential. The
recurring assertion is the important one: rendering a page must never reach the
provider. Only an explicit action may.
"""

from pathlib import Path

from streamlit.testing.v1 import AppTest

import prompt_workbench
from prompt_workbench.services import codex_cli_client, model_catalog, openrouter_client
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


def test_all_five_workspace_areas_are_present(monkeypatch):
    _without_credentials(monkeypatch)

    at = AppTest.from_file(APP).run()

    rendered = _rendered(at)
    for area in ("Define the use case", "Ground truth", "Candidate system prompts",
                 "Test a candidate", "Evaluate"):
        assert area in rendered, f"expected the {area!r} area to render"


def test_sidebar_offers_a_key_field_and_a_catalog_model(monkeypatch):
    _without_credentials(monkeypatch)

    at = AppTest.from_file(APP).run()

    assert at.sidebar.text_input[0].label == "API key"
    assert at.sidebar.selectbox[0].value in {m.id for m in model_catalog.all_models()}


def test_the_platform_instruction_is_editable_in_the_sidebar(monkeypatch):
    """The workbench's own instruction is inspectable, and separate from the
    candidate prompts under test."""
    _without_credentials(monkeypatch)

    at = AppTest.from_file(APP).run()

    labels = [area.label for area in at.sidebar.text_area]
    assert "Prompt-engineer instruction" in labels


def test_without_a_key_the_app_explains_what_is_disabled(monkeypatch):
    _without_credentials(monkeypatch)

    at = AppTest.from_file(APP).run()

    assert any(SETUP_NOTICE in message.value for message in at.info)


def test_with_a_key_the_setup_notice_disappears(monkeypatch):
    _with_credentials(monkeypatch)

    at = AppTest.from_file(APP).run()

    assert not any(SETUP_NOTICE in message.value for message in at.info)


def test_before_a_brief_is_confirmed_the_other_areas_say_so(monkeypatch):
    _with_credentials(monkeypatch)

    at = AppTest.from_file(APP).run()

    assert any("Confirm a brief" in message.value for message in at.info)


def test_the_brief_form_offers_every_discovery_field(monkeypatch):
    _without_credentials(monkeypatch)

    at = AppTest.from_file(APP).run()

    labels = {area.label for area in at.text_area}
    assert {"Purpose", "Audience", "Constraints", "Failure cases"} <= labels


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


def test_the_evaluate_area_states_the_limits_of_a_grade(monkeypatch):
    """The honest caveats belong next to the controls, not in a document
    nobody opens."""
    _without_credentials(monkeypatch)

    at = AppTest.from_file(APP).run()

    rendered = _rendered(at)
    assert "Nothing is scored until you do" in rendered


def test_a_missing_local_judge_is_reported_rather_than_assumed(monkeypatch):
    _without_credentials(monkeypatch)
    monkeypatch.setattr(codex_cli_client, "is_available", lambda **_: False)

    at = AppTest.from_file(APP).run()

    assert not at.exception
    assert any("not installed" in caption.value for caption in at.sidebar.caption)
