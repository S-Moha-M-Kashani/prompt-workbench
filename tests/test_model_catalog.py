from dataclasses import fields

import pytest

from prompt_workbench.models import ModelSettings
from prompt_workbench.services import model_catalog


def test_catalog_is_not_empty_and_ids_are_unique():
    ids = [model.id for model in model_catalog.all_models()]

    assert ids
    assert len(ids) == len(set(ids))


def test_every_tunable_name_is_a_model_settings_field():
    """A knob name that does not exist on ModelSettings can never be honoured,
    so it would silently mislead the caller."""
    known = set(model_catalog.SETTING_NAMES)

    assert known <= {f.name for f in fields(ModelSettings)}
    for model in model_catalog.all_models():
        assert model.tunable <= known, model.id


def test_get_returns_the_model_and_raises_for_an_unknown_id():
    first = model_catalog.all_models()[0]

    assert model_catalog.get(first.id) is first
    with pytest.raises(KeyError):
        model_catalog.get("nobody/no-such-model")


def test_supports_is_permissive_for_an_unknown_model():
    """The provider is the real enforcer; an id we have no facts about must not
    be blocked locally."""
    assert model_catalog.supports("nobody/no-such-model", "temperature") is True


def test_fully_tunable_models_honour_every_knob():
    for model in model_catalog.fully_tunable_models():
        for name in model_catalog.SETTING_NAMES:
            assert model_catalog.supports(model.id, name), (model.id, name)


def test_ignored_settings_reports_only_requested_knobs():
    reasoning_only = next(
        model
        for model in model_catalog.all_models()
        if "temperature" not in model.tunable
    )

    assert model_catalog.ignored_settings(reasoning_only.id, ModelSettings()) == ()
    assert model_catalog.ignored_settings(
        reasoning_only.id, ModelSettings(temperature=0.7)
    ) == ("temperature",)


def test_ignored_settings_is_empty_for_a_fully_tunable_model():
    tunable = model_catalog.fully_tunable_models()[0]

    assert (
        model_catalog.ignored_settings(tunable.id, ModelSettings(temperature=0.7, top_p=0.9))
        == ()
    )


def test_every_model_declares_a_context_window():
    """The context-limit warning needs a real number per model; a missing one
    would silently become 'unlimited' and block nothing."""
    for model in model_catalog.all_models():
        assert model.context_window > 0, model.id


def test_the_context_window_of_an_unknown_model_is_a_safe_floor():
    """A model the catalog has no facts about still needs a budget to warn
    against, and guessing high is the failure that hits a provider error."""
    assert model_catalog.context_window("nobody/no-such-model") == model_catalog.FALLBACK_CONTEXT


def test_the_default_model_is_in_the_catalog():
    assert model_catalog.get(model_catalog.DEFAULT_MODEL_ID)


def test_the_catalog_offers_both_reasoning_and_fully_tunable_models():
    """Comparing sampling settings needs a model that honours them; the
    recommended default is a reasoning model that does not."""
    assert model_catalog.fully_tunable_models()
    assert any("temperature" not in model.tunable for model in model_catalog.all_models())
