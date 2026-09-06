"""Settings: five typed knobs, plus any other parameter the model publishes."""

import pytest

from prompt_workbench.models import ModelSettings
from prompt_workbench.services import model_registry

SAMPLE = {
    "data": [
        {
            "id": "rich/model",
            "name": "Rich",
            "context_length": 128000,
            "pricing": {"prompt": "0.0000001", "completion": "0.0000002"},
            "supported_parameters": ["temperature", "max_tokens", "seed", "stop"],
        },
        {
            "id": "plain/model",
            "name": "Plain",
            "context_length": 8000,
            "pricing": {"prompt": "0.0000001", "completion": "0.0000002"},
            "supported_parameters": ["temperature"],
        },
    ]
}


@pytest.fixture
def registry() -> model_registry.ModelRegistry:
    return model_registry.ModelRegistry(fetch=lambda: SAMPLE)


def test_named_fields_stay_typed_and_omit_what_is_unset() -> None:
    settings = ModelSettings(temperature=0.3)
    assert settings.temperature == 0.3
    assert settings.as_params() == {"temperature": 0.3}


def test_extra_keys_serialise_beside_the_named_fields() -> None:
    settings = ModelSettings(temperature=0.3, extra={"seed": 7, "stop": ["END"]})
    assert settings.as_params() == {"temperature": 0.3, "seed": 7, "stop": ["END"]}


def test_a_named_field_is_not_duplicated_by_an_extra_of_the_same_name() -> None:
    settings = ModelSettings(temperature=0.3, extra={"temperature": 0.9})
    assert settings.as_params()["temperature"] == 0.9


def test_adding_a_published_parameter_is_allowed(
    registry: model_registry.ModelRegistry,
) -> None:
    settings = registry.with_parameter("rich/model", ModelSettings(), "seed", 7)
    assert settings.extra["seed"] == 7


def test_adding_a_parameter_the_model_does_not_publish_is_refused(
    registry: model_registry.ModelRegistry,
) -> None:
    with pytest.raises(ValueError, match="seed"):
        registry.with_parameter("plain/model", ModelSettings(), "seed", 7)


def test_switching_models_drops_unpublished_parameters_and_names_them(
    registry: model_registry.ModelRegistry,
) -> None:
    settings = ModelSettings(temperature=0.3, max_tokens=100, extra={"seed": 7})
    kept, dropped = registry.strip_unsupported("plain/model", settings)
    assert kept.as_params() == {"temperature": 0.3}
    assert sorted(dropped) == ["max_tokens", "seed"]


def test_addable_parameters_exclude_the_ones_already_shown(
    registry: model_registry.ModelRegistry,
) -> None:
    addable = registry.addable_parameters("rich/model", ModelSettings(extra={"seed": 7}))
    assert addable == ("stop",)
