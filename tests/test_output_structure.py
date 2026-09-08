"""An answer shape the provider enforces, or one the prompt merely asks for."""

import json

import pytest

from prompt_workbench.core import output_structure as shapes
from prompt_workbench.llm_call.openai_call import OpenAiCall
from prompt_workbench.models import CallRequest, ModelSettings, OutputStructure
from prompt_workbench.services import model_registry
from tests_support import FakeClient, answer

SHAPE = OutputStructure(
    name="verdict", schema={"type": "object", "properties": {"ok": {"type": "boolean"}}}
)

SAMPLE = {
    "data": [
        {
            "id": "strict/model",
            "name": "Strict",
            "context_length": 128000,
            "pricing": {"prompt": "0.0000001", "completion": "0.0000002"},
            "supported_parameters": ["temperature", "response_format", "structured_outputs"],
        },
        {
            "id": "loose/model",
            "name": "Loose",
            "context_length": 8000,
            "pricing": {"prompt": "0.0000001", "completion": "0.0000002"},
            "supported_parameters": ["temperature"],
        },
    ]
}


@pytest.fixture
def registry() -> model_registry.ModelRegistry:
    return model_registry.ModelRegistry(fetch=lambda: SAMPLE)


def _request(model_id: str, enforced: bool) -> CallRequest:
    return CallRequest(
        system_prompt="Be brief.",
        user_prompt="Is it ok?",
        model_id=model_id,
        settings=ModelSettings(temperature=0.1),
        output_structure=SHAPE,
        structure_is_enforced=enforced,
    )


# --- who decides ----------------------------------------------------------


def test_a_model_publishing_a_structured_output_parameter_can_enforce(
    registry: model_registry.ModelRegistry,
) -> None:
    assert registry.can_enforce_structure("strict/model") is True
    assert registry.can_enforce_structure("loose/model") is False


def test_the_request_records_which_of_the_two_is_in_force(
    registry: model_registry.ModelRegistry,
) -> None:
    strict = shapes.with_structure(_request("strict/model", False), SHAPE, registry)
    loose = shapes.with_structure(_request("loose/model", True), SHAPE, registry)
    assert strict.structure_is_enforced is True
    assert loose.structure_is_enforced is False


def test_the_two_states_are_described_differently_and_name_the_checking_metric() -> None:
    enforced = shapes.enforcement_note(True)
    asked = shapes.enforcement_note(False)
    assert enforced != asked
    assert "enforce" in enforced.lower()
    assert shapes.SHAPE_METRIC_KEY in asked


# --- what reaches the provider --------------------------------------------


def test_an_enforced_shape_travels_as_the_provider_parameter() -> None:
    client = FakeClient(answer("{}"))
    OpenAiCall(client=client).run(_request("strict/model", True))
    sent = client.calls[0]
    assert sent["response_format"]["json_schema"]["schema"] == dict(SHAPE.schema)
    assert "schema" not in sent["messages"][0]["content"]


def test_an_unenforced_shape_travels_in_the_system_prompt_instead() -> None:
    client = FakeClient(answer("{}"))
    OpenAiCall(client=client).run(_request("loose/model", False))
    sent = client.calls[0]
    assert "response_format" not in sent
    assert json.dumps(dict(SHAPE.schema), indent=2) in sent["messages"][0]["content"]


def test_no_structure_claims_nothing_of_the_answer() -> None:
    client = FakeClient(answer("free text"))
    result = OpenAiCall(client=client).run(
        CallRequest(
            system_prompt="Be brief.",
            user_prompt="Hi.",
            model_id="loose/model",
            settings=ModelSettings(),
        )
    )
    assert "response_format" not in client.calls[0]
    assert result.structure_was_enforced is False
