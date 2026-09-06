"""The call as code: a picture of the request, not the workbench's own code."""

import pytest

from prompt_workbench.core import code_sketch
from prompt_workbench.models import (
    CallRequest,
    ModelSettings,
    OutputStructure,
    ToolSpec,
)

FRAMEWORKS = ("openai", "langchain", "langgraph")

LOOKUP = ToolSpec(name="lookup", description="Look a customer up.")
NOTIFY = ToolSpec(name="notify", description="Send a notice.")
SHAPE = OutputStructure(name="verdict", schema={"type": "object"})


def _request(**overrides: object) -> CallRequest:
    fields: dict[str, object] = {
        "system_prompt": "Be brief.",
        "user_prompt": "Who is customer 7?",
        "model_id": "openai/gpt-4o-mini",
        "settings": ModelSettings(temperature=0.2, extra={"seed": 7}),
    }
    fields.update(overrides)
    return CallRequest(**fields)  # type: ignore[arg-type]


def _placeholders(segments) -> dict[str, object]:
    return {s.name: s for s in segments if s.is_placeholder}


def _text(segments) -> str:
    return "".join(s.text for s in segments)


@pytest.mark.parametrize("framework", FRAMEWORKS)
def test_every_framework_sketches_the_prompts_as_typed_placeholders(framework) -> None:
    segments = code_sketch.sketch(framework, _request())
    named = _placeholders(segments)
    assert named["system_prompt"].type_name == "str"
    assert named["user_prompt"].type_name == "str"
    assert named["system_prompt"].value == "Be brief."


@pytest.mark.parametrize("framework", FRAMEWORKS)
def test_the_sketch_is_in_the_frameworks_own_idiom(framework) -> None:
    text = _text(code_sketch.sketch(framework, _request()))
    expected = {
        "openai": "chat.completions.create",
        "langchain": "create_agent",
        "langgraph": "StateGraph",
    }[framework]
    assert expected in text


@pytest.mark.parametrize("framework", FRAMEWORKS)
def test_a_switched_off_input_never_appears(framework) -> None:
    text = _text(code_sketch.sketch(framework, _request()))
    assert "tools" not in text
    assert "response_format" not in text
    assert "json" not in text.lower()


@pytest.mark.parametrize("framework", FRAMEWORKS)
def test_tools_appear_when_on_and_carry_their_names(framework) -> None:
    segments = code_sketch.sketch(framework, _request(tools=(LOOKUP, NOTIFY)))
    assert "tools" in _text(segments)
    tools = _placeholders(segments)["tools"]
    assert tools.type_name == "list"
    assert tools.value == ["lookup", "notify"]
    assert "mocked" in tools.note.lower()


def test_an_enforced_shape_appears_as_the_provider_parameter() -> None:
    segments = code_sketch.sketch(
        "openai", _request(output_structure=SHAPE, structure_is_enforced=True)
    )
    assert "response_format" in _text(segments)
    assert _placeholders(segments)["output_schema"].type_name == "dict"


@pytest.mark.parametrize("framework", FRAMEWORKS)
def test_every_parameter_in_the_request_appears_and_nothing_else(framework) -> None:
    segments = code_sketch.sketch(framework, _request())
    named = _placeholders(segments)
    assert named["temperature"].value == 0.2
    assert named["seed"].value == 7
    assert "top_p" not in named


def test_a_parameter_the_model_does_not_publish_never_appears() -> None:
    """The published list is the filter, applied before the sketch is drawn."""
    segments = code_sketch.sketch(
        "openai", _request(), published_parameters=("temperature",)
    )
    named = _placeholders(segments)
    assert "temperature" in named
    assert "seed" not in named


def test_an_unknown_framework_is_refused_by_name() -> None:
    with pytest.raises(KeyError, match="nonsense"):
        code_sketch.sketch("nonsense", _request())


def test_a_placeholder_shows_its_name_not_its_content() -> None:
    """A 1,900-word prompt would drown the sketch; the shape is what is read."""
    long_prompt = "word " * 500
    segments = code_sketch.sketch("openai", _request(system_prompt=long_prompt))
    text = _text(segments)
    assert long_prompt not in text
    assert "str(system_prompt)" in text
