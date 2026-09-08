"""The one shape every framework is reached through: request in, result out."""

import dataclasses

import pytest

from prompt_workbench.models import ModelSettings, TokenUsage
from prompt_workbench.models.call import (
    CallRequest,
    CallResult,
    OutputStructure,
    ToolInvocation,
    ToolSpec,
)


def _request(**overrides: object) -> CallRequest:
    fields: dict[str, object] = {
        "system_prompt": "You answer briefly.",
        "user_prompt": "Say hello.",
        "model_id": "openai/gpt-4o-mini",
        "settings": ModelSettings(temperature=0.2),
    }
    fields.update(overrides)
    return CallRequest(**fields)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "shape", [ToolSpec, ToolInvocation, OutputStructure, CallRequest, CallResult]
)
def test_the_call_shapes_are_frozen(shape: type) -> None:
    assert dataclasses.is_dataclass(shape)
    assert shape.__dataclass_params__.frozen  # type: ignore[attr-defined]


@pytest.mark.parametrize("blank", ["", "   ", "\n"])
def test_a_blank_user_prompt_is_refused(blank: str) -> None:
    with pytest.raises(ValueError, match="user prompt"):
        _request(user_prompt=blank)


@pytest.mark.parametrize("blank", ["", "   ", "\n"])
def test_a_blank_system_prompt_is_refused(blank: str) -> None:
    with pytest.raises(ValueError, match="system prompt"):
        _request(system_prompt=blank)


def test_a_round_without_tools_says_so() -> None:
    assert _request().sends_tools is False
    assert _request(tools=(ToolSpec(name="lookup", description="d"),)).sends_tools is True


def test_a_failed_round_reports_no_latency() -> None:
    failed = CallResult(
        answer="",
        framework="openai",
        model_id="m",
        usage=TokenUsage(),
        latency_ms=None,
        error="the provider refused the request",
    )
    assert failed.latency_ms is None
    assert failed.failed is True


def test_a_result_reuses_token_usage_and_keeps_the_trace_in_order() -> None:
    result = CallResult(
        answer="done",
        framework="openai",
        model_id="m",
        usage=TokenUsage(tokens_in=10, tokens_out=4),
        latency_ms=12.5,
        tool_calls=(
            ToolInvocation(name="first", arguments={}, order=0),
            ToolInvocation(name="second", arguments={}, order=1),
        ),
        model_calls=2,
    )
    assert isinstance(result.usage, TokenUsage)
    assert result.usage.tokens_in == 10
    assert [call.name for call in result.tool_calls] == ["first", "second"]
    assert result.model_calls == 2
    assert result.failed is False
