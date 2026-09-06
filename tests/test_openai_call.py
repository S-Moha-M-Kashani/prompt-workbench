"""The bare-SDK adapter, exercised against an injected fake client."""

import pytest

from prompt_workbench.llm_call.openai_call import OpenAiCall
from prompt_workbench.models import CallRequest, ModelSettings, OutputStructure, ToolSpec
from tests_support import FakeClient, FakeMessage, FakeResponse, answer, asks_for_tool

SHAPE = OutputStructure(name="verdict", schema={"type": "object", "properties": {}})
LOOKUP = ToolSpec(name="lookup", description="Look a customer up.")


def _request(**overrides: object) -> CallRequest:
    fields: dict[str, object] = {
        "system_prompt": "Be brief.",
        "user_prompt": "Who is customer 7?",
        "model_id": "openai/gpt-4o-mini",
        "settings": ModelSettings(temperature=0.2),
    }
    fields.update(overrides)
    return CallRequest(**fields)  # type: ignore[arg-type]


def test_a_round_without_tools_makes_one_call_and_sends_no_tool_surface() -> None:
    client = FakeClient(answer("Ada Lovelace."))
    result = OpenAiCall(client=client).run(_request())

    assert result.answer == "Ada Lovelace."
    assert result.framework == "openai"
    assert result.model_calls == 1
    assert result.usage.tokens_in == 10 and result.usage.tokens_out == 4
    assert result.tool_calls == ()
    assert "tools" not in client.calls[0]
    assert client.calls[0]["messages"][0] == {"role": "system", "content": "Be brief."}
    assert client.calls[0]["temperature"] == 0.2


def test_a_tool_round_trip_is_recorded_in_order_and_both_calls_are_counted() -> None:
    client = FakeClient(asks_for_tool("lookup"), answer("Ada Lovelace.", (30, 5)))
    result = OpenAiCall(client=client).run(_request(tools=(LOOKUP,)))

    assert [c.name for c in result.tool_calls] == ["lookup"]
    assert result.tool_calls[0].arguments == {"id": 7}
    assert result.model_calls == 2
    assert result.usage.tokens_in == 50 and result.usage.tokens_out == 11
    assert client.calls[0]["tools"][0]["function"]["name"] == "lookup"


def test_an_enforced_structure_is_sent_as_the_provider_parameter() -> None:
    client = FakeClient(answer("{}"))
    result = OpenAiCall(client=client).run(
        _request(output_structure=SHAPE, structure_is_enforced=True)
    )

    assert client.calls[0]["response_format"]["type"] == "json_schema"
    assert result.structure_was_enforced is True


def test_an_empty_reply_is_a_failure_not_an_empty_answer() -> None:
    client = FakeClient(FakeResponse(FakeMessage("", None), None))
    result = OpenAiCall(client=client).run(_request())
    assert result.failed is True
    assert result.latency_ms is None


def test_the_adapter_opens_no_socket(monkeypatch: pytest.MonkeyPatch) -> None:
    import socket

    def refuse(*args: object, **kwargs: object) -> None:
        raise AssertionError("the adapter must not open a socket")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    OpenAiCall(client=FakeClient(answer("fine"))).run(_request())
