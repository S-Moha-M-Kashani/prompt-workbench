"""The Anthropic adapter, against an injected fake client. Its own credential."""

import pytest

pytest.importorskip("anthropic", reason="needs the anthropic extra")

from prompt_workbench.llm_call.anthropic_call import AnthropicCall  # noqa: E402
from prompt_workbench.models import CallRequest, ModelSettings, ToolSpec  # noqa: E402
from prompt_workbench.services import anthropic_catalog  # noqa: E402

LOOKUP = ToolSpec(name="lookup", description="Look a customer up.")


class Block:
    def __init__(self, **fields: object) -> None:
        self.__dict__.update(fields)


class Usage:
    def __init__(self, tokens_in: int, tokens_out: int) -> None:
        self.input_tokens = tokens_in
        self.output_tokens = tokens_out


class Response:
    def __init__(self, content: list[Block], usage: Usage, stop_reason: str) -> None:
        self.content = content
        self.usage = usage
        self.stop_reason = stop_reason


class FakeMessages:
    def __init__(self, responses: list[Response]) -> None:
        self._responses = responses
        self.calls: list[dict] = []

    def create(self, **kwargs: object) -> Response:
        self.calls.append(kwargs)
        return self._responses.pop(0)


class FakeClient:
    def __init__(self, *responses: Response) -> None:
        self.messages = FakeMessages(list(responses))

    @property
    def calls(self) -> list[dict]:
        return self.messages.calls


def _text(body: str, tokens: tuple[int, int] = (10, 4)) -> Response:
    return Response([Block(type="text", text=body)], Usage(*tokens), "end_turn")


def _asks_for(name: str, args: dict, tokens: tuple[int, int] = (20, 6)) -> Response:
    return Response(
        [Block(type="tool_use", id="tu_1", name=name, input=args)],
        Usage(*tokens),
        "tool_use",
    )


def _request(**overrides: object) -> CallRequest:
    fields: dict[str, object] = {
        "system_prompt": "Be brief.",
        "user_prompt": "Who is customer 7?",
        "model_id": "claude-opus-5",
        "settings": ModelSettings(temperature=0.2, max_tokens=512),
    }
    fields.update(overrides)
    return CallRequest(**fields)  # type: ignore[arg-type]


def test_a_round_without_tools_answers_and_reports_usage() -> None:
    client = FakeClient(_text("Ada Lovelace."))
    result = AnthropicCall(client=client).run(_request())

    assert result.answer == "Ada Lovelace."
    assert result.framework == "anthropic"
    assert result.usage.tokens_in == 10 and result.usage.tokens_out == 4
    assert result.model_calls == 1
    assert client.calls[0]["system"] == "Be brief."
    assert "tools" not in client.calls[0]


def test_a_tool_use_round_trip_is_recorded_and_both_calls_counted() -> None:
    client = FakeClient(_asks_for("lookup", {"id": 7}), _text("Ada Lovelace.", (30, 5)))
    result = AnthropicCall(client=client).run(_request(tools=(LOOKUP,)))

    assert [c.name for c in result.tool_calls] == ["lookup"]
    assert result.tool_calls[0].arguments == {"id": 7}
    assert result.model_calls == 2
    assert result.usage.tokens_in == 50 and result.usage.tokens_out == 11
    assert client.calls[0]["tools"][0]["name"] == "lookup"
    assert "input_schema" in client.calls[0]["tools"][0]


def test_the_tool_result_is_returned_under_the_id_that_asked_for_it() -> None:
    client = FakeClient(_asks_for("lookup", {}), _text("done"))
    AnthropicCall(client=client).run(_request(tools=(LOOKUP,)))
    reply = client.calls[1]["messages"][-1]
    assert reply["role"] == "user"
    assert reply["content"][0]["tool_use_id"] == "tu_1"
    assert reply["content"][0]["content"] == "tool lookup was called"


def test_max_tokens_is_always_sent_because_the_api_requires_it() -> None:
    client = FakeClient(_text("hi"))
    AnthropicCall(client=client).run(_request(settings=ModelSettings()))
    assert client.calls[0]["max_tokens"] > 0


def test_no_ambient_credential_is_ever_read(monkeypatch: pytest.MonkeyPatch) -> None:
    """A key in the environment must not silently become the one that is used.

    The SDK would fall back to it if the client were built without one, so the
    adapter refuses instead — and `run()` reports that as a failed round rather
    than raising, like every other framework failure.
    """
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-should-never-be-read")
    call = AnthropicCall()
    with pytest.raises(ValueError, match="explicitly"):
        _ = call.client

    result = call.run(_request())
    assert result.failed is True
    assert "explicitly" in (result.error or "")
    assert result.latency_ms is None


def test_an_empty_reply_is_a_failure() -> None:
    client = FakeClient(Response([], Usage(1, 0), "end_turn"))
    result = AnthropicCall(client=client).run(_request())
    assert result.failed is True
    assert result.latency_ms is None


# --- a separate catalogue, never merged into the provider list ------------


def test_the_anthropic_models_are_their_own_catalogue() -> None:
    entries = anthropic_catalog.all_models()
    assert entries
    assert all(entry.id.startswith("claude-") for entry in entries)


def test_its_prices_are_its_own_and_it_says_they_may_be_stale() -> None:
    entry = anthropic_catalog.get("claude-opus-5")
    assert entry.price_in_per_million > 0
    assert entry.price_out_per_million > entry.price_in_per_million
    assert anthropic_catalog.staleness_note()
    assert anthropic_catalog.AS_OF


def test_the_catalogues_are_never_merged() -> None:
    """A model priced from the wrong provider's list is a fabricated number."""
    from prompt_workbench.services.model_registry import ModelRegistry

    provider = ModelRegistry(fetch=lambda: {"data": []})
    for entry in anthropic_catalog.all_models():
        assert provider.find(entry.id) is None, entry.id


def test_the_catalogue_is_labelled_as_a_second_provider() -> None:
    assert "second" in anthropic_catalog.CATALOGUE_NOTE.lower() or (
        "separate" in anthropic_catalog.CATALOGUE_NOTE.lower()
    )
    assert "credential" in anthropic_catalog.CATALOGUE_NOTE.lower()
