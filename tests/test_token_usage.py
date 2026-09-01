"""Token accounting: what a run cost, reported in and out separately."""

import pytest

from prompt_workbench.core import one_shot
from prompt_workbench.models import ModelSettings, TokenUsage
from prompt_workbench.services import model_registry, openrouter_client



class FakeUsage:
    def __init__(self, prompt_tokens: int, completion_tokens: int) -> None:
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = prompt_tokens + completion_tokens


class FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class FakeChoice:
    def __init__(self, content: str) -> None:
        self.message = FakeMessage(content)


class FakeResponse:
    def __init__(self, content: str, usage: FakeUsage | None) -> None:
        self.choices = [FakeChoice(content)]
        self.usage = usage


class FakeClient:
    """Stands in for the provider SDK; records what it was asked for."""

    def __init__(self, response: FakeResponse) -> None:
        self._response = response
        self.calls: list[dict] = []
        self.chat = self  # type: ignore[assignment]
        self.completions = self  # type: ignore[assignment]

    def create(self, **kwargs: object) -> FakeResponse:
        self.calls.append(kwargs)
        return self._response


# --- the model ------------------------------------------------------------


def test_usage_totals_the_two_directions() -> None:
    usage = TokenUsage(tokens_in=120, tokens_out=30)
    assert usage.total == 150


def test_usage_defaults_to_zero_when_a_provider_reports_nothing() -> None:
    assert TokenUsage().total == 0


# --- the provider boundary ------------------------------------------------


def test_the_client_reports_both_directions_separately() -> None:
    client = FakeClient(FakeResponse("hello", FakeUsage(prompt_tokens=120, completion_tokens=30)))
    text, usage = openrouter_client.chat_completion_with_usage(
        [{"role": "user", "content": "hi"}], client=client, model="openai/gpt-4o-mini"
    )
    assert text == "hello"
    assert (usage.tokens_in, usage.tokens_out) == (120, 30)


def test_a_provider_that_reports_no_usage_yields_zeros_rather_than_failing() -> None:
    client = FakeClient(FakeResponse("hello", None))
    _text, usage = openrouter_client.chat_completion_with_usage(
        [{"role": "user", "content": "hi"}], client=client, model="openai/gpt-4o-mini"
    )
    assert usage == TokenUsage()


def test_only_the_settings_that_were_set_are_sent() -> None:
    """Capability filtering lives in the registry and the interface, not here.
    A second filter in the client would be a second source of truth, and the one
    most likely to go stale since nobody looks at it."""
    client = FakeClient(FakeResponse("hi", FakeUsage(1, 1)))
    openrouter_client.chat_completion_with_usage(
        [{"role": "user", "content": "hi"}],
        client=client,
        model="openai/gpt-4o-mini",
        settings=ModelSettings(temperature=0.7, max_tokens=100),
    )
    sent = client.calls[0]
    assert sent["temperature"] == 0.7
    assert sent["max_tokens"] == 100
    assert "top_p" not in sent, "an unset field keeps the model's own default"


# --- what a call reports -------------------------------------------------


def test_a_call_reports_the_tokens_it_cost() -> None:
    def complete(messages, *, model=None, settings=None):  # type: ignore[no-untyped-def]
        return "a response", TokenUsage(tokens_in=200, tokens_out=45)

    text, usage = one_shot.run_plain(
        system_prompt="You are a classifier.",
        user_message="what do you know?",
        model="openai/gpt-4o-mini",
        settings=ModelSettings(),
        complete=complete,
    )
    assert text == "a response"
    assert (usage.tokens_in, usage.tokens_out, usage.total) == (200, 45, 245)


def test_the_settings_reach_the_call() -> None:
    seen: dict = {}

    def complete(messages, *, model=None, settings=None):  # type: ignore[no-untyped-def]
        seen["settings"] = settings
        return "ok", TokenUsage()

    one_shot.run_plain(
        system_prompt="p", user_message="hi", model="openai/gpt-4o-mini",
        settings=ModelSettings(temperature=0.2, max_tokens=500), complete=complete,
    )
    assert seen["settings"].temperature == 0.2
    assert seen["settings"].max_tokens == 500


def test_a_call_with_no_reported_usage_still_returns_its_text() -> None:
    def complete(messages, *, model=None, settings=None):  # type: ignore[no-untyped-def]
        return "ok", TokenUsage()

    text, usage = one_shot.run_plain(
        system_prompt="p", user_message="hi", model="m", settings=None, complete=complete,
    )
    assert text == "ok"
    assert usage.total == 0


def test_an_empty_response_is_refused_rather_than_recorded() -> None:
    def complete(messages, *, model=None, settings=None):  # type: ignore[no-untyped-def]
        return "   ", TokenUsage()

    with pytest.raises(one_shot.RunFailed, match="empty response"):
        one_shot.run_plain(
            system_prompt="p", user_message="hi", model="m", settings=None, complete=complete,
        )


# --- which knobs a model honours -----------------------------------------


def _registry() -> model_registry.ModelRegistry:
    """The snapshot, which is what the suite reads — never the live list."""
    return model_registry.ModelRegistry(
        fetch=lambda: (_ for _ in ()).throw(RuntimeError("offline"))
    )


def test_the_registry_says_which_settings_a_reasoning_model_drops() -> None:
    registry = _registry()
    assert not registry.supports("openai/gpt-5-mini", "temperature")
    assert registry.supports("openai/gpt-5-mini", "max_tokens")
    assert registry.supports("openai/gpt-4o-mini", "temperature")


def test_ignored_settings_names_exactly_what_would_be_dropped() -> None:
    ignored = _registry().ignored_settings(
        "openai/gpt-5-mini", ModelSettings(temperature=0.7, top_p=0.9, max_tokens=100)
    )
    assert set(ignored) == {"temperature", "top_p"}
