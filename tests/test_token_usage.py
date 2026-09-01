"""Token accounting: what a run cost, reported in and out separately."""

from datetime import UTC, datetime

import pytest

from prompt_workbench.core import one_shot
from prompt_workbench.models import ModelSettings, PromptRun, TokenUsage, sequential_ids
from prompt_workbench.services import model_catalog, openrouter_client
from prompt_workbench.services.use_case_catalog import get as get_use_case

WHEN = datetime(2026, 3, 1, tzinfo=UTC)


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


def test_settings_a_model_ignores_are_not_sent_to_the_provider() -> None:
    """The catalog already knows; the request should never ask for a knob that
    would be silently dropped."""
    client = FakeClient(FakeResponse("hi", FakeUsage(1, 1)))
    openrouter_client.chat_completion_with_usage(
        [{"role": "user", "content": "hi"}],
        client=client,
        model="openai/gpt-5-mini",
        settings=ModelSettings(temperature=0.7, max_tokens=100),
    )
    sent = client.calls[0]
    assert "temperature" not in sent, "a reasoning model ignores sampling"
    assert sent["max_tokens"] == 100


# --- what a run records ---------------------------------------------------


def test_a_run_records_the_tokens_it_cost() -> None:
    def complete(messages, *, model=None, settings=None):  # type: ignore[no-untyped-def]
        return "a response", TokenUsage(tokens_in=200, tokens_out=45)

    record = one_shot.run_once(
        use_case=get_use_case("grounded_briefing"),
        prompt_under_test=get_use_case("grounded_briefing").system_prompt,
        prompt_revision=1,
        user_message="what do you know?",
        model="openai/gpt-4o-mini",
        settings=ModelSettings(),
        complete=complete,
        new_id=sequential_ids(),
        clock=lambda: WHEN,
    )
    assert record.usage.tokens_in == 200
    assert record.usage.tokens_out == 45
    assert record.usage.total == 245


def test_the_settings_reach_the_call() -> None:
    seen: dict = {}

    def complete(messages, *, model=None, settings=None):  # type: ignore[no-untyped-def]
        seen["settings"] = settings
        return "ok", TokenUsage()

    one_shot.run_once(
        use_case=get_use_case("grounded_briefing"),
        prompt_under_test=get_use_case("grounded_briefing").system_prompt,
        prompt_revision=1,
        user_message="hi",
        model="openai/gpt-4o-mini",
        settings=ModelSettings(temperature=0.2, max_tokens=500),
        complete=complete,
        new_id=sequential_ids(),
        clock=lambda: WHEN,
    )
    assert seen["settings"].temperature == 0.2
    assert seen["settings"].max_tokens == 500


def test_a_run_with_no_reported_usage_still_records() -> None:
    def complete(messages, *, model=None, settings=None):  # type: ignore[no-untyped-def]
        return "ok", TokenUsage()

    record = one_shot.run_once(
        use_case=get_use_case("grounded_briefing"),
        prompt_under_test=get_use_case("grounded_briefing").system_prompt,
        prompt_revision=1,
        user_message="hi",
        model="m",
        settings=None,
        complete=complete,
        new_id=sequential_ids(),
        clock=lambda: WHEN,
    )
    assert record.usage.total == 0
    assert record.is_scoreable


# --- which knobs a model honours -----------------------------------------


def test_the_catalog_says_which_settings_a_reasoning_model_drops() -> None:
    assert not model_catalog.supports("openai/gpt-5-mini", "temperature")
    assert model_catalog.supports("openai/gpt-5-mini", "max_tokens")
    assert model_catalog.supports("openai/gpt-4o-mini", "temperature")


def test_ignored_settings_names_exactly_what_would_be_dropped() -> None:
    ignored = model_catalog.ignored_settings(
        "openai/gpt-5-mini", ModelSettings(temperature=0.7, top_p=0.9, max_tokens=100)
    )
    assert set(ignored) == {"temperature", "top_p"}
