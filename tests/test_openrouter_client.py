"""Provider-boundary tests. No live calls: the client is always a fake."""

import os
from types import SimpleNamespace

import pytest

from prompt_workbench.models import ModelSettings
from prompt_workbench.services import openrouter_client
from prompt_workbench.services.openrouter_client import ProviderConfig

_TUNABLE_MODEL = "openai/gpt-4o-mini"
_SAMPLING_IGNORING_MODEL = "openai/gpt-5-mini"


class FakeCompletions:
    """Records the kwargs it was called with and returns a canned reply."""

    def __init__(self, text: str = "ok", total_tokens: int | None = 42, chunks=None):
        self.calls: list[dict] = []
        self._text = text
        self._total_tokens = total_tokens
        self._chunks = chunks

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs.get("stream"):
            return iter(
                SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=c))])
                for c in (self._chunks or [])
            )
        usage = (
            None if self._total_tokens is None else SimpleNamespace(total_tokens=self._total_tokens)
        )
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self._text))],
            usage=usage,
        )


def fake_client(**kwargs) -> SimpleNamespace:
    completions = FakeCompletions(**kwargs)
    return SimpleNamespace(chat=SimpleNamespace(completions=completions), completions=completions)


# --- configuration ---------------------------------------------------------


def test_config_from_env_reads_the_three_provider_variables():
    config = openrouter_client.config_from_env(
        env={
            "PROVIDER_API_KEY": " secret ",
            "PROVIDER_BASE_URL": "https://example.test/v1",
            "DEFAULT_MODEL": _TUNABLE_MODEL,
        },
        env_file=None,
    )

    assert config == ProviderConfig(
        api_key="secret", base_url="https://example.test/v1", default_model=_TUNABLE_MODEL
    )


def test_config_from_env_falls_back_to_the_default_base_url():
    config = openrouter_client.config_from_env(env={}, env_file=None)

    assert config.base_url == openrouter_client.DEFAULT_BASE_URL
    assert config.default_model is None
    assert config.has_api_key is False


def test_env_file_supplies_values_and_the_real_environment_wins(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "PROVIDER_API_KEY=from-file\nDEFAULT_MODEL=openai/gpt-4o\n", encoding="utf-8"
    )

    config = openrouter_client.config_from_env(
        env={"PROVIDER_API_KEY": "from-environment"}, env_file=str(env_file)
    )

    assert config.api_key == "from-environment"
    assert config.default_model == "openai/gpt-4o"


def test_reading_an_env_file_does_not_leak_the_key_into_the_process(tmp_path):
    """Credentials stay session-scoped: os.environ must be untouched."""
    env_file = tmp_path / ".env"
    env_file.write_text("PROVIDER_API_KEY=must-not-leak\n", encoding="utf-8")

    openrouter_client.config_from_env(env={}, env_file=str(env_file))

    assert os.environ.get("PROVIDER_API_KEY") != "must-not-leak"


def test_the_module_holds_no_key_at_import_time():
    """There is no module-level fallback credential to accidentally reuse."""
    module_strings = [
        value for value in vars(openrouter_client).values() if isinstance(value, str)
    ]

    assert all("sk-" not in value for value in module_strings)


def test_build_client_without_a_key_raises():
    with pytest.raises(RuntimeError, match="PROVIDER_API_KEY"):
        openrouter_client.build_client(ProviderConfig())


# --- calls -----------------------------------------------------------------


def test_chat_completion_returns_the_assistant_text():
    client = fake_client(text="hello")

    result = openrouter_client.chat_completion(
        [{"role": "user", "content": "hi"}], client=client, model=_TUNABLE_MODEL
    )

    assert result == "hello"
    assert client.completions.calls[0]["model"] == _TUNABLE_MODEL


def test_chat_completion_without_a_model_raises_before_calling():
    client = fake_client()

    with pytest.raises(ValueError, match="No model selected"):
        openrouter_client.chat_completion([{"role": "user", "content": "hi"}], client=client)

    assert client.completions.calls == []


def test_unset_settings_are_not_sent():
    client = fake_client()

    openrouter_client.chat_completion(
        [{"role": "user", "content": "hi"}],
        client=client,
        model=_TUNABLE_MODEL,
        settings=ModelSettings(temperature=0.2),
    )

    sent = client.completions.calls[0]
    assert sent["temperature"] == 0.2
    assert "top_p" not in sent


def test_settings_the_model_ignores_are_dropped():
    """Asking for a knob the model discards would only invite false conclusions."""
    client = fake_client()

    openrouter_client.chat_completion(
        [{"role": "user", "content": "hi"}],
        client=client,
        model=_SAMPLING_IGNORING_MODEL,
        settings=ModelSettings(temperature=0.9, max_tokens=128),
    )

    sent = client.completions.calls[0]
    assert "temperature" not in sent
    assert sent["max_tokens"] == 128


def test_response_format_is_passed_through_untouched():
    client = fake_client()
    schema = {"type": "json_schema", "json_schema": {"name": "x", "schema": {}}}

    openrouter_client.chat_completion(
        [{"role": "user", "content": "hi"}],
        client=client,
        model=_SAMPLING_IGNORING_MODEL,
        response_format=schema,
    )

    assert client.completions.calls[0]["response_format"] == schema


def test_missing_content_becomes_an_empty_string():
    client = fake_client(text=None)

    assert (
        openrouter_client.chat_completion(
            [{"role": "user", "content": "hi"}], client=client, model=_TUNABLE_MODEL
        )
        == ""
    )


def test_chat_completion_with_usage_reports_total_tokens():
    client = fake_client(text="hi", total_tokens=17)

    assert openrouter_client.chat_completion_with_usage(
        [{"role": "user", "content": "hi"}], client=client, model=_TUNABLE_MODEL
    ) == ("hi", 17)


def test_chat_completion_with_usage_defaults_to_zero_tokens():
    client = fake_client(text="hi", total_tokens=None)

    assert openrouter_client.chat_completion_with_usage(
        [{"role": "user", "content": "hi"}], client=client, model=_TUNABLE_MODEL
    ) == ("hi", 0)


def test_stream_yields_only_non_empty_chunks():
    client = fake_client(chunks=["a", None, "b"])

    chunks = list(
        openrouter_client.chat_completion_stream(
            [{"role": "user", "content": "hi"}], client=client, model=_TUNABLE_MODEL
        )
    )

    assert chunks == ["a", "b"]
    assert client.completions.calls[0]["stream"] is True
