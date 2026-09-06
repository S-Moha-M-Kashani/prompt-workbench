"""Streamlit session state, and the callables the UI needs from outside.

Streamlit re-runs the script on every interaction, so anything that must survive
a click lives in ``st.session_state``. This is the only module that touches it,
which keeps the workbench session a plain tested object everywhere else.

Credentials never leave this browser session: the config is built once and
passed explicitly into a client created per call.
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from prompt_workbench.core.session import Session
from prompt_workbench.llm_call import registry as frameworks
from prompt_workbench.models.model_settings import ModelSettings
from prompt_workbench.models.protocols import (
    CallRunner,
    CompletionFn,
    CompletionWithUsageFn,
    Message,
)
from prompt_workbench.models.usage import TokenUsage
from prompt_workbench.services import deepeval_judge, model_registry, openrouter_client
from prompt_workbench.services.openrouter_client import ProviderConfig

# The model the workbench itself uses to write cases and variants. Distinct from
# the models under test: a weak writer produces weak variants and makes every
# comparison downstream noisier.
AUTHORING_MODEL = "openai/gpt-4o-mini"


def _state() -> Any:
    return st.session_state


@st.cache_data(ttl=3600, show_spinner="Fetching current model prices…")
def _cached_model_list() -> dict:
    """The provider's model list, at most once an hour across all sessions.

    Cached and spinnered because it happens during the first render: without
    this the page sits blank while a network call completes, which reads as a
    broken app rather than a slow one. Prices do not move hourly, so an hour is
    a generous freshness bar for a number used to order a list.

    A failure propagates rather than being cached, and the registry falls back
    to the committed snapshot.
    """
    return model_registry.fetch_live()


def registry() -> model_registry.ModelRegistry:
    """One catalogue per browser session, over an hourly shared fetch."""
    if "registry" not in _state():
        _state().registry = model_registry.ModelRegistry(fetch=_cached_model_list)
    return _state().registry


def workbench() -> Session:
    if "workbench" not in _state():
        _state().workbench = Session(registry=registry())
    return _state().workbench


def provider_config() -> ProviderConfig:
    if "provider_config" not in _state():
        _state().provider_config = openrouter_client.config_from_env()
    return _state().provider_config


def set_provider_config(config: ProviderConfig) -> None:
    _state().provider_config = config


def has_credentials() -> bool:
    return provider_config().has_api_key


def authoring_model() -> str:
    return provider_config().default_model or AUTHORING_MODEL


def completion() -> CompletionFn:
    """A provider call bound to this session's credentials."""
    client = openrouter_client.build_client(provider_config())

    def complete(
        messages: list[Message],
        *,
        model: str | None = None,
        settings: ModelSettings | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> str:
        return openrouter_client.chat_completion(
            messages,
            client=client,
            model=model or authoring_model(),
            settings=settings,
            response_format=response_format,
        )

    return complete


def completion_with_usage() -> CompletionWithUsageFn:
    """A provider call that also reports what it cost."""
    client = openrouter_client.build_client(provider_config())

    def complete(
        messages: list[Message],
        *,
        model: str | None = None,
        settings: ModelSettings | None = None,
    ) -> tuple[str, TokenUsage]:
        return openrouter_client.chat_completion_with_usage(
            messages, client=client, model=model or authoring_model(), settings=settings
        )

    return complete


def judge_backend() -> str:
    if "judge_backend" not in _state():
        _state().judge_backend = deepeval_judge.available_backends()[0]
    return _state().judge_backend


def set_judge_backend(backend: str) -> None:
    _state().judge_backend = backend


def judge_model() -> str:
    backend = judge_backend()
    key = f"judge_model_{backend}"
    if key not in _state():
        _state()[key] = deepeval_judge.models_for(backend)[0]
    return _state()[key]


def set_judge_model(model: str) -> None:
    _state()[f"judge_model_{judge_backend()}"] = model


def judge() -> Any:
    """The configured deepeval judge model."""
    return deepeval_judge.build(
        judge_backend(), model=judge_model(), config=provider_config()
    )


def anthropic_key() -> str:
    """A second provider means a second session-scoped key, never an ambient one."""
    return str(_state().get("anthropic_api_key", ""))


def set_anthropic_key(key: str) -> None:
    _state().anthropic_api_key = key


def call_runner(framework_key: str) -> CallRunner:
    """A framework adapter bound to this session's credentials.

    Every adapter takes ``api_key`` and ``base_url`` the same way, so this needs
    no per-framework branch for the workbench's own provider. A framework that
    reaches a *different* provider gets that provider's own key — passed
    explicitly here, as everywhere else. Tests replace this with a fake.
    """
    entry = frameworks.get(framework_key)
    if entry.reaches_own_provider:
        return entry.build(api_key=anthropic_key())
    config = provider_config()
    return entry.build(api_key=config.api_key, base_url=config.base_url)
