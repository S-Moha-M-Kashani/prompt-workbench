"""Streamlit session state, and the two callables the UI needs from outside.

Streamlit re-runs the whole script on every interaction, so anything that must
survive a click lives in ``st.session_state``. This module is the only place
that reaches into it, which keeps the workbench session a plain tested object
everywhere else.

Credentials never leave this browser session: the config is built once and
passed explicitly into a client created per call, so two people on one server
never share a key and nothing is written to the process environment.
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from prompt_workbench.core.chat_memory import ThreadStore
from prompt_workbench.core.session import Session
from prompt_workbench.models.model_settings import ModelSettings
from prompt_workbench.models.protocols import CompletionFn, Message
from prompt_workbench.services import judges, model_catalog, openrouter_client
from prompt_workbench.services.metric_adapters import JudgeFn
from prompt_workbench.services.openrouter_client import ProviderConfig


def _state() -> Any:
    return st.session_state


def workbench() -> Session:
    if "workbench" not in _state():
        _state().workbench = Session()
    return _state().workbench


def threads() -> ThreadStore:
    if "threads" not in _state():
        _state().threads = ThreadStore()
    return _state().threads


def provider_config() -> ProviderConfig:
    if "provider_config" not in _state():
        _state().provider_config = openrouter_client.config_from_env()
    return _state().provider_config


def set_provider_config(config: ProviderConfig) -> None:
    _state().provider_config = config


def selected_model() -> str:
    return provider_config().default_model or model_catalog.DEFAULT_MODEL_ID


def has_credentials() -> bool:
    return provider_config().has_api_key


def completion() -> CompletionFn:
    """A provider call bound to this session's credentials.

    Built per call rather than cached: a client holds the key, and a cached one
    would outlive an edit to it in the sidebar.
    """
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
            model=model or selected_model(),
            settings=settings,
            response_format=response_format,
        )

    return complete


def judge_backend() -> str:
    if "judge_backend" not in _state():
        _state().judge_backend = judges.available_backends()[0]
    return _state().judge_backend


def set_judge_backend(backend: str) -> None:
    _state().judge_backend = backend


def judge_model() -> str:
    backend = judge_backend()
    key = f"judge_model_{backend}"
    if key not in _state():
        _state()[key] = judges.default_model_for(backend)
    return _state()[key]


def set_judge_model(model: str) -> None:
    _state()[f"judge_model_{judge_backend()}"] = model


def judge() -> JudgeFn:
    """The configured judge, given a client only if the provider backend needs one."""
    backend = judge_backend()
    client = (
        openrouter_client.build_client(provider_config())
        if backend == judges.PROVIDER_BACKEND and has_credentials()
        else None
    )
    return judges.build_judge(backend, model=judge_model(), client=client)
