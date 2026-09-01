"""Session state, and the two callables the UI needs from the outside world.

Streamlit re-runs the whole script on every interaction, so anything that must
survive a click lives in ``st.session_state``. This module is the only place
that reaches into it, which keeps the workspace a plain tested object everywhere
else.

Credentials never leave this session. The provider config is built once per
browser session and passed explicitly into a client that is created per call, so
two people using one server never share a key and nothing is written to the
process environment.
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from prompt_workbench.core.chat_memory import ThreadStore
from prompt_workbench.core.prompt_registry import load_system_prompt
from prompt_workbench.core.workspace import Workspace
from prompt_workbench.models.model_settings import ModelSettings
from prompt_workbench.models.protocols import CompletionFn, Message
from prompt_workbench.services import judges, model_catalog, openrouter_client
from prompt_workbench.services.metric_adapters import JudgeFn
from prompt_workbench.services.openrouter_client import ProviderConfig


def _state() -> Any:
    return st.session_state


def workspace() -> Workspace:
    if "workspace" not in _state():
        _state().workspace = Workspace()
    return _state().workspace


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


def platform_instruction() -> str:
    """The workbench's own prompt-engineer instruction, as the user has it."""
    if "platform_instruction" not in _state():
        _state().platform_instruction = load_system_prompt("platform_instruction")
    return _state().platform_instruction


def set_platform_instruction(text: str) -> None:
    if text != _state().get("platform_instruction"):
        _state().platform_instruction = text
        workspace().platform_instruction_revision += 1


def selected_model() -> str:
    return provider_config().default_model or model_catalog.DEFAULT_MODEL_ID


def model_settings() -> ModelSettings:
    if "model_settings" not in _state():
        _state().model_settings = ModelSettings()
    return _state().model_settings


def set_model_settings(settings: ModelSettings) -> None:
    _state().model_settings = settings


def has_credentials() -> bool:
    return provider_config().has_api_key


def completion() -> CompletionFn:
    """A provider call bound to this session's credentials.

    Built per call rather than cached: a client holds the key, and a cached one
    would outlive an edit to it in the sidebar.
    """
    config = provider_config()
    client = openrouter_client.build_client(config)

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
    """The configured judge, with a client only if the provider backend needs one."""
    backend = judge_backend()
    client = (
        openrouter_client.build_client(provider_config())
        if backend == judges.PROVIDER_BACKEND and has_credentials()
        else None
    )
    return judges.build_judge(backend, model=judge_model(), client=client)
