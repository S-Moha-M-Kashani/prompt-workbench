"""The only module that talks to the model provider.

Uses the OpenAI SDK pointed at an OpenAI-compatible endpoint (OpenRouter by
default). All network access lives here so the rest of the app stays pure and
testable.

Credentials are session-scoped by construction: nothing is captured at import
time, ``.env`` values are read into a returned object rather than pushed into
``os.environ``, and every call takes an explicitly built ``client``. There is no
module-level fallback key, so a caller cannot accidentally reach the provider
with someone else's credentials.
"""

from __future__ import annotations

import os
from collections.abc import Iterator, Mapping
from dataclasses import asdict, dataclass
from typing import Any

from dotenv import dotenv_values
from openai import OpenAI

from prompt_workbench.models import ModelSettings
from prompt_workbench.models.protocols import Message
from prompt_workbench.services import model_catalog

API_KEY_ENV = "PROVIDER_API_KEY"
BASE_URL_ENV = "PROVIDER_BASE_URL"
DEFAULT_MODEL_ENV = "DEFAULT_MODEL"

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"


@dataclass(frozen=True)
class ProviderConfig:
    """One session's provider access: the key, the endpoint, the default model.

    Held by the caller (the Streamlit session) and passed in explicitly, which
    keeps two sessions on one server from sharing a key.
    """

    api_key: str = ""
    base_url: str = DEFAULT_BASE_URL
    default_model: str | None = None

    @property
    def has_api_key(self) -> bool:
        return bool(self.api_key.strip())


def config_from_env(
    *,
    env: Mapping[str, str] | None = None,
    env_file: str | None = ".env",
) -> ProviderConfig:
    """Build a config from the environment and an optional ``.env`` file.

    Real environment variables win over file values. ``env`` and ``env_file``
    are injectable so tests never depend on the developer's machine, and the
    file is read with ``dotenv_values`` — it returns a mapping instead of
    mutating ``os.environ``, so a key never leaks into process-global state.
    """
    environ: Mapping[str, str] = os.environ if env is None else env
    file_values: dict[str, str] = {}
    if env_file:
        file_values = {k: v for k, v in dotenv_values(env_file).items() if v is not None}

    def value(name: str) -> str:
        return (environ.get(name) or file_values.get(name) or "").strip()

    return ProviderConfig(
        api_key=value(API_KEY_ENV),
        base_url=value(BASE_URL_ENV) or DEFAULT_BASE_URL,
        default_model=value(DEFAULT_MODEL_ENV) or None,
    )


def build_client(config: ProviderConfig) -> OpenAI:
    """Return a provider client for ``config``; raises when no key is available."""
    if not config.has_api_key:
        raise RuntimeError(f"No provider API key provided (expected {API_KEY_ENV})")
    return OpenAI(base_url=config.base_url, api_key=config.api_key.strip())


def _create_params(
    model: str | None, messages: list[Message], settings: ModelSettings | None
) -> dict[str, Any]:
    """Build kwargs for ``create``; send only set fields the model honours.

    ``ModelSettings`` field names match the provider parameters, so a ``None``
    field is simply omitted and the model keeps its own default. Fields the
    catalog says ``model`` ignores are dropped too, so the request only ever
    asks for what will actually take effect.
    """
    if not model:
        raise ValueError("No model selected for this call")

    params: dict[str, Any] = {"model": model, "messages": messages}
    if settings is not None:
        params.update(
            {
                k: v
                for k, v in asdict(settings).items()
                if v is not None and model_catalog.supports(model, k)
            }
        )
    return params


def chat_completion(
    messages: list[Message],
    *,
    client: OpenAI,
    model: str | None = None,
    settings: ModelSettings | None = None,
    response_format: dict[str, Any] | None = None,
) -> str:
    """Send ``messages`` to the provider and return the assistant's text.

    ``client`` is required and injectable, so tests supply a fake and never
    touch the network. ``settings`` is an optional ``ModelSettings`` of sampling
    parameters; unset fields are not sent. ``response_format`` is an optional
    provider JSON-schema ``response_format`` dict requesting a structured reply;
    unlike the sampling knobs it is never filtered by the catalog, since it is a
    transport contract, not a sampling preference.
    """
    params = _create_params(model, messages, settings)
    if response_format is not None:
        params["response_format"] = response_format
    response = client.chat.completions.create(**params)
    return response.choices[0].message.content or ""


def chat_completion_stream(
    messages: list[Message],
    *,
    client: OpenAI,
    model: str | None = None,
    settings: ModelSettings | None = None,
) -> Iterator[str]:
    """Yield the assistant's reply as text chunks (server-sent stream).

    Same parameters as ``chat_completion``; a separate function because the
    return shape differs (iterator, not str) — callers choose the contract.
    """
    stream = client.chat.completions.create(
        **_create_params(model, messages, settings), stream=True
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content if chunk.choices else None
        if delta:
            yield delta


def chat_completion_with_usage(
    messages: list[Message],
    *,
    client: OpenAI,
    model: str | None = None,
    settings: ModelSettings | None = None,
) -> tuple[str, int]:
    """Like ``chat_completion`` but also return the total token count.

    Returns ``(text, total_tokens)``; ``total_tokens`` is 0 when the provider
    does not report usage.
    """
    response = client.chat.completions.create(**_create_params(model, messages, settings))
    text = response.choices[0].message.content or ""
    usage = getattr(response, "usage", None)
    total_tokens = getattr(usage, "total_tokens", 0) or 0
    return text, total_tokens
