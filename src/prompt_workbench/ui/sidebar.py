"""Provider access, model settings, and the judge. Nothing about the prompt.

The sampling settings live here rather than beside the chat, so the main screen
stays a use case, a prompt and a conversation — but they are on screen, because
how a model samples is part of what a prompt has to survive.

A model that ignores a knob shows it **disabled** rather than hidden. Hiding it
reads as "this workbench does not offer temperature"; disabling it, with the
reason underneath, reads as "this model drops it" — which is the true and more
useful statement. It also means switching models does not make controls appear
and vanish, so the panel keeps its shape.
"""

from __future__ import annotations

import streamlit as st

from prompt_workbench.models.model_settings import ModelSettings
from prompt_workbench.services import codex_cli_client, judges, model_catalog, openrouter_client
from prompt_workbench.ui import session


def render() -> None:
    st.header("Provider")
    config = session.provider_config()

    api_key = st.text_input(
        "API key",
        value=config.api_key,
        type="password",
        help="Kept in this browser session only. It is never stored on the server.",
    )
    model_ids = [model.id for model in model_catalog.all_models()]
    index = model_ids.index(config.default_model) if config.default_model in model_ids else 0
    model_id = st.selectbox("Model under test", model_ids, index=index)
    st.caption(model_catalog.get(model_id).note)

    session.set_provider_config(
        openrouter_client.ProviderConfig(
            api_key=api_key, base_url=config.base_url, default_model=model_id
        )
    )

    _render_model_settings(model_id)

    st.divider()
    st.subheader("Judge")
    backends = judges.available_backends()
    backend = st.selectbox(
        "Backend", backends, format_func=lambda b: judges.BACKEND_LABELS[b], key="judge_backend_pick"
    )
    session.set_judge_backend(backend)
    judge_model = st.selectbox(
        "Judge model", judges.models_for(backend), key=f"judge_model_pick_{backend}"
    )
    session.set_judge_model(judge_model)
    st.caption(
        ("`codex` found on this machine. " if codex_cli_client.is_available()
         else "`codex` not installed. ")
        + "Judging with a model other than the one under test keeps a prompt "
        "from being graded by the model that wrote its output."
    )


def _render_model_settings(model_id: str) -> None:
    """The sampling knobs, disabled where this model would drop them.

    ``model_catalog`` is the single source of truth for what each model honours,
    so the disabled state and the request filtering cannot disagree — the client
    strips the same settings this panel greys out.
    """
    st.subheader("Model settings")
    current = session.model_settings()

    def honoured(name: str) -> bool:
        return model_catalog.supports(model_id, name)

    temperature = st.slider(
        "temperature", 0.0, 2.0, current.temperature if current.temperature is not None else 1.0,
        0.05, disabled=not honoured("temperature"),
    )
    top_p = st.slider(
        "top_p", 0.0, 1.0, current.top_p if current.top_p is not None else 1.0, 0.05,
        disabled=not honoured("top_p"),
    )
    frequency_penalty = st.slider(
        "frequency_penalty", -2.0, 2.0,
        current.frequency_penalty if current.frequency_penalty is not None else 0.0, 0.1,
        disabled=not honoured("frequency_penalty"),
    )
    presence_penalty = st.slider(
        "presence_penalty", -2.0, 2.0,
        current.presence_penalty if current.presence_penalty is not None else 0.0, 0.1,
        disabled=not honoured("presence_penalty"),
    )
    max_tokens = st.number_input(
        "max_tokens", min_value=0, max_value=32_000,
        value=current.max_tokens or 0, step=64,
        help="0 leaves the model's own default.",
        disabled=not honoured("max_tokens"),
    )

    # Only settings this model honours are stored, so a value left over from a
    # previously selected model cannot travel silently into the next request.
    session.set_model_settings(
        ModelSettings(
            temperature=temperature if honoured("temperature") else None,
            top_p=top_p if honoured("top_p") else None,
            frequency_penalty=frequency_penalty if honoured("frequency_penalty") else None,
            presence_penalty=presence_penalty if honoured("presence_penalty") else None,
            max_tokens=(int(max_tokens) or None) if honoured("max_tokens") else None,
        )
    )

    dropped = [name for name in model_catalog.SETTING_NAMES if not honoured(name)]
    if dropped:
        st.caption(
            "This model would ignore " + ", ".join(f"`{name}`" for name in dropped)
            + ", so they are disabled and never sent."
        )
