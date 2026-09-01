"""Provider access, the settings for a run, and the judge.

The settings sit here rather than beside the work, so the page itself stays the
sequence the user is walking. A model that ignores a knob shows it disabled
rather than hidden: "this model drops temperature" is a more useful thing to
read than a control that quietly is not there, and the panel keeps its shape as
models change.
"""

from __future__ import annotations

import streamlit as st

from prompt_workbench.core.session import Session
from prompt_workbench.models.model_settings import ModelSettings
from prompt_workbench.services import (
    codex_cli_client,
    deepeval_judge,
    deepeval_metrics,
    model_registry,
    openrouter_client,
)
from prompt_workbench.ui import session


def render() -> None:
    workbench = session.workbench()
    registry = session.registry()

    st.header("Provider")
    config = session.provider_config()
    api_key = st.text_input(
        "API key",
        value=config.api_key,
        type="password",
        help="Kept in this browser session only. Never stored on the server.",
    )

    # The authoring model writes cases and variants, so it defaults to a capable
    # one rather than the cheapest thing available: weak variants make every
    # comparison downstream noisier for no saving worth having.
    shortlist = registry.recommended()
    model_ids = [entry.id for entry in shortlist]
    default = config.default_model if config.default_model in model_ids else session.AUTHORING_MODEL
    index = model_ids.index(default) if default in model_ids else 0
    model_id = st.selectbox(
        "Authoring model",
        model_ids,
        index=index,
        help="Writes your test cases and prompt variants. Not the model under test.",
    )
    entry = registry.find(model_id)
    if entry is not None and entry.has_price:
        st.caption(
            f"${entry.price_in_per_million:.2f} in / "
            f"${entry.price_out_per_million:.2f} out per 1M tokens"
        )
    if registry.is_stale:
        st.warning("Using the bundled model snapshot — prices may be out of date.")

    session.set_provider_config(
        openrouter_client.ProviderConfig(
            api_key=api_key, base_url=config.base_url, default_model=model_id
        )
    )

    _render_settings(workbench, registry, model_id)
    st.divider()
    _render_judge()


def _render_settings(
    workbench: Session, registry: model_registry.ModelRegistry, model_id: str
) -> None:
    st.subheader("Model settings")
    task = workbench.task_type
    if task is not None:
        st.caption(task.settings_note)
    current = workbench.settings

    def honoured(name: str) -> bool:
        return registry.supports(model_id, name)

    temperature = st.slider(
        "temperature", 0.0, 2.0,
        current.temperature if current.temperature is not None else 1.0, 0.05,
        disabled=not honoured("temperature"),
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
        "max_tokens", min_value=0, max_value=32_000, value=current.max_tokens or 0, step=64,
        help="0 leaves the model's own default.",
        disabled=not honoured("max_tokens"),
    )

    # Only honoured settings are stored, so a value left over from a previously
    # selected model cannot travel silently into the next request.
    workbench.settings = ModelSettings(
        temperature=temperature if honoured("temperature") else None,
        top_p=top_p if honoured("top_p") else None,
        frequency_penalty=frequency_penalty if honoured("frequency_penalty") else None,
        presence_penalty=presence_penalty if honoured("presence_penalty") else None,
        max_tokens=(int(max_tokens) or None) if honoured("max_tokens") else None,
    )

    dropped = [n for n in model_registry.SETTING_NAMES if not honoured(n)]
    if dropped:
        st.caption(
            "This model would ignore " + ", ".join(f"`{n}`" for n in dropped)
            + ", so they are disabled and never sent."
        )


def _render_judge() -> None:
    st.subheader("Judge")
    if not deepeval_metrics.is_available():
        st.warning(
            "The metric layer needs deepeval, an optional extra:\n\n"
            f"`{deepeval_metrics.INSTALL_HINT}`"
        )
        return

    backends = deepeval_judge.available_backends()
    backend = st.selectbox(
        "Backend", backends, format_func=lambda b: deepeval_judge.BACKEND_LABELS[b],
        key="judge_backend_pick",
    )
    session.set_judge_backend(backend)
    judge_model = st.selectbox(
        "Judge model", deepeval_judge.models_for(backend), key=f"judge_model_pick_{backend}"
    )
    session.set_judge_model(judge_model)

    note = deepeval_judge.fidelity_note(backend)
    if note:
        st.caption(note)
    if backend == deepeval_judge.CLI_BACKEND and not codex_cli_client.is_available():
        st.warning("`codex` is not installed on this machine.")
