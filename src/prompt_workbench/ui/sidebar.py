"""Provider settings and the workbench's own instruction.

The API key lives in this browser session and is passed explicitly into a client
per call. It is never written to the process environment, so two people using
one deployment cannot end up on each other's credentials.
"""

from __future__ import annotations

import streamlit as st

from prompt_workbench.core.workspace import Workspace
from prompt_workbench.services import codex_cli_client, model_catalog, openrouter_client
from prompt_workbench.ui import session


def render(workspace: Workspace) -> None:
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
    model_id = st.selectbox("Model", model_ids, index=index)
    st.caption(model_catalog.get(model_id).note)

    session.set_provider_config(
        openrouter_client.ProviderConfig(
            api_key=api_key, base_url=config.base_url, default_model=model_id
        )
    )

    st.divider()
    st.caption(
        "Local judge: "
        + ("`codex` found" if codex_cli_client.is_available() else "`codex` not installed")
    )

    with st.expander("Advanced: platform instruction"):
        st.caption(
            "How the workbench itself conducts discovery and writes artifacts. "
            "This is not a candidate prompt — those live in the Candidates tab."
        )
        text = st.text_area(
            "Prompt-engineer instruction",
            value=session.platform_instruction(),
            height=240,
            label_visibility="collapsed",
        )
        session.set_platform_instruction(text)
        st.caption(f"Revision {workspace.platform_instruction_revision}")

    stale = workspace.stale_artifacts()
    if stale:
        st.divider()
        st.warning("**Out of date**\n\n" + "\n\n".join(f"- {item}" for item in stale))
