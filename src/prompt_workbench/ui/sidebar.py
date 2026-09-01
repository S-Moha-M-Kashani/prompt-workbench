"""Provider access and the judge. Nothing about the prompt lives here.

Kept deliberately thin: the previous design spread sampling knobs across five
areas, which turned tuning into the activity and left the prompt as an
afterthought. The prompt is the subject of this application, so the sidebar
holds only what it takes to reach a model at all.
"""

from __future__ import annotations

import streamlit as st

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
