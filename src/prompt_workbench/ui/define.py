"""Define — turn a rough description into a confirmed brief.

The chat and the brief sit side by side on purpose. A discovery chat that fills
in a form the user cannot see is a chat that gets to decide what they meant;
here every answer lands in an editable field, and the user's edit wins.
"""

from __future__ import annotations

import streamlit as st

from prompt_workbench.core import discovery
from prompt_workbench.core.chat_memory import ContextLimitReached, ThreadStore
from prompt_workbench.core.workspace import Workspace
from prompt_workbench.models.brief import BRIEF_FIELDS, FIELD_LABELS, PromptBrief
from prompt_workbench.services import model_catalog
from prompt_workbench.ui import common, session

THREAD_KEY = "discovery_thread"


def _thread(store: ThreadStore) -> str:
    if THREAD_KEY not in st.session_state:
        st.session_state[THREAD_KEY] = store.open("discovery")
    return st.session_state[THREAD_KEY]


def render(workspace: Workspace) -> None:
    st.subheader("Define the use case")
    st.markdown(
        "Describe what you need a system prompt to do. The clarification chat "
        "asks one question at a time and fills in the brief on the right; every "
        "field stays yours to edit. Confirming records an immutable snapshot that "
        "everything else is generated from."
    )

    store = session.threads()
    thread_id = _thread(store)
    chat_column, brief_column = st.columns([3, 2], gap="large")

    with chat_column:
        _render_chat(workspace, store, thread_id)
    with brief_column:
        _render_brief(workspace)


def _render_chat(workspace: Workspace, store: ThreadStore, thread_id: str) -> None:
    st.markdown("#### Clarification chat")

    header, control = st.columns([3, 1])
    with control:
        if st.button("Clear thread", use_container_width=True, key="clear_discovery"):
            st.session_state[THREAD_KEY] = store.clear(thread_id)
            st.rerun()
    with header:
        st.caption(f"Thread `{thread_id}` · history is kept until you clear it")

    for message in store.history(thread_id):
        with st.chat_message(message.role):
            st.markdown(message.content)

    budget = model_catalog.context_window(session.selected_model())
    used = sum(
        len(m.content) // 4 + 4 for m in store.history(thread_id)
    )
    if used:
        common.context_meter(used, budget)

    disabled = not session.has_credentials()
    if disabled:
        st.info(common.NO_KEY_HELP)

    prompt = st.chat_input("Describe what you need, or answer the question", disabled=disabled)
    if not prompt:
        return

    try:
        with st.spinner("Thinking…"):
            result = discovery.clarify(
                store=store,
                thread_id=thread_id,
                user_message=prompt,
                brief=workspace.draft_brief,
                platform_instruction=session.platform_instruction(),
                complete=session.completion(),
                model=session.selected_model(),
                settings=session.model_settings(),
            )
    except ContextLimitReached as error:
        st.warning(str(error))
        return
    except Exception as error:  # noqa: BLE001 - one place, one message
        common.show_error("Clarification", error)
        return

    workspace.update_draft(result.brief)
    if result.ready:
        st.session_state["brief_ready"] = True
    st.rerun()


def _render_brief(workspace: Workspace) -> None:
    st.markdown("#### Brief")
    if st.session_state.get("brief_ready"):
        st.success("The assistant thinks this brief is complete. Review it and confirm.")

    values: dict[str, str] = {}
    for name in BRIEF_FIELDS:
        values[name] = st.text_area(
            FIELD_LABELS[name],
            value=getattr(workspace.draft_brief, name),
            key=f"brief_{name}",
            height=80,
        )
    edited = PromptBrief(**values)
    if edited != workspace.draft_brief:
        workspace.update_draft(edited)

    confirmed = workspace.confirmed_brief
    if confirmed is not None:
        st.caption(f"Confirmed snapshot `{confirmed.ref}`")
    if workspace.brief_is_dirty:
        st.warning(
            "The brief has changed since it was last confirmed. Confirm again to "
            "make the new version available to generation."
        )

    if st.button("Confirm brief", type="primary", use_container_width=True):
        try:
            snapshot = workspace.confirm_brief()
        except ValueError as error:
            st.error(str(error))
        else:
            st.success(f"Confirmed as revision {snapshot.revision}.")
            st.rerun()
