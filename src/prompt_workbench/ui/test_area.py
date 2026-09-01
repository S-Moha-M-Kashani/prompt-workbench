"""Test — run a candidate by hand and record what came back.

Manual on purpose. Automatic runs across every candidate would be quicker and
would also let a user avoid ever reading a response, which is where the useful
information actually is. Sending a message is a decision; so is scoring it.
"""

from __future__ import annotations

import streamlit as st

from prompt_workbench.core import testing
from prompt_workbench.core.chat_memory import ContextLimitReached
from prompt_workbench.core.workspace import Workspace
from prompt_workbench.models.model_settings import ModelSettings
from prompt_workbench.services import model_catalog
from prompt_workbench.ui import common, session

THREAD_KEY = "test_thread"
FREE_TEXT = "__free_text__"


def render(workspace: Workspace) -> None:
    st.subheader("Test a candidate")
    st.markdown(
        "Pick a candidate, a model, and a test case, then send it yourself. Every "
        "response is recorded with the exact candidate revision, model, settings, "
        "brief, and case that produced it — which is what makes it scoreable later."
    )

    if common.needs_brief(workspace.confirmed_brief):
        return
    if not workspace.candidates:
        st.info("Generate at least one candidate in the **Candidates** tab first.")
        return

    controls, conversation = st.columns([2, 3], gap="large")
    with controls:
        candidate, model, settings, case = _render_controls(workspace)
    with conversation:
        _render_conversation(workspace, candidate, model, settings, case)


def _render_controls(workspace: Workspace):  # type: ignore[no-untyped-def]
    labels = {c.id: f"{c.label} · r{c.revision}" for c in workspace.candidates}
    candidate_id = st.selectbox(
        "Candidate", list(labels), format_func=lambda cid: labels[cid], key="test_candidate"
    )
    candidate = workspace.candidate(candidate_id)

    model_ids = [m.id for m in model_catalog.all_models()]
    model = st.selectbox("Model", model_ids, key="test_model")
    info = model_catalog.get(model)
    st.caption(info.note)

    settings = _render_settings(model)

    cases = workspace.dataset.cases if workspace.dataset else ()
    # A sentinel rather than None, so the "free text" option is a real value the
    # select box can format and compare.
    options = [FREE_TEXT, *[c.id for c in cases]]

    def case_label(case_id: str) -> str:
        if case_id == FREE_TEXT:
            return "Free text (not scoreable)"
        return workspace.dataset.case(case_id).test_message[:60]  # type: ignore[union-attr]

    case_id = st.selectbox("Test case", options, format_func=case_label, key="test_case")
    case = (
        workspace.dataset.case(case_id)
        if case_id != FREE_TEXT and workspace.dataset is not None
        else None
    )
    if case is None:
        st.caption(
            "A free-text response is recorded but cannot be evaluated: there is "
            "nothing to check it against."
        )
    return candidate, model, settings, case


def _render_settings(model: str) -> ModelSettings:
    """Only offer knobs this model honours, so a request never asks for nothing."""
    with st.expander("Model settings"):
        values: dict[str, float | int | None] = {}
        if model_catalog.supports(model, "temperature"):
            values["temperature"] = st.slider("temperature", 0.0, 2.0, 1.0, 0.05, key="t_temp")
            values["top_p"] = st.slider("top_p", 0.0, 1.0, 1.0, 0.05, key="t_topp")
        else:
            st.caption(
                "This is a reasoning model: it ignores sampling settings, so they "
                "are hidden rather than sent and dropped."
            )
        cap = st.number_input("max_tokens (0 = model default)", 0, 32_000, 0, key="t_max")
        values["max_tokens"] = int(cap) or None
        return ModelSettings(**values)  # type: ignore[arg-type]


def _render_conversation(workspace, candidate, model, settings, case) -> None:  # type: ignore[no-untyped-def]
    store = session.threads()
    wanted = testing.thread_config(candidate=candidate, model=model, settings=settings)

    thread_id = st.session_state.get(THREAD_KEY)
    if thread_id is None or store.thread(thread_id).config != wanted:
        if thread_id is not None and not store.thread(thread_id).is_empty:
            st.warning(
                "The candidate, model, or settings changed, so a new thread was "
                "started. The previous conversation's responses are still recorded."
            )
        thread_id = testing.open_test_thread(
            store, candidate=candidate, model=model, settings=settings
        )
        st.session_state[THREAD_KEY] = thread_id

    header, control = st.columns([3, 1])
    with header:
        st.caption(f"Thread `{thread_id}` · {candidate.label} on {model}")
    with control:
        if st.button("New thread", use_container_width=True, key="new_test_thread"):
            st.session_state[THREAD_KEY] = store.clear(thread_id)
            st.rerun()

    for message in store.history(thread_id):
        with st.chat_message(message.role):
            st.markdown(message.content)

    budget = model_catalog.context_window(model)
    used = sum(len(m.content) // 4 + 4 for m in store.history(thread_id))
    if used:
        common.context_meter(used, budget)

    disabled = not session.has_credentials()
    if disabled:
        st.info(common.NO_KEY_HELP)

    default = case.test_message if case is not None else ""
    if case is not None and st.button(
        "Send the selected test case", type="primary", disabled=disabled
    ):
        _send(workspace, store, thread_id, candidate, case, default, model, settings)

    typed = st.chat_input("Or type a message yourself", disabled=disabled)
    if typed:
        _send(workspace, store, thread_id, candidate, case, typed, model, settings)


def _send(workspace, store, thread_id, candidate, case, message, model, settings) -> None:  # type: ignore[no-untyped-def]
    brief = workspace.confirmed_brief
    assert brief is not None
    try:
        with st.spinner("Running…"):
            record = testing.send_test_message(
                store=store,
                thread_id=thread_id,
                candidate=candidate,
                brief=brief,
                case=case,
                user_message=message,
                model=model,
                settings=settings,
                complete=session.completion(),
                new_id=workspace.new_id,
                clock=workspace.clock,
            )
    except (ContextLimitReached, testing.ConfigurationChanged) as error:
        st.warning(str(error))
        return
    except Exception as error:  # noqa: BLE001
        common.show_error("Running the candidate", error)
        return
    workspace.record_execution(record)
    st.rerun()
