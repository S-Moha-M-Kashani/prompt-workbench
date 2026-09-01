"""The whole screen: a use case, the prompt under test, and one chat.

Three things, in the order you meet them. Pick a situation from the dropdown and
its explanation is written into the chat. Open the panel to read or edit the
prompt that situation starts from. Then talk — to the prompt engineer, who
discusses it and hands back a complete replacement, or as the end user, where
one message gets exactly one response under the current prompt.

The mode switch is the only control on the page, because it is the only one that
changes what a message *means*. Everything else that used to be a widget —
sampling knobs, technique pickers, dataset editors — has been removed: they made
tuning the activity, when the prompt is the subject.
"""

from __future__ import annotations

import streamlit as st

from prompt_workbench.core import engineer, evaluation, one_shot
from prompt_workbench.core.chat_memory import ContextLimitReached, ThreadStore
from prompt_workbench.core.evaluation import EvaluationBlocked
from prompt_workbench.core.one_shot import RunFailed
from prompt_workbench.core.session import ENGINEER_MODE, MODE_LABELS, MODES, USER_MODE, Session
from prompt_workbench.models.evaluation import EvaluationRun
from prompt_workbench.services import use_case_catalog
from prompt_workbench.ui import session

NO_KEY_HELP = "Add a provider API key in the sidebar to enable this."
THREAD_KEY = "engineer_thread"

LIMITATIONS = (
    "A grade is evidence, not a fact: the judge is a language model with its own "
    "noise. Read the reasons, not just the number."
)


def render() -> None:
    workbench = session.workbench()
    _render_picker(workbench)

    if workbench.use_case is None:
        st.info("Pick a use case above to begin.")
        return

    _render_prompt_panel(workbench)
    st.divider()
    _render_chat(workbench)


# --- the dropdown ---------------------------------------------------------


def _render_picker(workbench: Session) -> None:
    cases = use_case_catalog.all_use_cases()
    keys = [case.key for case in cases]
    labels = {
        case.key: f"{use_case_catalog.FAMILY_LABELS.get(case.family, case.family)} · {case.label}"
        for case in cases
    }

    current = workbench.use_case.key if workbench.use_case else None
    chosen = st.selectbox(
        "Use case",
        keys,
        index=keys.index(current) if current in keys else None,
        format_func=lambda key: labels[key],
        placeholder="Pick a prompt-engineering situation…",
        key="use_case_pick",
    )
    if chosen and chosen != current:
        _select(workbench, chosen)


def _select(workbench: Session, key: str) -> None:
    """Load a use case and write its situation into a fresh chat."""
    use_case = use_case_catalog.get(key)
    workbench.select(use_case)
    store = session.threads()
    thread_id = store.open("engineer")
    st.session_state[THREAD_KEY] = thread_id
    store.append(thread_id, "assistant", use_case.as_situation_message())
    st.rerun()


# --- the prompt under test ------------------------------------------------


def _render_prompt_panel(workbench: Session) -> None:
    use_case = workbench.use_case
    assert use_case is not None
    modified = " · edited" if workbench.prompt_is_modified else ""
    with st.expander(
        f"System prompt under test — revision {workbench.prompt_revision}{modified}"
    ):
        st.caption(
            "Placeholders are filled from the mocked context at send time, so what "
            "you edit here keeps its `{name}` markers."
        )
        text = st.text_area(
            "System prompt",
            value=workbench.prompt_under_test,
            height=320,
            key=f"prompt_{use_case.key}_{workbench.prompt_revision}",
            label_visibility="collapsed",
        )
        save, reset = st.columns([1, 1])
        if save.button("Save prompt", use_container_width=True):
            try:
                if workbench.update_prompt(text):
                    st.rerun()
            except ValueError as error:
                st.error(str(error))
        if reset.button(
            "Restore original", use_container_width=True, disabled=not workbench.prompt_is_modified
        ):
            workbench.reset_prompt()
            st.rerun()

        missing = use_case.unfilled_placeholders(workbench.prompt_under_test)
        if missing:
            st.warning(
                "No mock supplies "
                + ", ".join(f"`{{{name}}}`" for name in missing)
                + " — it would reach the model as literal text."
            )
        with st.popover("Mocked context"):
            for block in use_case.mocks:
                st.markdown(f"**`{block.placeholder}` — {block.label}**")
                if block.note:
                    st.caption(block.note)
                st.code(block.content or "(empty)", language=None)


# --- the one chat ---------------------------------------------------------


def _thread(store: ThreadStore) -> str:
    if THREAD_KEY not in st.session_state:
        st.session_state[THREAD_KEY] = store.open("engineer")
    return st.session_state[THREAD_KEY]


def _render_chat(workbench: Session) -> None:
    store = session.threads()
    thread_id = _thread(store)

    mode = st.radio(
        "Mode",
        MODES,
        format_func=lambda m: MODE_LABELS[m],
        horizontal=True,
        key="chat_mode",
        help=(
            "Prompt engineer discusses the prompt and hands back a complete "
            "replacement. End user sends one message under the prompt and keeps "
            "no history, so the response reflects the prompt alone."
        ),
    )
    workbench.mode = mode

    for message in store.history(thread_id):
        with st.chat_message(message.role):
            st.markdown(message.content)

    _render_proposal(workbench)
    _render_evaluation(workbench)

    disabled = not session.has_credentials()
    if disabled:
        st.info(NO_KEY_HELP)

    if mode == USER_MODE:
        _render_user_mode_extras(workbench, store, thread_id, disabled)

    placeholder = (
        "Ask about the prompt, or ask for a complete rewrite"
        if mode == ENGINEER_MODE
        else "Send one message as the end user"
    )
    typed = st.chat_input(placeholder, disabled=disabled)
    if typed:
        if mode == ENGINEER_MODE:
            _send_to_engineer(workbench, store, thread_id, typed)
        else:
            _run_one_shot(workbench, store, thread_id, typed)


def _render_user_mode_extras(
    workbench: Session, store: ThreadStore, thread_id: str, disabled: bool
) -> None:
    use_case = workbench.use_case
    assert use_case is not None
    left, right = st.columns([3, 1])
    left.caption(
        "One message in, one response out. No history: the response reflects the "
        "prompt and the mocks, nothing else."
    )
    if right.button("Send the example", disabled=disabled, use_container_width=True):
        _run_one_shot(workbench, store, thread_id, use_case.example_message)


def _send_to_engineer(
    workbench: Session, store: ThreadStore, thread_id: str, message: str
) -> None:
    use_case = workbench.use_case
    assert use_case is not None
    try:
        with st.spinner("Thinking…"):
            result = engineer.discuss(
                store=store,
                thread_id=thread_id,
                user_message=message,
                use_case=use_case,
                prompt_under_test=workbench.prompt_under_test,
                complete=session.completion(),
                model=session.selected_model(),
            )
    except ContextLimitReached as error:
        st.warning(str(error))
        return
    except Exception as error:  # noqa: BLE001
        st.error(f"The prompt engineer failed: {error}")
        return
    st.session_state["proposed_prompt"] = result.proposed_prompt
    st.rerun()


def _render_proposal(workbench: Session) -> None:
    """A complete prompt the engineer offered, and one action to take it."""
    proposed = st.session_state.get("proposed_prompt", "")
    if not proposed:
        return
    with st.container(border=True):
        st.markdown("**The prompt engineer proposed a complete replacement prompt.**")
        st.code(proposed, language=None)
        apply, discard = st.columns([1, 1])
        if apply.button("Use this prompt", type="primary", use_container_width=True):
            workbench.update_prompt(proposed)
            st.session_state["proposed_prompt"] = ""
            st.rerun()
        if discard.button("Keep the current one", use_container_width=True):
            st.session_state["proposed_prompt"] = ""
            st.rerun()


def _run_one_shot(
    workbench: Session, store: ThreadStore, thread_id: str, message: str
) -> None:
    use_case = workbench.use_case
    assert use_case is not None
    store.append(thread_id, "user", message)
    try:
        with st.spinner("Running the prompt…"):
            record = one_shot.run_once(
                use_case=use_case,
                prompt_under_test=workbench.prompt_under_test,
                prompt_revision=workbench.prompt_revision,
                user_message=message,
                model=session.selected_model(),
                settings=session.model_settings(),
                complete=session.completion_with_usage(),
                new_id=workbench.new_id,
                clock=workbench.clock,
            )
    except RunFailed as error:
        st.warning(str(error))
        return
    except Exception as error:  # noqa: BLE001
        st.error(f"The run failed: {error}")
        return
    workbench.record_run(record)
    store.append(thread_id, "assistant", record.response)
    st.rerun()


# --- evaluation, only when asked ------------------------------------------


def _render_evaluation(workbench: Session) -> None:
    latest = workbench.latest_run
    if latest is None:
        return

    cost = (
        f" · {latest.usage}" if latest.usage.is_reported else " · token usage not reported"
    )
    left, right = st.columns([3, 1])
    left.caption(
        f"Last response recorded as `{latest.id}` at revision {latest.prompt_revision}"
        f"{cost}. Nothing is scored until you ask."
    )
    if right.button("Evaluate", type="primary", use_container_width=True):
        _evaluate(workbench)

    run = workbench.latest_evaluation
    if run is not None:
        _render_result(run)


def _evaluate(workbench: Session) -> None:
    use_case = workbench.use_case
    assert use_case is not None
    try:
        with st.spinner("Scoring the response…"):
            result = evaluation.run(
                runs=workbench.scoreable_runs()[-1:],
                use_case=use_case,
                metrics=workbench.metrics,
                judge=session.judge(),
                judge_backend=session.judge_backend(),
                judge_model=session.judge_model(),
                new_id=workbench.new_id,
                clock=workbench.clock,
            )
    except EvaluationBlocked as error:
        st.error(f"Evaluation was not started:\n\n{error}")
        return
    except Exception as error:  # noqa: BLE001
        st.error(f"Evaluation failed: {error}")
        return
    workbench.record_evaluation(result)
    st.rerun()


def _render_result(run: EvaluationRun) -> None:
    with st.container(border=True):
        grade, detail = st.columns([1, 3])
        grade.metric("Grade", str(run.overall))
        detail.caption(
            f"Judged by {run.judge_backend}/{run.judge_model} · prompt revision "
            f"{run.prompt_revision} · against this use case's own criteria"
        )
        for case in run.cases:
            for score in case.scores:
                if score.failed:
                    st.error(f"**{score.metric_name}** failed: {score.failure}")
                elif not score.applicable:
                    st.info(f"**{score.metric_name}** not applicable: {score.reason}")
                else:
                    st.markdown(
                        f"**{score.metric_name}** — {score.score:.2f} "
                        f"(weight {score.weight:g})  \n{score.reason}"
                    )
        st.caption(LIMITATIONS)
