"""Candidates — one system prompt per technique, side by side.

The techniques are checkboxes rather than a single button so a comparison can be
narrowed to the two that matter, and every candidate stays editable: the
generated text is a starting point, not a verdict.
"""

from __future__ import annotations

import streamlit as st

from prompt_workbench.core import candidate_generation
from prompt_workbench.core.workspace import Workspace
from prompt_workbench.models.candidates import PromptTechnique
from prompt_workbench.ui import common, session


def render(workspace: Workspace) -> None:
    st.subheader("Candidate system prompts")
    st.markdown(
        "One complete system prompt per technique, generated from the confirmed "
        "brief. Only the few-shot candidate is shown the test cases — giving them "
        "to the others would quietly make every candidate few-shot, and the "
        "comparison would stop meaning anything."
    )

    if common.needs_brief(workspace.confirmed_brief):
        return

    _render_generation(workspace)
    st.divider()

    if not workspace.candidates:
        st.caption("No candidates yet.")
        return

    for candidate in workspace.candidates:
        with st.expander(candidate.label, expanded=False):
            common.provenance_caption(
                ("brief", candidate.source_brief),
                ("dataset", candidate.source_dataset),
            )
            st.caption(candidate.technique.description)
            text = st.text_area(
                "System prompt",
                value=candidate.system_prompt,
                height=280,
                key=f"cand_{candidate.id}_{candidate.revision}",
            )
            if st.button("Save edit", key=f"save_cand_{candidate.id}"):
                if text.strip() and text != candidate.system_prompt:
                    workspace.edit_candidate(candidate.id, text)
                    st.rerun()


def _render_generation(workspace: Workspace) -> None:
    st.markdown("**Techniques to generate**")
    columns = st.columns(len(PromptTechnique))
    chosen: list[PromptTechnique] = []
    for column, technique in zip(columns, PromptTechnique, strict=True):
        with column:
            if st.checkbox(
                technique.label,
                value=True,
                key=f"tech_{technique.value}",
                help=technique.description,
            ):
                chosen.append(technique)

    disabled = not session.has_credentials() or not chosen
    if st.button(
        "Generate candidates",
        type="primary",
        disabled=disabled,
        help=common.NO_KEY_HELP if not session.has_credentials() else None,
    ):
        _generate(workspace, tuple(chosen), overwrite=st.session_state.get("confirm_overwrite", False))

    if workspace.has_edited_candidates:
        st.checkbox(
            "Overwrite my hand-edited candidates",
            key="confirm_overwrite",
            help="Regenerating replaces every candidate, including ones you edited.",
        )


def _generate(
    workspace: Workspace, techniques: tuple[PromptTechnique, ...], *, overwrite: bool
) -> None:
    brief = workspace.confirmed_brief
    assert brief is not None
    try:
        with st.spinner("Writing candidates…"):
            candidates = candidate_generation.generate(
                brief=brief,
                dataset=workspace.dataset,
                techniques=techniques,
                platform_instruction=session.platform_instruction(),
                complete=session.completion(),
                model=session.selected_model(),
                settings=session.model_settings(),
                new_id=workspace.new_id,
                clock=workspace.clock,
            )
        workspace.set_candidates(candidates, overwrite=overwrite)
    except Exception as error:  # noqa: BLE001
        common.show_error("Generating candidates", error)
        return
    st.rerun()
