"""Ground Truth — the evidence a candidate is judged against.

Everything here is editable, and that is the feature. A generated dataset is a
first draft of what "good" means; leaving it unread is how an evaluation ends up
measuring the generator's assumptions instead of the user's.
"""

from __future__ import annotations

import streamlit as st

from prompt_workbench.core import dataset_generation
from prompt_workbench.core.workspace import Workspace
from prompt_workbench.models.ground_truth import CaseCategory, GroundTruthCase
from prompt_workbench.ui import common, session


def _lines(text: str) -> tuple[str, ...]:
    return tuple(line.strip() for line in text.splitlines() if line.strip())


def render(workspace: Workspace) -> None:
    st.subheader("Ground truth")
    st.markdown(
        "Test cases describe what any acceptable response must do and must not "
        "do. Required criteria and forbidden behaviours are what a judge checks; "
        "a reference answer is optional, and only worth writing when the case "
        "really has one right shape."
    )

    if common.needs_brief(workspace.confirmed_brief):
        return

    _render_generation(workspace)
    st.divider()

    dataset = workspace.dataset
    if dataset is None or not len(dataset):
        st.caption("No test cases yet.")
    else:
        common.provenance_caption(
            ("brief", dataset.source_brief), ("dataset", dataset.ref)
        )
        for index, case in enumerate(dataset.cases, start=1):
            _render_case(workspace, case, index)

    _render_new_case(workspace)


def _render_generation(workspace: Workspace) -> None:
    count_column, button_column = st.columns([1, 2])
    with count_column:
        count = st.number_input(
            "Cases to generate",
            min_value=1,
            max_value=30,
            value=dataset_generation.DEFAULT_CASE_COUNT,
            key="dataset_count",
        )
    with button_column:
        st.write("")
        disabled = not session.has_credentials()
        if st.button(
            "Generate test cases",
            type="primary",
            disabled=disabled,
            help=common.NO_KEY_HELP if disabled else None,
        ):
            _generate(workspace, int(count))


def _generate(workspace: Workspace, count: int) -> None:
    brief = workspace.confirmed_brief
    assert brief is not None  # guarded by needs_brief
    try:
        with st.spinner("Writing test cases…"):
            dataset = dataset_generation.generate(
                brief=brief,
                count=count,
                platform_instruction=session.platform_instruction(),
                complete=session.completion(),
                model=session.selected_model(),
                settings=session.model_settings(),
                new_id=workspace.new_id,
                clock=workspace.clock,
            )
    except Exception as error:  # noqa: BLE001
        common.show_error("Generating test cases", error)
        return
    workspace.set_dataset(dataset)
    st.rerun()


def _render_case(workspace: Workspace, case: GroundTruthCase, index: int) -> None:
    title = f"{index}. [{case.category.label}] {case.test_message[:70]}"
    with st.expander(title):
        message = st.text_area("Test message", value=case.test_message, key=f"msg_{case.id}")
        criteria = st.text_area(
            "Required criteria (one per line)",
            value="\n".join(case.required_criteria),
            key=f"crit_{case.id}",
        )
        forbidden = st.text_area(
            "Forbidden behaviours (one per line)",
            value="\n".join(case.forbidden_behaviours),
            key=f"forb_{case.id}",
        )
        tags = st.text_input("Tags (comma separated)", value=", ".join(case.tags), key=f"tag_{case.id}")
        reference = st.text_area(
            "Reference answer (optional)",
            value=case.reference_answer or "",
            key=f"ref_{case.id}",
            help="Leave blank when several different answers would be equally correct.",
        )
        category = st.selectbox(
            "Category",
            [c.value for c in CaseCategory],
            index=[c.value for c in CaseCategory].index(case.category.value),
            key=f"cat_{case.id}",
        )

        save, duplicate, remove = st.columns(3)
        if save.button("Save", key=f"save_{case.id}", use_container_width=True):
            try:
                workspace.replace_case(
                    case.id,
                    GroundTruthCase(
                        id=case.id,
                        test_message=message,
                        required_criteria=_lines(criteria),
                        forbidden_behaviours=_lines(forbidden),
                        tags=tuple(t.strip() for t in tags.split(",") if t.strip()),
                        reference_answer=reference.strip() or None,
                        category=CaseCategory(category),
                    ),
                )
            except ValueError as error:
                st.error(str(error))
            else:
                st.rerun()
        if duplicate.button("Duplicate", key=f"dup_{case.id}", use_container_width=True):
            workspace.duplicate_case(case.id)
            st.rerun()
        if remove.button("Remove", key=f"del_{case.id}", use_container_width=True):
            workspace.remove_case(case.id)
            st.rerun()


def _render_new_case(workspace: Workspace) -> None:
    with st.expander("Add a case by hand"):
        message = st.text_area("Test message", key="new_case_message")
        criteria = st.text_area("Required criteria (one per line)", key="new_case_criteria")
        forbidden = st.text_area("Forbidden behaviours (one per line)", key="new_case_forbidden")
        if st.button("Add case"):
            try:
                workspace.add_case(
                    GroundTruthCase(
                        id="pending",
                        test_message=message,
                        required_criteria=_lines(criteria),
                        forbidden_behaviours=_lines(forbidden),
                    )
                )
            except ValueError as error:
                st.error(str(error))
            else:
                st.rerun()
