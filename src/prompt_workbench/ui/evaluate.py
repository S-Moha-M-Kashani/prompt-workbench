"""Evaluate — configure the metrics, then explicitly ask for a grade.

Nothing on any other tab reaches this code. Generating a candidate, editing a
case, or running a test cannot start a scoring call, and the import graph is
where that is enforced rather than in a comment. Pressing the button is the only
way across.

Everything the grade is made of is on screen next to it: each metric's rubric,
weight, score, and reason, plus every failure. A grade you cannot reproduce by
hand from what is displayed is a number to be suspicious of.
"""

from __future__ import annotations

import streamlit as st

from prompt_workbench.core import evaluation, metric_config
from prompt_workbench.core.evaluation import EvaluationBlocked
from prompt_workbench.core.workspace import Workspace
from prompt_workbench.models.evaluation import EvaluationRun
from prompt_workbench.models.execution import ExecutionRecord
from prompt_workbench.services import judges
from prompt_workbench.ui import common, session

LIMITATIONS = (
    "A grade here is evidence, not a fact. The judge is a language model with "
    "its own noise and biases; the ground truth may itself have been generated "
    "rather than written; and a metric only measures what its rubric says. Read "
    "the reasons, not just the number. See docs/LIMITATIONS.md for what a grade "
    "does and does not mean."
)


def render(workspace: Workspace) -> None:
    st.subheader("Evaluate")
    st.markdown(
        "Choose which recorded responses to score and which metrics to score them "
        "against, then press Evaluate. **Nothing is scored until you do** — no "
        "edit, generation, or test run ever triggers a judge call."
    )

    _render_metrics(workspace)
    st.divider()
    _render_run_controls(workspace)

    latest = workspace.latest_evaluation
    if latest is not None:
        st.divider()
        _render_results(latest)


# --- metric configuration -------------------------------------------------


def _render_metrics(workspace: Workspace) -> None:
    st.markdown("#### Metrics")
    for metric in workspace.metrics:
        with st.expander(
            f"{'☑' if metric.enabled else '☐'} {metric.name} · {metric.kind.label}"
            f" · weight {metric.weight:g}"
        ):
            st.caption(metric.rubric)
            enabled_column, weight_column, remove_column = st.columns([1, 2, 1])
            enabled = enabled_column.checkbox(
                "Enabled", value=metric.enabled, key=f"en_{metric.id}"
            )
            if enabled != metric.enabled:
                workspace.set_metric_enabled(metric.id, enabled)
                st.rerun()
            weight = weight_column.number_input(
                "Weight", min_value=0.0, max_value=10.0, value=float(metric.weight),
                step=0.5, key=f"w_{metric.id}",
            )
            if weight != metric.weight:
                workspace.set_metric_weight(metric.id, float(weight))
                st.rerun()
            if not metric.builtin and remove_column.button("Remove", key=f"rm_{metric.id}"):
                workspace.remove_metric(metric.id)
                st.rerun()

    with st.expander("Add a custom metric"):
        name = st.text_input("Name", key="new_metric_name")
        rubric = st.text_area(
            "Rubric",
            key="new_metric_rubric",
            help="Say what a 1.0 and a 0.0 look like. A judge given three words invents the rest.",
        )
        weight = st.number_input("Weight", 0.0, 10.0, 1.0, 0.5, key="new_metric_weight")
        if st.button("Add metric"):
            try:
                workspace.add_metric(name=name, rubric=rubric, weight=float(weight))
            except ValueError as error:
                st.error(str(error))
            else:
                st.rerun()

    for problem in metric_config.validate(workspace.metrics):
        st.warning(problem)


# --- running --------------------------------------------------------------


def _render_run_controls(workspace: Workspace) -> None:
    st.markdown("#### Responses to score")

    attributable = [record for record in workspace.executions if record.is_attributable]
    if not attributable:
        st.info(
            "No scoreable responses yet. Send a **ground-truth test case** in the "
            "Test tab — a free-text response has nothing to be checked against."
        )
        return

    labels = {
        record.id: f"{record.user_message[:50]} → {record.response[:50]}"
        for record in attributable
    }
    chosen_ids = st.multiselect(
        "Recorded responses",
        list(labels),
        default=list(labels),
        format_func=lambda rid: labels[rid],
        key="eval_selection",
    )
    chosen = [record for record in attributable if record.id in chosen_ids]

    backend_column, model_column = st.columns(2)
    backends = judges.available_backends()
    backend = backend_column.selectbox(
        "Judge backend",
        backends,
        format_func=lambda b: judges.BACKEND_LABELS[b],
        key="judge_backend_choice",
    )
    session.set_judge_backend(backend)
    models = judges.models_for(backend)
    model = model_column.selectbox("Judge model", models, key=f"judge_model_choice_{backend}")
    session.set_judge_model(model)
    st.caption(
        "The local CLI judge uses this machine's own login and never sees your "
        "workspace API key. Judging with a different model from the one under "
        "test avoids a model grading its own output."
    )

    if st.button("Evaluate", type="primary", disabled=not chosen):
        _run(workspace, chosen, backend=backend, model=model)

    st.caption(LIMITATIONS)


def _run(
    workspace: Workspace,
    executions: list[ExecutionRecord],
    *,
    backend: str,
    model: str,
) -> None:
    candidate = workspace.candidate(executions[0].candidate.id)
    brief = workspace.confirmed_brief
    dataset = workspace.dataset
    if brief is None or dataset is None:
        st.error("A confirmed brief and a dataset are needed to evaluate.")
        return
    try:
        with st.spinner(f"Scoring {len(executions)} response(s)…"):
            run = evaluation.run(
                executions=executions,
                dataset=dataset,
                brief=brief,
                candidate=candidate,
                metrics=workspace.metrics,
                judge=session.judge(),
                judge_backend=backend,
                judge_model=model,
                new_id=workspace.new_id,
                clock=workspace.clock,
            )
    except EvaluationBlocked as error:
        st.error(f"Evaluation was not started:\n\n{error}")
        return
    except Exception as error:  # noqa: BLE001
        common.show_error("Evaluation", error)
        return
    workspace.record_evaluation(run)
    st.rerun()


# --- results --------------------------------------------------------------


def _render_results(run: EvaluationRun) -> None:
    st.markdown("#### Latest result")
    grade_column, detail_column = st.columns([1, 3])
    grade_column.metric("Overall grade", str(run.overall))
    detail_column.caption(
        f"Run `{run.id}` · judged by {run.judge_backend}/{run.judge_model} · "
        f"candidate {run.candidate} · brief {run.source_brief} · dataset {run.source_dataset}"
    )

    if run.has_failures:
        st.warning(
            f"{len(run.failures)} metric score(s) failed and were left out of the "
            "grade rather than replaced with a neutral value. They are listed below."
        )

    for case in run.cases:
        header = (
            f"Case {case.case_id} · "
            + (f"{case.grade:.0%}" if case.grade is not None else "no grade")
        )
        with st.expander(header):
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
            if case.grade is not None:
                st.caption(
                    "Weighted mean of the scored metrics above; failed and "
                    "not-applicable metrics are excluded, not counted as zero."
                )
