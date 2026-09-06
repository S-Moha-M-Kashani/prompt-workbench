"""The screen: a sequence from a described case to a configuration you can keep.

Sections rather than tabs, in the order the work happens, because each step is
only meaningful once the one before it exists — and a tab strip invites picking
them in an order that cannot work. Later sections stay visible but say what they
are waiting for.

The deliverable is the bottom of the page: the cheapest configuration that
cleared every threshold, and the deepeval code that measured it. That code is
shown, never written to disk — you paste it into the project that ships, where
the same library will produce the same numbers.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

import streamlit as st

from prompt_workbench.core import case_intake, one_shot, scoring, sweep, variants
from prompt_workbench.core.session import Session
from prompt_workbench.core.sweep import SweepCell
from prompt_workbench.models.case import CaseBrief, EvalCase
from prompt_workbench.models.task_type import FineTuneVerdict, TaskType, VariantApproach
from prompt_workbench.models.usage import TokenUsage
from prompt_workbench.services import deepeval_metrics, task_catalog, use_case_catalog
from prompt_workbench.ui import lab, session

NO_KEY = "Add a provider API key in the sidebar to enable this."
NO_CASE = "Describe the case at the top — a sweep is scored against it."

# The widgets whose value the session also owns. Streamlit keeps a widget's value
# under its key once it has been drawn, and that copy wins over any ``value=``
# passed later — so anything that fills a box in on the user's behalf has to
# write the key, not the session attribute alone.
DESCRIPTION_KEY = "case_description"
TASK_TYPE_KEY = "task_type_pick"


def render() -> None:
    workbench = session.workbench()
    _describe(workbench)
    st.divider()
    _cases(workbench)
    st.divider()
    _variants(workbench)
    st.divider()
    lab.render(workbench)
    st.divider()
    _metrics(workbench)
    st.divider()
    _sweep(workbench)


# --- 1. the case ----------------------------------------------------------


def _adopt_task_type(workbench: Session, task: TaskType) -> None:
    """Settle the kind of job on the user's behalf, picker included.

    Only for the paths that choose *for* the user — loading an example, or
    accepting a proposal read out of the description. A page that has quietly
    adopted a task type while the visible control still says "Pick the kind of
    job…" is telling the user two different things.

    Streamlit refuses a write to a widget's key once that widget has been drawn,
    so this must run before the picker — which is also the only time it is
    needed, since a choice made *in* the picker is already in the picker.
    """
    workbench.set_task_type(task)
    st.session_state[TASK_TYPE_KEY] = task.key


def _load_example(workbench: Session) -> None:
    """Start from a worked situation instead of a blank box.

    Kept in an expander, and deliberately not the first thing on the page: the
    examples are here to show what a well-shaped case looks like, not to be the
    thing the workbench is for.
    """
    with st.expander("Start from an example instead"):
        examples = use_case_catalog.all_use_cases()
        keys = [example.key for example in examples]
        labels = {example.key: example.label for example in examples}
        chosen = st.selectbox(
            "Worked situations",
            keys,
            index=None,
            format_func=lambda key: labels[key],
            placeholder="Pick one to fill in the case below…",
            key="example_pick",
        )
        if chosen and st.button("Load it", key="load_example"):
            example = use_case_catalog.get(chosen)
            description = (
                f"{example.situation}\n\nWhat usually goes wrong: {example.trap}"
            )
            workbench.description = description
            st.session_state[DESCRIPTION_KEY] = description
            if example.task_type:
                _adopt_task_type(workbench, task_catalog.get(example.task_type))
            if example.example_message.strip():
                workbench.set_cases(
                    (EvalCase(id=workbench.new_id("case"), input=example.example_message.strip()),)
                )
            st.rerun()


def _describe(workbench: Session) -> None:
    st.subheader("1 · Your case")
    _load_example(workbench)

    # The box owns the text once it has been drawn, so the session is seeded
    # into it rather than passed as a default that the box would then ignore.
    st.session_state.setdefault(DESCRIPTION_KEY, workbench.description)
    description = st.text_area(
        "What must this prompt do?",
        height=110,
        placeholder="e.g. Sort incoming support tickets into one of six queues, "
        "from the subject line and first message.",
        key=DESCRIPTION_KEY,
    )
    if description != workbench.description:
        proposed = workbench.describe(description)
        if proposed is not None and workbench.task_type is None:
            _adopt_task_type(workbench, proposed)
            st.rerun()

    tasks = task_catalog.all_task_types()
    keys = [task.key for task in tasks]
    labels = {task.key: task.label for task in tasks}
    current = workbench.task_type.key if workbench.task_type else None
    st.session_state.setdefault(TASK_TYPE_KEY, current)

    left, right = st.columns([2, 3])
    with left:
        chosen = st.selectbox(
            "Kind of job",
            keys,
            format_func=lambda key: labels[key],
            placeholder="Pick the kind of job…",
            key=TASK_TYPE_KEY,
        )
        if chosen and chosen != current:
            # The picker already holds the choice; only the session needs telling.
            workbench.set_task_type(task_catalog.get(chosen))
            st.rerun()
    with right:
        workbench.output_format = st.text_input(
            "Required output format (optional)",
            placeholder="e.g. the queue name alone, or a JSON object",
            key="output_format",
        )

    task = workbench.task_type
    if task is None:
        st.info("Describe the case, or pick the kind of job, to begin.")
        return

    st.caption(task.description)
    if task.fine_tune is FineTuneVerdict.LIKELY:
        st.warning(f"**{task.fine_tune.label}.** {task.fine_tune_note}")
    else:
        st.caption(f"**{task.fine_tune.label}.** {task.fine_tune_note}")

    _model_ladder(workbench)


def _model_ladder(workbench: Session) -> None:
    """What this job could run on, cheapest first."""
    task = workbench.task_type
    assert task is not None
    registry = session.registry()
    ladder = [entry for entry in registry.recommended() if entry.has_price]
    if task.needs_strong_model:
        st.caption(
            "This job needs a capable model — a cheap one here is a false economy, "
            "since its noise is indistinguishable from a worse prompt."
        )
    with st.expander("Models for this job, cheapest first"):
        for entry in ladder:
            st.markdown(
                f"`{entry.id}` — ${entry.price_in_per_million:.2f} in / "
                f"${entry.price_out_per_million:.2f} out per 1M"
            )


# --- 2. cases -------------------------------------------------------------


def _cases(workbench: Session) -> None:
    st.subheader("2 · Test cases")
    if workbench.task_type is None:
        st.info("Pick the kind of job above, then cases can be written or generated.")
        return

    st.caption(
        "What the prompt gets judged on. Generated from your description, and "
        "yours to edit — a case you would not have written is a measurement you "
        "should not trust."
    )
    count_col, button_col = st.columns([1, 3])
    with count_col:
        count = st.number_input(
            "How many", 1, 30, case_intake.DEFAULT_CASE_COUNT, key="case_count"
        )
    with button_col:
        st.write("")
        disabled = not session.has_credentials()
        if st.button("Generate test cases", type="primary", disabled=disabled,
                     help=NO_KEY if disabled else None):
            _generate_cases(workbench, int(count))

    _add_case_by_hand(workbench)

    if not workbench.cases:
        st.caption("No test cases yet.")
        return

    for index, case in enumerate(workbench.cases, start=1):
        with st.expander(f"{index}. {case.input[:70]}"):
            text = st.text_area("Input", value=case.input, key=f"in_{case.id}")
            expected = st.text_input(
                "Expected output (leave blank when several answers are equally correct)",
                value=case.expected_output or "", key=f"exp_{case.id}",
            )
            context = st.text_area(
                "Context, one per line (optional)",
                value="\n".join(case.context), key=f"ctx_{case.id}",
            )
            if case.notes:
                st.caption(case.notes)
            save, remove = st.columns(2)
            if save.button("Save", key=f"save_{case.id}", use_container_width=True):
                try:
                    workbench.replace_case(case.id, EvalCase(
                        id=case.id, input=text,
                        expected_output=expected.strip() or None,
                        context=tuple(l.strip() for l in context.splitlines() if l.strip()),
                        notes=case.notes,
                    ))
                except ValueError as error:
                    st.error(str(error))
                else:
                    st.rerun()
            if remove.button("Remove", key=f"rm_{case.id}", use_container_width=True):
                workbench.remove_case(case.id)
                st.rerun()


def _add_case_by_hand(workbench: Session) -> None:
    """Writing a case down needs no provider key, and is often the better start:
    the case you already know you care about is the one worth pinning first.

    A form, so the boxes empty themselves on submit. Left full, they read as a
    case still waiting to be added, and the second click adds it twice.
    """
    with st.expander("Write a case by hand"), st.form("new_case", clear_on_submit=True):
        text = st.text_area("Input", key="new_case_input",
                            placeholder="What the prompt will actually receive")
        expected = st.text_input(
            "Expected output (optional)", key="new_case_expected",
            help="Only where the job has one right answer — a label, a destination, a field set.",
        )
        context = st.text_area(
            "Context, one per line (optional)", key="new_case_context",
            help="Background the prompt is given. Needed by metrics like faithfulness.",
        )
        if st.form_submit_button("Add this case"):
            try:
                workbench.add_case(
                    input=text, expected_output=expected,
                    context=tuple(l.strip() for l in context.splitlines() if l.strip()),
                )
            except ValueError as error:
                st.error(str(error))
            else:
                st.rerun()


def _generate_cases(workbench: Session, count: int) -> None:
    task = workbench.task_type
    assert task is not None
    try:
        with st.spinner("Writing test cases…"):
            cases = case_intake.generate_cases(
                description=workbench.description, task=task, count=count,
                complete=session.completion(), model=session.authoring_model(),
                settings=None, new_id=workbench.new_id,
            )
    except Exception as error:  # noqa: BLE001
        st.error(f"Generating test cases failed: {error}")
        return
    workbench.set_cases(cases)
    st.rerun()


# --- 3. variants ----------------------------------------------------------


def _variants(workbench: Session) -> None:
    st.subheader("3 · Prompt variants")
    task = workbench.task_type
    if task is None:
        st.info("Pick the kind of job first.")
        return

    st.caption(
        "One prompt per approach, chosen for this kind of job. The point is to "
        "find which approach wins, so they are written to differ."
    )
    columns = st.columns(len(task.variants))
    chosen = []
    for column, approach in zip(columns, task.variants, strict=True):
        with column:
            if st.checkbox(approach.label, value=True, key=f"ap_{task.key}_{approach.key}",
                           help=approach.instruction):
                chosen.append(approach)

    generate_blocked = not session.has_credentials() or not chosen or not workbench.cases
    reason = (
        NO_KEY if not session.has_credentials()
        else "Add a test case above first." if not workbench.cases
        else None
    )
    if st.button("Write the variants", type="primary", disabled=generate_blocked, help=reason):
        _generate_variants(workbench, tuple(chosen))
    if reason:
        st.caption(reason)

    _add_variant_by_hand(workbench)

    for variant in workbench.variants:
        with st.expander(f"{variant.label} · revision {variant.revision}"):
            text = st.text_area(
                "System prompt", value=variant.system_prompt, height=260,
                key=f"var_{variant.id}_{variant.revision}", label_visibility="collapsed",
            )
            if st.button("Save edit", key=f"save_var_{variant.id}"):
                if text.strip() and text != variant.system_prompt:
                    workbench.edit_variant(variant.id, text)
                    st.rerun()


def _add_variant_by_hand(workbench: Session) -> None:
    """Bring the prompt you already have.

    The point of the sweep is to find out whether any approach beats what you
    are doing today, and that comparison is impossible if today's prompt cannot
    be entered. Needs no provider key.
    """
    with st.expander("Paste a prompt you already have"):
        st.caption(
            "It joins the comparison as its own variant, and becomes the baseline "
            "every generated approach has to beat to be worth adopting."
        )
        with st.form("new_variant", clear_on_submit=True):
            text = st.text_area(
                "System prompt", key="new_variant_prompt", height=180,
                placeholder="Paste your current prompt here",
            )
            if st.form_submit_button("Add this prompt"):
                try:
                    workbench.add_variant(text)
                except ValueError as error:
                    st.error(str(error))
                else:
                    st.rerun()


def _generate_variants(workbench: Session, approaches: Sequence[VariantApproach]) -> None:
    brief = workbench.brief
    task = workbench.task_type
    if brief is None or task is None:
        return
    try:
        with st.spinner(f"Writing {len(approaches)} prompt(s)…"):
            written = variants.generate(
                case=brief, task=task, approaches=approaches,
                complete=session.completion(), model=session.authoring_model(),
                settings=None, new_id=workbench.new_id, clock=workbench.clock,
            )
    except Exception as error:  # noqa: BLE001
        st.error(f"Writing variants failed: {error}")
        return
    workbench.set_variants(written, keep_hand_written=True)
    st.rerun()


# --- 4. metrics -----------------------------------------------------------


def _metrics(workbench: Session) -> None:
    st.subheader("4 · Metrics")
    if workbench.task_type is None:
        st.info("Pick the kind of job first.")
        return

    if not deepeval_metrics.is_available():
        st.warning(
            "The metric layer is deepeval, an optional extra so the base install "
            f"stays small. Install it with `{deepeval_metrics.INSTALL_HINT}`, then "
            "reload. Everything above works without it."
        )
        return

    st.caption(
        "These are deepeval metrics. What you settle on here is what you write "
        "into the other project's test suite — same classes, same thresholds."
    )

    for choice in workbench.metrics:
        spec = choice.spec
        direction = "higher is better" if spec.higher_is_better else "lower is better"
        with st.expander(
            f"{'☑' if choice.enabled else '☐'} {spec.label} · {direction} · "
            f"threshold {choice.threshold:g}"
        ):
            st.caption(spec.purpose)
            if spec.note:
                st.caption(spec.note)

            enabled = st.checkbox("Enabled", value=choice.enabled, key=f"me_{choice.key}")
            threshold = st.slider(
                "Threshold", 0.0, 1.0, float(choice.threshold), 0.05, key=f"mt_{choice.key}"
            )
            updated = replace(choice, enabled=enabled, threshold=threshold)

            if spec.custom_criteria:
                steps = st.text_area(
                    "Evaluation steps, one per line",
                    value="\n".join(choice.evaluation_steps),
                    key=f"ms_{choice.key}",
                    help="More stable between runs than a single criteria sentence.",
                )
                updated = replace(
                    updated,
                    evaluation_steps=tuple(l.strip() for l in steps.splitlines() if l.strip()),
                )

            config = dict(choice.config)
            for field_spec in spec.config_fields:
                config[field_spec.name] = st.text_area(
                    field_spec.label, value=config.get(field_spec.name, ""),
                    help=field_spec.help, key=f"mc_{choice.key}_{field_spec.name}",
                )
            updated = replace(updated, config=config)

            if updated != choice:
                workbench.update_metric(choice.key, updated)
                st.rerun()

            st.code(updated.as_code(), language="python")
            if st.button("Remove", key=f"mr_{choice.key}"):
                workbench.remove_metric(choice.key)
                st.rerun()

    available = [k for k in deepeval_metrics.METRIC_SPECS if k not in {c.key for c in workbench.metrics}]
    if available:
        add_col, button_col = st.columns([3, 1])
        extra = add_col.selectbox(
            "Add a metric", available,
            format_func=lambda k: deepeval_metrics.get(k).label, key="add_metric",
        )
        button_col.write("")
        if button_col.button("Add", use_container_width=True):
            workbench.add_metric(extra)
            st.rerun()

    brief = workbench.brief
    if brief is not None:
        for problem in scoring.preflight(brief=brief, choices=workbench.metrics):
            st.warning(problem)


# --- 5. the sweep ---------------------------------------------------------


def _sweep(workbench: Session) -> None:
    st.subheader("5 · Sweep")
    if not workbench.variants or not workbench.cases:
        st.info("Add at least one prompt variant above — generate them, or paste one you have.")
        return
    if not deepeval_metrics.is_available():
        st.info("Install the deepeval extra to score a sweep.")
        return

    registry = session.registry()
    priced = [e.id for e in registry.recommended() if e.has_price]
    models = st.multiselect(
        "Models to try", priced, default=workbench.sweep_models or priced[:2],
        help="Cheapest first. The point is to find the least expensive one that passes.",
    )
    workbench.sweep_models = tuple(models)

    variant_ids = st.multiselect(
        "Variants to try",
        [v.id for v in workbench.variants],
        default=[v.id for v in workbench.variants],
        format_func=lambda vid: workbench.variant(vid).label,
    )

    plan = sweep.plan(
        variant_keys=tuple(variant_ids), model_ids=tuple(models),
        case_count=len(workbench.cases),
        judged_metric_count=workbench.judged_metric_count,
        registry=registry,
    )
    st.info(plan.summary())
    if plan.has_unpriced_model:
        st.caption("Some selected models publish no price, so the estimate is incomplete.")

    brief = workbench.brief
    problems = (
        scoring.preflight(brief=brief, choices=workbench.metrics) if brief is not None else ()
    )
    for problem in problems:
        st.warning(problem)

    # Every reason the run cannot happen is named. A primary button that is grey
    # for an unstated reason is the same bug as one that runs and does nothing.
    reason = (
        NO_CASE if brief is None
        else NO_KEY if not session.has_credentials()
        else "Pick at least one model and one variant."
        if not plan.is_runnable
        else "Enable at least one metric in step 4 — an unmeasured sweep decides nothing."
        if not workbench.enabled_metrics
        else f"{len(problems)} metric(s) in step 4 still need something. See the warnings there."
        if problems
        else None
    )
    if st.button("Run the sweep", type="primary", disabled=reason is not None, help=reason):
        _run_sweep(workbench, brief, tuple(variant_ids), tuple(models))
    if reason:
        st.caption(reason)

    if workbench.sweep_results:
        _render_results(workbench, workbench.sweep_results)


def _run_sweep(
    workbench: Session,
    brief: CaseBrief | None,
    variant_ids: tuple[str, ...],
    model_ids: tuple[str, ...],
) -> None:
    if brief is None:
        st.error(NO_CASE)
        return
    judge = session.judge()
    complete = session.completion_with_usage()
    progress = st.progress(0.0, text="Running…")
    total = max(len(variant_ids) * len(model_ids), 1)
    done = 0

    def run_cell(variant_id: str, model_id: str) -> sweep.SweepCell:
        nonlocal done
        variant = workbench.variant(variant_id)
        outcomes: list[scoring.MetricOutcome] = []
        spent_in = spent_out = 0
        for case in workbench.cases:
            record = one_shot.run_plain(
                system_prompt=variant.system_prompt, user_message=case.input,
                model=model_id, settings=workbench.settings, complete=complete,
            )
            spent_in += record[1].tokens_in
            spent_out += record[1].tokens_out
            for choice in workbench.enabled_metrics:
                outcomes.append(
                    scoring.score_one(case=case, response=record[0], choice=choice, judge=judge)
                )
        done += 1
        progress.progress(done / total, text=f"{done}/{total} combinations")
        return sweep.SweepCell(
            variant_key=variant_id, model_id=model_id, settings=workbench.settings,
            score=scoring.aggregate(outcomes),
            usage=TokenUsage(tokens_in=spent_in // max(len(workbench.cases), 1),
                             tokens_out=spent_out // max(len(workbench.cases), 1)),
            met_every_threshold=scoring.met_every_threshold(outcomes),
        )

    cells = sweep.run(variant_keys=variant_ids, model_ids=model_ids, run_cell=run_cell)
    progress.empty()
    workbench.sweep_results = cells
    st.rerun()


def _render_results(workbench: Session, cells: Sequence[SweepCell]) -> None:
    registry = session.registry()
    st.markdown("#### Results")
    for cell in sweep.rank(cells, registry=registry):
        label = workbench.variant(cell.variant_key).label if cell.variant_key else "?"
        cost = cell.cost_per_thousand(registry)
        cost_text = "cost unknown" if cost is None else f"${cost:.3f} / 1k calls"
        if cell.failed:
            st.error(f"**{label}** on `{cell.model_id}` failed: {cell.failure}")
        else:
            mark = "✓" if cell.met_every_threshold else "✗"
            score = "—" if cell.score is None else f"{cell.score:.2f}"
            st.markdown(f"{mark} **{label}** on `{cell.model_id}` — {score} · {cost_text}")

    winner = sweep.cheapest_passing(cells, registry=registry)
    st.divider()
    if winner is None:
        st.warning(
            "Nothing cleared every threshold. Lower a threshold you can defend, edit "
            "a variant, or try a more capable model."
        )
        return

    cost = winner.cost_per_thousand(registry)
    st.success(
        f"**Cheapest configuration that passed:** {workbench.variant(winner.variant_key).label} "
        f"on `{winner.model_id}`"
        + ("" if cost is None else f" at ${cost:.3f} per 1,000 calls")
    )
    st.markdown("Write this into the other project's test suite:")
    lines = [f"# model: {winner.model_id}", f"# settings: {workbench.settings}", "metrics = ["]
    lines += [f"    {choice.as_code()}," for choice in workbench.enabled_metrics]
    lines.append("]")
    st.code("\n".join(lines), language="python")
