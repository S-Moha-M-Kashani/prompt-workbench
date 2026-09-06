"""The round: what one LLM call is made of, and what it produced.

The unit this workbench measures is a single round — a system prompt, a user
prompt, optionally some mocked tools, optionally a shape for the answer, run
through a chosen framework. This section is where that round is assembled.

Streamlit only, as everywhere in ``ui/``: the state and every consequence of
changing it live on ``core.session.Session``, which is why the rules about what
invalidates a result are unit-tested without a browser.
"""

from __future__ import annotations

import json

import streamlit as st

from prompt_workbench.core import code_sketch, considerations
from prompt_workbench.core.session import Session
from prompt_workbench.llm_call import registry as frameworks
from prompt_workbench.models.call import CallResult, OutputStructure, ToolSpec
from prompt_workbench.models.usage import TokenUsage
from prompt_workbench.services import anthropic_catalog, model_registry
from prompt_workbench.ui import session

FRAMEWORKS_KEY = "round_frameworks"
SYSTEM_PROMPT_KEY = "round_system_prompt"
USER_PROMPT_KEY = "round_user_prompt"
TOOLS_ON_KEY = "round_tools_on"
SHOW_SKETCH_KEY = "show_sketch"
ROUND_MODEL_KEY = "round_model_pick"
MODEL_SEARCH_KEY = "round_model_search"
EXTRA_PARAM_KEY = "extra_param_pick"
EXTRA_VALUE_KEY = "extra_param_value"
SHAPE_ON_KEY = "round_shape_on"
SHAPE_NAME_KEY = "round_shape_name"
SHAPE_SCHEMA_KEY = "round_shape_schema"

# The traffic shape the cost-per-thousand estimate assumes before any run has
# measured one. Shared with the registry's own ordering so the two numbers on
# screen cannot disagree about what they assume.
ASSUMED_USAGE = TokenUsage(
    tokens_in=model_registry.ASSUMED_TOKENS_IN,
    tokens_out=model_registry.ASSUMED_TOKENS_OUT,
)

DEFAULT_SHAPE_NAME = "answer"
DEFAULT_SHAPE_SCHEMA = json.dumps(
    {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
    },
    indent=2,
)


def render(workbench: Session) -> None:
    """The round, assembled top to bottom."""
    st.subheader("The round")
    st.caption(
        "One LLM call as it will actually ship: a system prompt, a user prompt, "
        "optionally mocked tools and optionally a fixed answer shape. An agent "
        "that takes five rounds is brought here one round at a time."
    )
    _render_frameworks(workbench)
    _render_model(workbench)
    _render_second_catalogue(workbench)
    _render_prompts(workbench)
    _render_tools(workbench)
    render_output_structure(workbench)
    _render_run(workbench)
    _render_sketch(workbench)
    _render_result(workbench)
    _render_considerations()


def _render_considerations() -> None:
    """What the numbers above do and do not support, on the page that shows them.

    `docs/LIMITATIONS.md` exists and nobody reading a score opens it. These are
    the same strings, from `core.considerations`, so the two cannot drift.
    """
    with st.expander("What these numbers do and do not support", expanded=False):
        for item in considerations.all_considerations():
            st.markdown(f"**{item.title}**")
            st.caption(item.statement)


def _render_sketch(workbench: Session) -> None:
    """The round drawn as code, on request.

    Deliberately not a download and deliberately not runnable. A runnable file
    would be a second implementation to keep in step with the adapters, and it
    would drift the day after it was written. What is useful is the *shape* of
    the call, so each input is a typed placeholder whose value is revealed on
    demand rather than pasted into the listing.
    """
    if not st.checkbox("Show this call as code", key=SHOW_SKETCH_KEY):
        return
    if not workbench.round_is_runnable:
        st.caption("Fill the round in first — there is no call to draw yet.")
        return

    request = workbench.call_request()
    published = workbench.registry.published_parameters(workbench.round_model_id)
    for framework in workbench.framework_keys:
        if framework not in code_sketch.frameworks():
            continue
        segments = code_sketch.sketch(framework, request, published_parameters=published)
        st.markdown(f"**{framework}**")
        st.code("".join(segment.text for segment in segments), language="python")
        st.caption(
            "A representation of the call, not the workbench's own code. It is not "
            "offered as a file and will not run unchanged."
        )
        _render_placeholders(segments, framework)


def _render_placeholders(segments: tuple[code_sketch.Segment, ...], framework: str) -> None:
    """Each placeholder's current value, revealed only when asked for."""
    seen: set[str] = set()
    with st.expander("What the placeholders currently hold"):
        for segment in segments:
            if not segment.is_placeholder or segment.name in seen:
                continue
            seen.add(segment.name)
            st.markdown(f"`{segment.type_name}({segment.name})`")
            st.code(str(segment.value), language="text")
            if segment.note:
                st.caption(segment.note)


def _render_frameworks(workbench: Session) -> None:
    """Which runtimes to measure. An absent one names its install command."""
    entries = frameworks.all_frameworks()
    labels = {entry.key: entry.label for entry in entries}
    available = set(frameworks.available_keys())

    def label_for(key: str) -> str:
        return labels[key] + ("" if key in available else " — not installed")

    chosen = st.multiselect(
        "Frameworks",
        [entry.key for entry in entries],
        default=[key for key in workbench.framework_keys if key in available],
        format_func=label_for,
        key=FRAMEWORKS_KEY,
        help="The runtime the round is made through. A framework changes latency "
        "and token count far more than a rephrasing does.",
    )
    usable = [key for key in chosen if key in available]
    workbench.set_framework_keys(usable)

    for entry in entries:
        if entry.key in available:
            if entry.key in usable and entry.note:
                st.caption(f"**{entry.label}** — {entry.note}")
        else:
            st.warning(entry.unavailable_reason())
    unusable = [key for key in chosen if key not in available]
    if unusable:
        st.caption(
            "Not installed, so not selected: "
            + ", ".join(f"`{key}`" for key in unusable)
        )


def _render_second_catalogue(workbench: Session) -> None:
    """Anthropic's models, priced from Anthropic's own table, labelled as such.

    Shown beside the provider catalogue rather than merged into it: a Claude
    model priced from the workbench's provider list would be a fabricated
    number wearing the look of a measured one.
    """
    if not any(
        frameworks.get(key).reaches_own_provider
        for key in workbench.framework_keys
        if key in {entry.key for entry in frameworks.all_frameworks()}
    ):
        return
    st.markdown("**Anthropic's own catalogue**")
    st.caption(anthropic_catalog.CATALOGUE_NOTE)
    for entry in anthropic_catalog.all_models():
        st.markdown(
            f"`{entry.id}` — {entry.label} · "
            f"${entry.price_in_per_million:.2f} in / "
            f"${entry.price_out_per_million:.2f} out per 1M · "
            f"{entry.context_window:,} token context"
        )
    st.caption(anthropic_catalog.staleness_note())


def _render_prompts(workbench: Session) -> None:
    """Both prompts, both required.

    The system prompt is no longer optional: a round without one is not a
    configuration anyone ships, and measuring it would describe a call the
    user will never make.
    """
    st.session_state.setdefault(SYSTEM_PROMPT_KEY, workbench.system_prompt)
    st.session_state.setdefault(USER_PROMPT_KEY, workbench.user_prompt)
    left, right = st.columns(2)
    with left:
        system_prompt = st.text_area(
            "Round system prompt",
            height=200,
            key=SYSTEM_PROMPT_KEY,
            placeholder="Required. What the model is and what it must do.",
        )
    with right:
        user_prompt = st.text_area(
            "Round user prompt",
            height=200,
            key=USER_PROMPT_KEY,
            placeholder="Required. The message the round actually sends.",
        )
    workbench.set_prompts(system_prompt=system_prompt, user_prompt=user_prompt)

    task = workbench.task_type
    if task is not None and not workbench.system_prompt.strip():
        st.caption(
            f"The **{task.label}** starting kit can fill both of these in — "
            "it is a shape to edit, not a prompt to keep."
        )
    if task is not None and st.button("Fill in from the starting kit", key="apply_kit"):
        try:
            workbench.apply_kit(task)
        except Session.EditsWouldBeLost as refusal:
            st.session_state["kit_overwrite_asked"] = True
            st.warning(str(refusal))
        else:
            st.rerun()
    if st.session_state.get("kit_overwrite_asked") and task is not None:
        if st.button("Yes, replace my edits", key="apply_kit_force"):
            workbench.apply_kit(task, overwrite=True)
            st.session_state["kit_overwrite_asked"] = False
            st.rerun()


def _render_tools(workbench: Session) -> None:
    """The mocked tool surface: a name and a description each.

    Execution is always mocked — a tool announces itself, returns that text and
    records the call. What is measured is whether the model asked for the right
    tool, never whether real tool output is handled.
    """
    st.markdown("**Tools** (optional, always mocked)")
    wanted = st.checkbox(
        "Send a tool surface with this round",
        key=TOOLS_ON_KEY,
        help="Each tool prints `tool <name> was called` and returns that text.",
    )
    if not wanted:
        workbench.set_tools(())
        st.caption("No tool surface is sent, and the result says the round sent none.")
        return

    count = st.number_input(
        "How many tools", min_value=1, max_value=6, value=max(1, len(workbench.tools)),
        step=1, key="round_tool_count",
    )
    defined: list[ToolSpec] = []
    for index in range(int(count)):
        existing = workbench.tools[index] if index < len(workbench.tools) else None
        left, right = st.columns([1, 2])
        with left:
            name = st.text_input(
                "Tool name",
                value=existing.name if existing else "",
                key=f"tool_name_{index}",
                placeholder="lookup_customer",
            )
        with right:
            description = st.text_area(
                "Tool description",
                value=existing.description if existing else "",
                height=68,
                key=f"tool_desc_{index}",
                placeholder="What the model should use it for.",
            )
        if name.strip():
            defined.append(ToolSpec(name=name.strip(), description=description.strip()))
    workbench.set_tools(tuple(defined))
    st.caption(
        "Mocked tools measure whether a tool was asked for correctly. They do not "
        "measure whether real tool output is handled — that is a different bench."
    )


def _render_run(workbench: Session) -> None:
    """One control, disabled with its reason rather than silently inert."""
    reasons = list(workbench.round_blockers)
    if not session.has_credentials():
        reasons.insert(0, "Add a provider API key in the sidebar.")
    disabled = bool(reasons)
    if st.button(
        "Run the round",
        type="primary",
        disabled=disabled,
        help=" ".join(reasons) if reasons else None,
        key="run_round",
    ):
        _run_round(workbench)


def _run_round(workbench: Session) -> None:
    """Run the round through the first selected framework and keep the result."""
    try:
        request = workbench.call_request()
    except (ValueError, Session.ToolsUnsupported) as refusal:
        st.error(str(refusal))
        return
    framework = workbench.framework_keys[0]
    with st.spinner(f"Running one round through {framework}…"):
        runner = session.call_runner(framework)
        workbench.last_result = runner.run(request)


def _render_result(workbench: Session) -> None:
    """What the round cost, and nothing it did not actually report."""
    result = workbench.last_result
    if result is None:
        return
    st.markdown("**Result**")
    if result.failed:
        st.error(
            f"The round failed: {result.error}\n\nNo latency, token or cost figure "
            "is shown, because none of them was measured."
        )
        return

    st.markdown(result.answer)
    latency = f"{result.latency_ms:,.0f} ms" if result.latency_ms is not None else "unknown"
    entry = workbench.registry.find(result.model_id)
    per_thousand = entry.cost_per_thousand(result.usage) if entry is not None else None
    cost = f"${per_thousand:,.2f}" if per_thousand is not None else "unknown"

    columns = st.columns(5)
    columns[0].metric("Latency", latency)
    columns[1].metric("Tokens in", f"{result.usage.tokens_in:,}")
    columns[2].metric("Tokens out", f"{result.usage.tokens_out:,}")
    columns[3].metric("Model calls", str(result.model_calls))
    columns[4].metric("Cost per 1,000 calls", cost)

    st.caption(f"Framework: **{result.framework}** · Model: `{result.model_id}`")
    if not result.usage.is_reported:
        st.caption("The framework reported no token counts, so the cost is unknown, not zero.")
    if result.latency_ms is not None:
        st.caption(
            "One run's latency carries network and provider variance. It is an "
            "observation, not a benchmark."
        )
    _render_trace(result)


def _render_trace(result: CallResult) -> None:
    if not result.tool_calls:
        st.caption("Tool trace: this round called no tools.")
        return
    st.markdown("**Tool trace**, in the order it happened")
    for call in result.tool_calls:
        st.markdown(f"{call.order + 1}. `{call.name}` with `{dict(call.arguments)}`")


def _render_model(workbench: Session) -> None:
    """The model under test — not the authoring model in the sidebar.

    The whole catalogue is reachable, searchable, with the curated shortlist
    pinned on top. Filtering the long tail out would be the workbench deciding
    for the user; ordering it is advice.
    """
    registry = workbench.registry
    query = st.text_input(
        "Search models",
        key=MODEL_SEARCH_KEY,
        placeholder="gpt-5, haiku, qwen…",
        help="Searches the provider's whole list by identifier or name. "
        "Leave it empty for the full catalogue, recommendations first.",
    )
    matches = registry.search(query)
    if not matches:
        st.warning(f"No model in the catalogue matches {query!r}.")
        return

    ids = [entry.id for entry in matches]
    chosen = st.selectbox(
        "Model under test",
        ids,
        key=ROUND_MODEL_KEY,
        help="The model the round runs on. The sidebar's authoring model writes "
        "cases and variants instead.",
    )
    workbench.set_round_model(chosen)
    _render_model_panel(workbench, chosen)
    _render_further_parameters(workbench, chosen)


def _render_model_panel(workbench: Session, model_id: str) -> None:
    """What this model costs, what it can hold, and what it accepts."""
    registry = workbench.registry
    entry = registry.find(model_id)
    if entry is None:
        st.caption("This model is not in the catalogue, so nothing is known about it.")
        return

    curated = model_id in set(registry.curated_ids())
    if not curated:
        st.caption(
            "Outside the recommended shortlist — usable for runs and sweeps, "
            "but not presented as a recommendation."
        )

    per_thousand = entry.cost_per_thousand(ASSUMED_USAGE)
    cost = f"${per_thousand:,.2f}" if per_thousand is not None else "unknown"
    columns = st.columns(3)
    columns[0].metric(
        "Price per 1M tokens",
        f"${entry.price_in_per_million:.2f} in"
        if entry.has_price
        else "unknown",
        f"${entry.price_out_per_million:.2f} out" if entry.has_price else None,
    )
    columns[1].metric("Context window", f"{entry.context_window:,} tokens")
    columns[2].metric("Cost per 1,000 calls", cost)
    if per_thousand is None:
        columns[2].caption("The provider published no price, so this is unknown, not zero.")
    else:
        columns[2].caption(
            f"At an assumed {ASSUMED_USAGE.tokens_in:,} in / "
            f"{ASSUMED_USAGE.tokens_out:,} out, until a run measures it."
        )

    tools = "yes" if registry.supports_tools(model_id) else "no"
    shape = "yes" if registry.can_enforce_structure(model_id) else "no"
    st.caption(f"Tool use: **{tools}** · Enforces an answer shape: **{shape}**")

    published = registry.published_parameters(model_id)
    if published:
        st.caption("Publishes: " + ", ".join(f"`{name}`" for name in published))
    else:
        st.caption("This model publishes no parameter list, so none is offered.")
    st.caption(
        "Prices and capabilities may be stale — read from the bundled snapshot."
        if registry.is_stale
        else "Prices and capabilities are live from the provider's list."
    )


def _render_further_parameters(workbench: Session, model_id: str) -> None:
    """Any published parameter beyond the five sliders in the sidebar.

    The list offered is the model's own, so nothing here needs maintaining as
    the provider adds parameters — and a value for a parameter the model does
    not publish cannot be set in the first place.
    """
    registry = workbench.registry
    settings = workbench.settings
    with st.expander("Further parameters this model publishes"):
        for name, value in sorted(settings.extra.items()):
            row = st.columns([3, 1])
            row[0].code(f"{name} = {value!r}", language="python")
            if row[1].button("Remove", key=f"drop_param_{name}"):
                workbench.settings = registry.without_parameter(settings, name)
                st.rerun()

        addable = registry.addable_parameters(model_id, settings)
        name = st.selectbox(
            "Parameter",
            addable,
            key=EXTRA_PARAM_KEY,
            help="Exactly what this model publishes and does not already show.",
        )
        raw = st.text_input(
            "Value (JSON)",
            key=EXTRA_VALUE_KEY,
            placeholder='7, "END", ["a", "b"], true',
            help="Read as JSON, so a string needs quotes. The value is yours — "
            "the provider judges it, and its error is shown verbatim.",
        )
        if st.button("Add parameter", disabled=not (name and raw.strip())):
            workbench.settings = registry.with_parameter(
                model_id, settings, name, _as_value(raw)
            )
            st.rerun()


def _as_value(raw: str) -> object:
    """A JSON value, falling back to the text as typed."""
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return raw.strip()


def render_output_structure(workbench: Session) -> None:
    """The optional answer shape, and an honest statement of what holds it.

    The distinction is the whole point of the control. Where the model publishes
    a structured-output parameter the provider enforces the schema; where it
    does not, the schema is only appended to the system prompt and has to be
    checked afterwards by a metric. Showing the second as the first would turn a
    measurement into an assumption.
    """
    st.markdown("**Answer shape** (optional)")
    wanted = st.checkbox(
        "Ask for a fixed answer shape",
        key=SHAPE_ON_KEY,
        help="A JSON schema the answer must match.",
    )
    if not wanted:
        workbench.set_output_structure(None)
        st.caption("No shape is asked for, so the answer is free text and nothing is claimed of it.")
        return

    name = st.text_input("Shape name", value=DEFAULT_SHAPE_NAME, key=SHAPE_NAME_KEY)
    raw = st.text_area("JSON schema", value=DEFAULT_SHAPE_SCHEMA, height=160, key=SHAPE_SCHEMA_KEY)
    schema = _parsed(raw)
    if schema is None:
        st.error("That is not valid JSON, so no shape is being sent.")
        workbench.set_output_structure(None)
        return

    workbench.set_output_structure(
        OutputStructure(name=name.strip() or DEFAULT_SHAPE_NAME, schema=schema)
    )
    if workbench.structure_is_enforced:
        st.success(workbench.structure_note)
    else:
        st.warning(workbench.structure_note)


def _parsed(raw: str) -> dict[str, object] | None:
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None
