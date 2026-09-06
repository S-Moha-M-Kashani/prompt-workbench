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

from prompt_workbench.core.session import Session
from prompt_workbench.models.call import OutputStructure
from prompt_workbench.models.usage import TokenUsage
from prompt_workbench.services import model_registry

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
    _render_model(workbench)
    render_output_structure(workbench)


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
