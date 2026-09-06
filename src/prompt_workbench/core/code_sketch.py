"""The current round, drawn as code in the selected framework's own idiom.

A picture of the call, explicitly not the workbench's own code and explicitly
not an export. A runnable file would be a second implementation to keep in step
with the adapters, and it would drift the day after it was written.

Two rules keep it honest, and both are asserted by tests:

- it is generated from the same ``CallRequest`` the adapter receives, so it
  cannot describe a call that is not the one being made;
- exactly the parameters in the request appear, and only those the selected
  model publishes — a line for something that would never be sent is a lie in
  the shape of documentation.

Inputs are named as typed placeholders rather than inlined: a 1,900-word user
prompt would drown the sketch, and the *shape* of the call is the thing worth
reading. The UI reveals a placeholder's current value on demand and knows
nothing about frameworks; that knowledge lives here, where it is testable.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from prompt_workbench.models.call import CallRequest

#: Parameters that are part of the call's structure rather than its settings,
#: so they are drawn as their own lines instead of in the parameter list.
STRUCTURAL_PARAMETERS = frozenset({"tools", "tool_choice", "response_format"})


@dataclass(frozen=True)
class Segment:
    """One piece of the sketch: literal code, or a placeholder for an input."""

    text: str
    name: str = ""
    type_name: str = ""
    value: Any = None
    note: str = ""

    @property
    def is_placeholder(self) -> bool:
        return bool(self.name)


def sketch(
    framework: str,
    request: CallRequest,
    *,
    published_parameters: Sequence[str] | None = None,
) -> tuple[Segment, ...]:
    """The round as ordered segments, in ``framework``'s idiom.

    ``published_parameters`` narrows the parameter lines to what the selected
    model accepts. ``None`` means "do not narrow" — the request's own settings
    were already filtered at the registry boundary when a model was chosen.
    """
    try:
        draw = _DRAWERS[framework]
    except KeyError:
        raise KeyError(f"No code sketch for framework {framework!r}") from None
    return tuple(draw(request, _parameters(request, published_parameters)))


def frameworks() -> tuple[str, ...]:
    return tuple(_DRAWERS)


# --- the pieces every framework needs -------------------------------------


def _parameters(
    request: CallRequest, published: Sequence[str] | None
) -> tuple[tuple[str, Any], ...]:
    """The settings this call carries, minus anything structural or unpublished."""
    allowed = None if published is None else set(published)
    return tuple(
        (name, value)
        for name, value in request.settings.as_params().items()
        if name not in STRUCTURAL_PARAMETERS
        and (allowed is None or name in allowed)
    )


def _literal(text: str) -> Segment:
    return Segment(text=text)


def _hole(name: str, type_name: str, value: Any, note: str = "") -> Segment:
    """A named, typed placeholder: `str(user_prompt)` rather than the prose."""
    return Segment(
        text=f"{type_name}({name})", name=name, type_name=type_name, value=value, note=note
    )


def _prompt_holes(request: CallRequest) -> list[Segment]:
    return [
        _hole("system_prompt", "str", request.system_prompt),
        _hole("user_prompt", "str", request.user_prompt),
    ]


def _tools_hole(request: CallRequest) -> Segment:
    return _hole(
        "tools",
        "list",
        list(request.tool_names),
        note="Mocked: each tool announces itself, returns that text and records the call.",
    )


def _schema_hole(request: CallRequest) -> Segment:
    assert request.output_structure is not None
    return _hole(
        "output_schema",
        "dict",
        dict(request.output_structure.schema),
        note="Enforced by the provider for this model.",
    )


def _parameter_lines(
    parameters: tuple[tuple[str, Any], ...], indent: str
) -> list[Segment]:
    lines: list[Segment] = []
    for name, value in parameters:
        lines.append(_literal(f"{indent}{name}="))
        lines.append(_hole(name, type(value).__name__, value))
        lines.append(_literal(",\n"))
    return lines


def _enforces(request: CallRequest) -> bool:
    return request.output_structure is not None and request.structure_is_enforced


def _shape_comment(request: CallRequest) -> list[Segment]:
    """The note that an unenforced shape rides in the system prompt."""
    if request.output_structure is None or request.structure_is_enforced:
        return []
    return [
        _literal(
            "# This model publishes no structured-output parameter, so the shape\n"
            "# is appended to the system prompt and checked afterwards by a metric.\n"
        )
    ]


# --- one drawer per framework ---------------------------------------------


def _openai(
    request: CallRequest, parameters: tuple[tuple[str, Any], ...]
) -> list[Segment]:
    system, user = _prompt_holes(request)
    out = [_literal("from openai import OpenAI\n\nclient = OpenAI(base_url=..., api_key=...)\n\n")]
    out += _shape_comment(request)
    out.append(_literal("response = client.chat.completions.create(\n    model="))
    out.append(_hole("model", "str", request.model_id))
    out.append(_literal(",\n    messages=[\n        {\"role\": \"system\", \"content\": "))
    out.append(system)
    out.append(_literal("},\n        {\"role\": \"user\", \"content\": "))
    out.append(user)
    out.append(_literal("},\n    ],\n"))
    if request.sends_tools:
        out.append(_literal("    tools="))
        out.append(_tools_hole(request))
        out.append(_literal(",\n"))
    if _enforces(request):
        out.append(_literal("    response_format={\"type\": \"json_schema\", \"json_schema\": "))
        out.append(_schema_hole(request))
        out.append(_literal("},\n"))
    out += _parameter_lines(parameters, "    ")
    out.append(_literal(")\n"))
    return out


def _langchain(
    request: CallRequest, parameters: tuple[tuple[str, Any], ...]
) -> list[Segment]:
    system, user = _prompt_holes(request)
    out = [
        _literal(
            "from langchain.agents import create_agent\n"
            "from langchain_openai import ChatOpenAI\n\nmodel = ChatOpenAI(\n    model="
        )
    ]
    out.append(_hole("model", "str", request.model_id))
    out.append(_literal(",\n    base_url=..., api_key=...,\n"))
    out += _parameter_lines(parameters, "    ")
    out.append(_literal(")\n\n"))
    out += _shape_comment(request)
    out.append(_literal("agent = create_agent(\n    model=model,\n"))
    if request.sends_tools:
        out.append(_literal("    tools="))
        out.append(_tools_hole(request))
        out.append(_literal(",\n"))
    out.append(_literal("    system_prompt="))
    out.append(system)
    out.append(_literal(",\n"))
    if _enforces(request):
        out.append(_literal("    response_format="))
        out.append(_schema_hole(request))
        out.append(_literal(",\n"))
    out.append(_literal(")\n\nstate = agent.invoke({\"messages\": [{\"role\": \"user\", \"content\": "))
    out.append(user)
    out.append(_literal("}]})\n"))
    return out


def _langgraph(
    request: CallRequest, parameters: tuple[tuple[str, Any], ...]
) -> list[Segment]:
    system, user = _prompt_holes(request)
    out = [
        _literal(
            "from langgraph.graph import END, START, MessagesState, StateGraph\n"
            "from langchain_openai import ChatOpenAI\n\nmodel = ChatOpenAI(\n    model="
        )
    ]
    out.append(_hole("model", "str", request.model_id))
    out.append(_literal(",\n    base_url=..., api_key=...,\n"))
    out += _parameter_lines(parameters, "    ")
    out.append(_literal(")\n"))
    if request.sends_tools:
        out.append(_literal("bound = model.bind_tools("))
        out.append(_tools_hole(request))
        out.append(_literal(")\n"))
    else:
        out.append(_literal("bound = model\n"))
    out.append(_literal("\n"))
    out += _shape_comment(request)
    if _enforces(request):
        out.append(_literal("# The shape is enforced by the provider: schema = "))
        out.append(_schema_hole(request))
        out.append(_literal("\n"))
    out.append(
        _literal(
            "\ndef think(state: MessagesState):\n"
            "    return {\"messages\": [bound.invoke(state[\"messages\"])]}\n\n"
            "builder = StateGraph(MessagesState)\n"
            "builder.add_node(\"think\", think)\n"
            "builder.add_edge(START, \"think\")\n"
        )
    )
    if request.sends_tools:
        out.append(
            _literal(
                "builder.add_node(\"act\", ToolNode("
            )
        )
        out.append(_tools_hole(request))
        out.append(
            _literal(
                "))\n"
                "builder.add_conditional_edges(\"think\", tools_condition,"
                " {\"tools\": \"act\", END: END})\n"
                "builder.add_edge(\"act\", \"think\")\n"
            )
        )
    else:
        out.append(_literal("builder.add_edge(\"think\", END)\n"))
    out.append(_literal("\nstate = builder.compile().invoke({\"messages\": [\n    SystemMessage("))
    out.append(system)
    out.append(_literal("),\n    HumanMessage("))
    out.append(user)
    out.append(_literal("),\n]})\n"))
    return out


_DRAWERS = {
    "openai": _openai,
    "langchain": _langchain,
    "langgraph": _langgraph,
}
