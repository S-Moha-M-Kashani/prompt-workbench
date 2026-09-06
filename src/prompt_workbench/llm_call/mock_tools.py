"""Tools that announce themselves instead of doing anything.

Every tool in this workbench is mocked, always, including in a sweep. What is
being measured is whether the model asked for the right tool at the right
moment — not whether real tool output is handled, which is a different question
and needs a different bench.

So a tool is built once here and wrapped by each framework in its own tool
object. A difference in a recorded trace is then a difference in the framework
or the model, never in the tool.

The recorder is created per call and passed in, never module-global: two sweep
cells running the same tools must not share a trace.
"""

from collections.abc import Callable, Mapping
from typing import Any

from prompt_workbench.models.call import ToolInvocation, ToolSpec

ToolCallable = Callable[[Mapping[str, Any]], str]


def announcement(name: str) -> str:
    """What a mocked tool prints and returns."""
    return f"tool {name} was called"


class TraceRecorder:
    """The tool calls one round made, in the order they happened."""

    def __init__(self) -> None:
        self._calls: list[ToolInvocation] = []

    def record(self, name: str, arguments: Mapping[str, Any]) -> ToolInvocation:
        invocation = ToolInvocation(
            name=name, arguments=dict(arguments), order=len(self._calls)
        )
        self._calls.append(invocation)
        return invocation

    def trace(self) -> tuple[ToolInvocation, ...]:
        return tuple(self._calls)

    def __len__(self) -> int:
        return len(self._calls)


def build(spec: ToolSpec, recorder: TraceRecorder) -> ToolCallable:
    """The callable behind ``spec``: announce, record, return the announcement."""

    def call(arguments: Mapping[str, Any] | None = None) -> str:
        recorder.record(spec.name, arguments or {})
        text = announcement(spec.name)
        print(text)
        return text

    call.__name__ = spec.name
    call.__doc__ = spec.description
    return call


def build_all(
    specs: tuple[ToolSpec, ...], recorder: TraceRecorder
) -> dict[str, ToolCallable]:
    """Every tool in the round, by name, sharing one recorder."""
    return {spec.name: build(spec, recorder) for spec in specs}


def json_schema(spec: ToolSpec) -> dict[str, Any]:
    """``spec`` as the JSON function schema the OpenAI-shaped APIs expect."""
    return {
        "type": "function",
        "function": {
            "name": spec.name,
            "description": spec.description,
            "parameters": dict(spec.input_schema)
            if spec.input_schema
            else {"type": "object", "properties": {}},
        },
    }
