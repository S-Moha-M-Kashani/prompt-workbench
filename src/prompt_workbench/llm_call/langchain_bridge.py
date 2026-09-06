"""Conversions the LangChain and LangGraph adapters both need.

Its own module rather than one adapter importing the other's private names:
LangGraph is not a client of LangChain's agent, it is a sibling that happens to
share a message and tool vocabulary. Naming that shared vocabulary once is what
keeps a trace difference between the two a real difference.

Verified against langchain-core 1.6.2: ``AIMessage.usage_metadata`` carries
``input_tokens`` and ``output_tokens``; an assistant message has ``type == "ai"``.
"""

from __future__ import annotations

from typing import Any

from prompt_workbench.llm_call import mock_tools
from prompt_workbench.models.call import CallRequest, ToolSpec
from prompt_workbench.models.usage import TokenUsage


def as_langchain_tools(request: CallRequest, recorder: mock_tools.TraceRecorder) -> list[Any]:
    """The same mocked callables, wrapped in LangChain's tool objects.

    One tool implementation, four wrappers: a difference in a recorded trace is
    then a difference in the framework or the model, never in the tool.
    """
    if not request.sends_tools:
        return []
    from langchain_core.tools import StructuredTool

    built = mock_tools.build_all(request.tools, recorder)
    tools: list[Any] = []
    for spec in request.tools:
        run = built[spec.name]
        tools.append(
            StructuredTool.from_function(
                func=_as_kwargs_callable(run),
                name=spec.name,
                description=spec.description,
                args_schema=_args_schema(spec),
            )
        )
    return tools


def _args_schema(spec: ToolSpec) -> dict[str, Any]:
    """The tool's argument schema, permissive when the user defined none.

    ``args_schema=None`` would make LangChain derive the schema from the wrapper
    function, which takes ``**kwargs`` and therefore declares no fields — and
    every argument the model sent would be dropped before it reached the
    recorder. An empty-but-open object schema keeps the trace honest.
    """
    if spec.input_schema:
        return dict(spec.input_schema)
    return {"type": "object", "properties": {}, "additionalProperties": True}


def _as_kwargs_callable(run: mock_tools.ToolCallable) -> Any:
    """LangChain calls a tool with keyword arguments; the mock takes a mapping."""

    def call(**kwargs: Any) -> str:
        return run(kwargs)

    return call


def final_answer(messages: list[Any]) -> str:
    """The last assistant message with actual content."""
    for message in reversed(messages):
        if getattr(message, "type", "") != "ai":
            continue
        content = message.content
        text = content if isinstance(content, str) else _joined(content)
        if text.strip():
            return text.strip()
    return ""


def _joined(content: Any) -> str:
    """Content blocks flattened to text, for a model that returns a list."""
    if not isinstance(content, list):
        return str(content)
    parts = [
        block.get("text", "") if isinstance(block, dict) else str(block)
        for block in content
    ]
    return "".join(part for part in parts if part)


def usages(messages: list[Any]) -> tuple[TokenUsage, ...]:
    """One entry per model call, from what LangChain reported and nothing else."""
    usages: list[TokenUsage] = []
    for message in messages:
        if getattr(message, "type", "") != "ai":
            continue
        reported = getattr(message, "usage_metadata", None) or {}
        usages.append(
            TokenUsage(
                tokens_in=int(reported.get("input_tokens", 0) or 0),
                tokens_out=int(reported.get("output_tokens", 0) or 0),
            )
        )
    return tuple(usages)
