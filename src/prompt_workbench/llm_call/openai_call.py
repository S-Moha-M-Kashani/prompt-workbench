"""One round through the bare OpenAI SDK: no framework, just the API.

This is the baseline every other framework is compared against — what the call
costs in time and tokens when nothing sits between the prompt and the provider.
A tool round-trip is written out by hand here (ask, dispatch, ask again) rather
than hidden in a loop helper, because the number of model calls that round-trip
costs is one of the numbers this workbench exists to show.

The client is injected, already built with a session-scoped key. This module
never reads a credential from the environment, and a test hands in a fake object
with a ``create`` method and opens no socket.
"""

from __future__ import annotations

import json
from typing import Any

from prompt_workbench.llm_call import mock_tools
from prompt_workbench.llm_call.base import InvocationOutcome, LlmCall, system_prompt_for
from prompt_workbench.models.call import CallRequest
from prompt_workbench.models.usage import TokenUsage

# A round is one question, so a model that keeps asking for tools is stopped
# rather than allowed to become an agent loop by accident.
MAX_TOOL_ROUND_TRIPS = 4


class OpenAiCall(LlmCall):
    """The provider's own API, called directly."""

    framework = "openai"

    def __init__(self, *, client: Any, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._client = client

    def _invoke(self, request: CallRequest) -> InvocationOutcome:
        recorder = mock_tools.TraceRecorder()
        callables = mock_tools.build_all(request.tools, recorder)
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt_for(request)},
            {"role": "user", "content": request.user_prompt},
        ]
        usages: list[TokenUsage] = []

        for _ in range(MAX_TOOL_ROUND_TRIPS + 1):
            response = self._client.chat.completions.create(
                **self._params(request, messages)
            )
            usages.append(_usage_of(response))
            message = response.choices[0].message
            requested = list(getattr(message, "tool_calls", None) or [])
            if not requested:
                answer = (getattr(message, "content", "") or "").strip()
                if not answer:
                    raise ValueError("The model returned an empty answer.")
                return InvocationOutcome(
                    answer=answer,
                    usages=tuple(usages),
                    tool_calls=recorder.trace(),
                    structure_was_enforced=_enforces(request),
                )
            messages.append(_assistant_turn(message, requested))
            messages.extend(_tool_replies(requested, callables, recorder))

        raise RuntimeError(
            f"The model asked for tools more than {MAX_TOOL_ROUND_TRIPS} times; "
            "this bench measures one round, not an agent loop."
        )

    def _params(self, request: CallRequest, messages: list[dict[str, Any]]) -> dict[str, Any]:
        params: dict[str, Any] = {
            "model": request.model_id,
            "messages": messages,
            **request.settings.as_params(),
        }
        if request.sends_tools:
            params["tools"] = [mock_tools.json_schema(spec) for spec in request.tools]
        if _enforces(request) and request.output_structure is not None:
            params["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": request.output_structure.name,
                    "schema": dict(request.output_structure.schema),
                },
            }
        return params


def _enforces(request: CallRequest) -> bool:
    return request.output_structure is not None and request.structure_is_enforced


def _usage_of(response: Any) -> TokenUsage:
    """The provider's own count, never an estimate."""
    usage = getattr(response, "usage", None)
    if usage is None:
        return TokenUsage()
    return TokenUsage(
        tokens_in=int(getattr(usage, "prompt_tokens", 0) or 0),
        tokens_out=int(getattr(usage, "completion_tokens", 0) or 0),
    )


def _assistant_turn(message: Any, requested: list[Any]) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": getattr(message, "content", None),
        "tool_calls": [
            {
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.function.name,
                    "arguments": call.function.arguments,
                },
            }
            for call in requested
        ],
    }


def _tool_replies(
    requested: list[Any],
    callables: dict[str, mock_tools.ToolCallable],
    recorder: mock_tools.TraceRecorder,
) -> list[dict[str, Any]]:
    """Run each requested tool through the shared mock and answer the model."""
    replies: list[dict[str, Any]] = []
    for call in requested:
        name = call.function.name
        arguments = _arguments_of(call)
        run = callables.get(name)
        if run is None:
            # The model invented a tool. Record it so the trace stays honest and
            # tell the model, rather than failing the round on the model's error.
            recorder.record(name, arguments)
            content = f"tool {name} is not defined for this round"
        else:
            content = run(arguments)
        replies.append({"role": "tool", "tool_call_id": call.id, "content": content})
    return replies


def _arguments_of(call: Any) -> dict[str, Any]:
    raw = getattr(call.function, "arguments", "") or "{}"
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        return {"_unparsed": raw}
    return parsed if isinstance(parsed, dict) else {"_value": parsed}


__all__ = ["MAX_TOOL_ROUND_TRIPS", "OpenAiCall"]
