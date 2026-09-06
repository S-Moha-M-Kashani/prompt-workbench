"""One round through the Anthropic Messages API.

Verified against the installed SDK (anthropic 1.4.0) rather than from memory:

- ``client.messages.create(model, max_tokens, system=..., messages=[...],
  tools=[{"name", "description", "input_schema"}])``
- a tool request arrives as a ``tool_use`` content block with ``.id``, ``.name``
  and ``.input``, and ``stop_reason == "tool_use"``
- the result goes back as a *user* message of ``tool_result`` blocks carrying
  ``tool_use_id``
- usage is ``response.usage.input_tokens`` / ``.output_tokens``

Two things differ from the other adapters and are load-bearing:

``max_tokens`` is **required** by this API, so a round that set none gets a
default rather than a 400 — and the number sent is visible in the code sketch.

This provider is not the workbench's provider. It takes its own session-scoped
credential and its models are priced from ``services.anthropic_catalog``, which
is presented as a separate catalogue. The SDK would fall back to an ambient
``ANTHROPIC_API_KEY`` if constructed without one, so it never is: a missing key
is refused by name instead.
"""

from __future__ import annotations

from typing import Any

from prompt_workbench.llm_call import mock_tools
from prompt_workbench.llm_call.base import InvocationOutcome, LlmCall, system_prompt_for
from prompt_workbench.models.call import CallRequest
from prompt_workbench.models.usage import TokenUsage

FRAMEWORK_KEY = "anthropic"

# Required by the Messages API. Used only when the round set none, so a request
# that never asked for a ceiling still gets one it can see in the sketch.
DEFAULT_MAX_TOKENS = 4096

# A round is one question, not an agent loop.
MAX_TOOL_ROUND_TRIPS = 4

# Parameters this API does not accept, dropped rather than sent and rejected.
# The provider catalogue's published-parameter list describes the workbench's
# own provider, not this one, so the filtering has to happen here.
UNSUPPORTED_PARAMETERS = frozenset({"frequency_penalty", "presence_penalty"})


class AnthropicCall(LlmCall):
    """Anthropic's own API, with its own credential and its own price list."""

    framework = FRAMEWORK_KEY

    def __init__(
        self,
        *,
        client: Any | None = None,
        api_key: str = "",
        base_url: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._client = client
        self._api_key = api_key
        self._base_url = base_url

    @property
    def client(self) -> Any:
        if self._client is not None:
            return self._client
        import anthropic

        if not self._api_key.strip():
            raise ValueError(
                "Anthropic is a second provider and needs its own API key, passed "
                "explicitly. This adapter never reads one from the environment."
            )
        self._client = anthropic.Anthropic(
            api_key=self._api_key.strip(), base_url=self._base_url
        )
        return self._client

    def _invoke(self, request: CallRequest) -> InvocationOutcome:
        recorder = mock_tools.TraceRecorder()
        callables = mock_tools.build_all(request.tools, recorder)
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": request.user_prompt}
        ]
        usages: list[TokenUsage] = []

        for _ in range(MAX_TOOL_ROUND_TRIPS + 1):
            response = self.client.messages.create(**self._params(request, messages))
            usages.append(_usage_of(response))
            blocks = list(getattr(response, "content", None) or [])
            requested = [b for b in blocks if getattr(b, "type", "") == "tool_use"]
            if not requested:
                answer = _answer_of(blocks)
                if not answer:
                    raise ValueError("The model returned an empty answer.")
                return InvocationOutcome(
                    answer=answer,
                    usages=tuple(usages),
                    tool_calls=recorder.trace(),
                )
            messages.append({"role": "assistant", "content": blocks})
            messages.append(
                {"role": "user", "content": _tool_results(requested, callables, recorder)}
            )

        raise RuntimeError(
            f"The model asked for tools more than {MAX_TOOL_ROUND_TRIPS} times; "
            "this bench measures one round, not an agent loop."
        )

    def _params(
        self, request: CallRequest, messages: list[dict[str, Any]]
    ) -> dict[str, Any]:
        settings = {
            name: value
            for name, value in request.settings.as_params().items()
            if name not in UNSUPPORTED_PARAMETERS
        }
        max_tokens = int(settings.pop("max_tokens", 0) or DEFAULT_MAX_TOKENS)
        params: dict[str, Any] = {
            "model": request.model_id,
            "max_tokens": max_tokens,
            "system": system_prompt_for(request),
            "messages": messages,
            **settings,
        }
        if request.sends_tools:
            params["tools"] = [_tool_schema(spec) for spec in request.tools]
        return params


def _tool_schema(spec: Any) -> dict[str, Any]:
    """This API's tool shape: no ``function`` wrapper, and ``input_schema``."""
    return {
        "name": spec.name,
        "description": spec.description,
        "input_schema": dict(spec.input_schema)
        if spec.input_schema
        else {"type": "object", "properties": {}},
    }


def _usage_of(response: Any) -> TokenUsage:
    usage = getattr(response, "usage", None)
    if usage is None:
        return TokenUsage()
    return TokenUsage(
        tokens_in=int(getattr(usage, "input_tokens", 0) or 0),
        tokens_out=int(getattr(usage, "output_tokens", 0) or 0),
    )


def _answer_of(blocks: list[Any]) -> str:
    """The text blocks joined; thinking and tool blocks are not the answer."""
    return "".join(
        getattr(block, "text", "") or ""
        for block in blocks
        if getattr(block, "type", "") == "text"
    ).strip()


def _tool_results(
    requested: list[Any],
    callables: dict[str, mock_tools.ToolCallable],
    recorder: mock_tools.TraceRecorder,
) -> list[dict[str, Any]]:
    """Every requested tool answered in one user message, under its own id."""
    results: list[dict[str, Any]] = []
    for call in requested:
        arguments = dict(getattr(call, "input", None) or {})
        run = callables.get(call.name)
        if run is None:
            recorder.record(call.name, arguments)
            content = f"tool {call.name} is not defined for this round"
        else:
            content = run(arguments)
        results.append(
            {"type": "tool_result", "tool_use_id": call.id, "content": content}
        )
    return results
