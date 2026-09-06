"""One round through a hand-built LangGraph graph: model node, tool node.

Verified against langgraph 1.2.11 and langchain-core 1.6.2 rather than from
memory:

- ``from langgraph.graph import StateGraph, START, END, MessagesState``
- ``from langgraph.prebuilt import ToolNode, tools_condition`` —
  ``tools_condition`` returns ``"tools"`` or ``"__end__"`` from the last message
- usage arrives as ``AIMessage.usage_metadata`` (``input_tokens`` /
  ``output_tokens``); tool requests as ``AIMessage.tool_calls``

The graph is written out — model node, conditional edge, tool node, back to the
model — rather than delegated to a prebuilt agent. `create_agent` *is* the
prebuilt agent, and an adapter that called it would measure the same thing twice
under two names. Two nodes is also the smallest graph that can show a tool
round-trip costing a second model call.

A recursion limit keeps a round a round: a model that keeps asking for tools is
stopped rather than allowed to become an agent loop by accident.
"""

from __future__ import annotations

from typing import Any

from prompt_workbench.llm_call import mock_tools
from prompt_workbench.llm_call.base import InvocationOutcome, LlmCall, system_prompt_for
from prompt_workbench.llm_call.langchain_bridge import (
    as_langchain_tools,
    final_answer,
    usages,
)
from prompt_workbench.models.call import CallRequest

FRAMEWORK_KEY = "langgraph"

# A round is one question. Each tool round-trip costs two graph steps, so this
# allows the same four round-trips the bare-SDK adapter does.
RECURSION_LIMIT = 10


class LangGraphCall(LlmCall):
    """A single-node graph plus a tool node, over one round."""

    framework = FRAMEWORK_KEY

    def __init__(
        self,
        *,
        chat_model: Any | None = None,
        api_key: str = "",
        base_url: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._chat_model = chat_model
        self._api_key = api_key
        self._base_url = base_url

    def _invoke(self, request: CallRequest) -> InvocationOutcome:
        from langchain_core.messages import HumanMessage, SystemMessage

        recorder = mock_tools.TraceRecorder()
        tools = as_langchain_tools(request, recorder)
        graph = self._compiled(request, tools)
        state = graph.invoke(
            {
                "messages": [
                    SystemMessage(content=system_prompt_for(request)),
                    HumanMessage(content=request.user_prompt),
                ]
            },
            {"recursion_limit": RECURSION_LIMIT},
        )
        messages = list(state.get("messages", []))
        answer = final_answer(messages)
        if not answer:
            raise ValueError("The graph produced no answer.")
        return InvocationOutcome(
            answer=answer,
            usages=usages(messages),
            tool_calls=recorder.trace(),
            structure_was_enforced=False,
        )

    def _compiled(self, request: CallRequest, tools: list[Any]) -> Any:
        """The graph: think, and where tools exist, act and think again."""
        from langgraph.graph import END, START, MessagesState, StateGraph
        from langgraph.prebuilt import ToolNode, tools_condition

        model = self._model_for(request)
        bound = model.bind_tools(tools) if tools else model

        def think(state: MessagesState) -> dict[str, list[Any]]:
            return {"messages": [bound.invoke(state["messages"])]}

        builder = StateGraph(MessagesState)
        builder.add_node("think", think)
        builder.add_edge(START, "think")
        if not tools:
            builder.add_edge("think", END)
            return builder.compile()

        builder.add_node("act", ToolNode(tools))
        builder.add_conditional_edges("think", tools_condition, {"tools": "act", END: END})
        builder.add_edge("act", "think")
        return builder.compile()

    def _model_for(self, request: CallRequest) -> Any:
        """The injected chat model, or one built for this session's credential."""
        if self._chat_model is not None:
            return self._chat_model
        from langchain_openai import ChatOpenAI
        from pydantic import SecretStr

        if not self._api_key.strip():
            raise ValueError(
                "LangGraph needs a provider API key passed explicitly; "
                "this adapter never reads one from the environment."
            )
        return ChatOpenAI(
            model=request.model_id,
            api_key=SecretStr(self._api_key),
            base_url=self._base_url,
            **request.settings.as_params(),
        )
