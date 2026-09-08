"""One round through LangChain's `create_agent`.

Verified against the installed libraries rather than from memory, because these
move fast and a wrong attribute name is a silent zero:

- ``from langchain.agents import create_agent`` (langchain 1.4.0)
- ``create_agent(model, tools, *, system_prompt=..., response_format=...)``
  returns a compiled graph; ``.invoke({"messages": [...]})`` returns a state
  dict whose ``messages`` holds the whole exchange
- usage arrives as ``AIMessage.usage_metadata`` with ``input_tokens`` and
  ``output_tokens``; tool requests as ``AIMessage.tool_calls``

The agent runs its own loop until the model stops asking for tools, so the
number of model calls is counted from the messages it produced rather than
assumed. That is the framework's own behaviour, and hiding it would defeat the
comparison this workbench exists to make.

The chat model is either injected (a scripted fake, in tests) or built here from
an explicitly passed key and endpoint. Nothing in this module reads an ambient
credential — ``ChatOpenAI`` would fall back to ``OPENAI_API_KEY`` if it were
constructed without one, so it never is.
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

FRAMEWORK_KEY = "langchain"


class LangChainCall(LlmCall):
    """LangChain's own agent, over one round."""

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
        from langchain.agents import create_agent

        recorder = mock_tools.TraceRecorder()
        tools = as_langchain_tools(request, recorder)
        agent = create_agent(
            model=self._model_for(request),
            tools=tools,
            system_prompt=system_prompt_for(request),
        )
        state = agent.invoke({"messages": [{"role": "user", "content": request.user_prompt}]})
        messages = list(state.get("messages", []))
        answer = final_answer(messages)
        if not answer:
            raise ValueError("The agent produced no answer.")
        return InvocationOutcome(
            answer=answer,
            usages=usages(messages),
            tool_calls=recorder.trace(),
            structure_was_enforced=False,
        )

    def _model_for(self, request: CallRequest) -> Any:
        """The injected chat model, or one built for this session's credential."""
        if self._chat_model is not None:
            return self._chat_model
        from langchain_openai import ChatOpenAI
        from pydantic import SecretStr

        if not self._api_key.strip():
            raise ValueError(
                "LangChain needs a provider API key passed explicitly; "
                "this adapter never reads one from the environment."
            )
        return ChatOpenAI(
            model=request.model_id,
            api_key=SecretStr(self._api_key),
            base_url=self._base_url,
            **request.settings.as_params(),
        )
