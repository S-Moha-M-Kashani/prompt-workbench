"""A LangChain chat model that answers from a script and records what it saw.

Built on ``langchain_core``'s own ``FakeMessagesListChatModel`` so the object
really is a ``BaseChatModel`` — ``create_agent`` type-checks its model, and a
duck-typed stand-in would pass the test while failing in the app.

``bind_tools`` records the names and returns ``self`` rather than a binding, so
one object can be inspected after the round. Imported only after an
``importorskip``, since it needs the optional extra.
"""

from typing import Any

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, BaseMessage


def ai(content: str, tokens: tuple[int, int] | None = (10, 4)) -> AIMessage:
    """A plain answer, with usage reported unless ``tokens`` is ``None``."""
    return AIMessage(
        content=content,
        usage_metadata=_usage(tokens),
    )


def ai_asks_for(
    calls: list[tuple[str, dict[str, Any]]], tokens: tuple[int, int] | None = (20, 6)
) -> AIMessage:
    """An answer that asks for tools instead of replying."""
    return AIMessage(
        content="",
        tool_calls=[
            {"name": name, "args": args, "id": f"call_{index}", "type": "tool_call"}
            for index, (name, args) in enumerate(calls)
        ],
        usage_metadata=_usage(tokens),
    )


def _usage(tokens: tuple[int, int] | None) -> dict[str, int] | None:
    if tokens is None:
        return None
    tokens_in, tokens_out = tokens
    return {
        "input_tokens": tokens_in,
        "output_tokens": tokens_out,
        "total_tokens": tokens_in + tokens_out,
    }


class FakeChatModel(FakeMessagesListChatModel):
    """Replays scripted messages; remembers the prompts and the bound tools."""

    seen: list[list[BaseMessage]] = []
    bound_tool_names: list[str] = []

    def __init__(self, responses: list[BaseMessage], **kwargs: Any) -> None:
        super().__init__(responses=responses, **kwargs)
        # Fresh per instance: Pydantic class defaults are shared otherwise, and
        # two fakes sharing a transcript is exactly the bug the recorder in
        # ``mock_tools`` exists to avoid.
        object.__setattr__(self, "seen", [])
        object.__setattr__(self, "bound_tool_names", [])

    def _generate(self, messages: list[BaseMessage], *args: Any, **kwargs: Any) -> Any:
        self.seen.append(list(messages))
        return super()._generate(messages, *args, **kwargs)

    def bind_tools(self, tools: Any, **kwargs: Any) -> "FakeChatModel":
        self.bound_tool_names.extend(
            getattr(tool, "name", getattr(tool, "__name__", str(tool))) for tool in tools
        )
        return self
