"""Stand-ins for the provider SDK, shared by the adapter tests.

Kept out of the individual test modules because more than one of them needs the
same shape, and two copies of a fake drift the same way two copies of anything
else do. No test in this suite reaches the network; these objects are what it
reaches instead.
"""

from typing import Any


class FakeUsage:
    def __init__(self, tokens_in: int, tokens_out: int) -> None:
        self.prompt_tokens = tokens_in
        self.completion_tokens = tokens_out


class FakeToolCall:
    def __init__(self, name: str, arguments: str, call_id: str = "c1") -> None:
        self.id = call_id
        self.type = "function"
        self.function = type("F", (), {"name": name, "arguments": arguments})()


class FakeMessage:
    def __init__(
        self, content: str | None, tool_calls: list[FakeToolCall] | None = None
    ) -> None:
        self.content = content
        self.tool_calls = tool_calls
        self.role = "assistant"


class FakeResponse:
    def __init__(self, message: FakeMessage, usage: FakeUsage | None) -> None:
        self.choices = [type("C", (), {"message": message})()]
        self.usage = usage


class FakeClient:
    """Records every ``create`` it is asked for and replays queued responses."""

    def __init__(self, *responses: FakeResponse) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []
        self.chat = self
        self.completions = self

    def create(self, **kwargs: Any) -> FakeResponse:
        self.calls.append(kwargs)
        return self._responses.pop(0)


def answer(text: str, tokens: tuple[int, int] = (10, 4)) -> FakeResponse:
    """A plain reply with usage reported."""
    return FakeResponse(FakeMessage(text), FakeUsage(*tokens))


def asks_for_tool(
    name: str, arguments: str = '{"id": 7}', tokens: tuple[int, int] = (20, 6)
) -> FakeResponse:
    """A reply that asks for one tool instead of answering."""
    return FakeResponse(FakeMessage(None, [FakeToolCall(name, arguments)]), FakeUsage(*tokens))
