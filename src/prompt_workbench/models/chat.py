"""Conversations — the only part of the workbench that carries implicit state.

Everything else here takes an explicit snapshot and returns a result. A chat
cannot: the third message only makes sense after the first two. So history is
confined to this one shape, keyed by thread, and no generator or evaluator is
ever handed one.

Only the prompt-engineer conversation uses one. The one-shot runner keeps no
history at all, which is what makes its output evidence about the prompt rather
than about the conversation.
"""

from dataclasses import dataclass, replace
from datetime import datetime

from prompt_workbench.models.protocols import Message

ROLES = ("system", "user", "assistant")


@dataclass(frozen=True)
class ChatMessage:
    """One turn. The timestamp is for the transcript, never sent to a model."""

    role: str
    content: str
    created_at: datetime

    def __post_init__(self) -> None:
        if self.role not in ROLES:
            raise ValueError(f"Unknown message role {self.role!r}; expected one of {ROLES}")


@dataclass(frozen=True)
class ChatThread:
    """One isolated conversation with its complete short-term history."""

    id: str
    purpose: str
    created_at: datetime
    messages: tuple[ChatMessage, ...] = ()

    def appended(self, message: ChatMessage) -> "ChatThread":
        """A copy with ``message`` on the end; the original is untouched."""
        return replace(self, messages=self.messages + (message,))

    def as_provider_messages(self) -> list[Message]:
        """The history in the shape a provider call expects."""
        return [{"role": m.role, "content": m.content} for m in self.messages]

    @property
    def is_empty(self) -> bool:
        return not self.messages
