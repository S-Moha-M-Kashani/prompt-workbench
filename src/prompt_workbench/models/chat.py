"""Conversations — the only part of the workbench that carries implicit state.

Everything else here takes an explicit snapshot and returns a result. A chat
cannot: the third message only makes sense after the first two. So history is
confined to this one shape, keyed by thread, and no generator or evaluator is
ever handed one.

``ThreadConfig`` is what a testing thread was opened with. It compares by value,
which is how "you changed the model halfway through" becomes detectable rather
than a mystery in the transcript.
"""

from dataclasses import dataclass, replace
from datetime import datetime

from prompt_workbench.models.model_settings import ModelSettings
from prompt_workbench.models.protocols import Message
from prompt_workbench.models.provenance import SourceRef

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
class ThreadConfig:
    """The candidate, model, and settings a testing thread is pinned to."""

    candidate: SourceRef
    model_id: str
    settings: ModelSettings


@dataclass(frozen=True)
class ChatThread:
    """One isolated conversation with its complete short-term history."""

    id: str
    purpose: str
    created_at: datetime
    config: ThreadConfig | None = None
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
