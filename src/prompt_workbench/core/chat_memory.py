"""Thread-scoped short-term memory, owned by this application.

A thread is a list of messages behind an id. That is the whole idea, and it is
implemented here rather than delegated because the workbench's purpose is to
make the *prompt* the only variable in a measurement — every framework between
the brief and the score is one more thing that could explain a result.

Two properties matter more than the storage:

- **Isolation.** Threads never see each other. A candidate tested in one thread
  cannot be helped by a conversation that happened in another, which is what
  makes two candidates comparable at all.
- **No silent trimming.** When history no longer fits, this raises. Dropping the
  earliest turns to make room would change the input without changing anything
  on screen, and the response would be attributed to a conversation that did not
  happen.
"""

from datetime import datetime, UTC
from collections.abc import Callable

from prompt_workbench.core import context_limits
from prompt_workbench.models.chat import ChatMessage, ChatThread, ThreadConfig
from prompt_workbench.models.identifiers import IdFactory, random_id


class ContextLimitReached(RuntimeError):
    """The thread plus the next message no longer fits the model's window."""


def _now() -> datetime:
    return datetime.now(UTC)


class ThreadStore:
    """Every live conversation in one session, keyed by thread id.

    ``new_id`` and ``clock`` are injected so a test can assert on an exact
    thread — the store is otherwise the one place in ``core`` that would be
    non-deterministic.
    """

    def __init__(
        self,
        *,
        new_id: IdFactory = random_id,
        clock: Callable[[], datetime] = _now,
    ) -> None:
        self._new_id = new_id
        self._clock = clock
        self._threads: dict[str, ChatThread] = {}

    def open(self, purpose: str, *, config: ThreadConfig | None = None) -> str:
        """Start an empty thread and return its id."""
        thread_id = self._new_id("thread")
        self._threads[thread_id] = ChatThread(
            id=thread_id, purpose=purpose, created_at=self._clock(), config=config
        )
        return thread_id

    def thread(self, thread_id: str) -> ChatThread:
        """The thread, or ``KeyError`` — never a silently empty stand-in."""
        try:
            return self._threads[thread_id]
        except KeyError:
            raise KeyError(f"No chat thread {thread_id!r}") from None

    def history(self, thread_id: str) -> tuple[ChatMessage, ...]:
        return self.thread(thread_id).messages

    def provider_messages(self, thread_id: str) -> list[dict[str, str]]:
        """This thread's history in the shape a provider call expects."""
        return self.thread(thread_id).as_provider_messages()

    def append(self, thread_id: str, role: str, content: str) -> ChatMessage:
        """Record one turn and return it."""
        message = ChatMessage(role=role, content=content, created_at=self._clock())
        self._threads[thread_id] = self.thread(thread_id).appended(message)
        return message

    def clear(self, thread_id: str) -> str:
        """Drop this thread and open an empty replacement with the same purpose.

        Returns the new id. The old id becomes invalid rather than being reused,
        so a stale reference fails loudly instead of appending to a conversation
        the user believes they ended.
        """
        old = self.thread(thread_id)
        del self._threads[thread_id]
        return self.open(old.purpose, config=old.config)

    def config_changed(self, thread_id: str, current: ThreadConfig) -> bool:
        """Whether the live configuration differs from what this thread was opened with."""
        return self.thread(thread_id).config != current

    def check_budget(
        self, thread_id: str, *, next_message: str, token_budget: int, system_prompt: str = ""
    ) -> int:
        """Estimated tokens for this thread plus the next turn, or raise.

        Raises ``ContextLimitReached`` rather than trimming. The message names
        the two things the user can actually do about it.
        """
        messages = self.provider_messages(thread_id)
        if system_prompt:
            messages = [{"role": "system", "content": system_prompt}, *messages]
        messages = [*messages, {"role": "user", "content": next_message}]
        used = context_limits.estimate_messages(messages)
        if used > token_budget * context_limits.BLOCK_AT:
            raise ContextLimitReached(
                f"This conversation is estimated at {used:,} tokens, which no longer "
                f"fits the selected model's {token_budget:,}-token window. Nothing has "
                "been dropped — clear the thread to start fresh, or switch to a model "
                "with a larger window."
            )
        return used
