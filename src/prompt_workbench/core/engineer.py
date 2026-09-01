"""The prompt-engineer side of the chat: discuss, then hand back a whole prompt.

Two things come back from a turn, for the same reason discovery worked that way:
a message to read, and — when there is one — a complete replacement prompt. It
is deliberately all-or-nothing. A diff or a "change line 4 to..." puts the work
of reassembling the prompt back on the user, and reassembly is where prompts
quietly lose a placeholder and start sending `{history}` to the model as text.

This side keeps conversation history. The other side, the one-shot runner,
keeps none — that asymmetry is the point of having two modes.
"""

from dataclasses import dataclass

from prompt_workbench.core.chat_memory import ThreadStore
from prompt_workbench.core.generation import GenerationError, parse_json_object
from prompt_workbench.core.prompt_registry import load_system_prompt
from prompt_workbench.models.model_settings import ModelSettings
from prompt_workbench.models.protocols import CompletionFn
from prompt_workbench.models.use_case import UseCase


@dataclass(frozen=True)
class EngineerReply:
    """One turn of the engineering conversation."""

    reply: str
    proposed_prompt: str

    @property
    def has_prompt(self) -> bool:
        return bool(self.proposed_prompt.strip())


def briefing(use_case: UseCase, prompt_under_test: str) -> str:
    """What the engineer is told before every turn."""
    blocks = [
        f"SITUATION\n{use_case.situation}",
        f"THE FAILURE THIS IS KNOWN TO PRODUCE\n{use_case.trap}",
    ]
    if use_case.mocks:
        mocked = "\n\n".join(
            f"{block.placeholder} — {block.label}\n{block.content}"
            for block in use_case.mocks
        )
        blocks.append(f"MOCKED CONTEXT THE PROMPT WILL RECEIVE\n{mocked}")
    blocks.append(
        "WHAT A GOOD RESPONSE MUST DO\n"
        + "\n".join(f"- {c}" for c in use_case.criteria)
    )
    if use_case.forbidden:
        blocks.append(
            "AND MUST NOT\n" + "\n".join(f"- {f}" for f in use_case.forbidden)
        )
    blocks.append(f"THE PROMPT AS IT STANDS\n{prompt_under_test}")
    return "\n\n".join(blocks)


def discuss(
    *,
    store: ThreadStore,
    thread_id: str,
    user_message: str,
    use_case: UseCase,
    prompt_under_test: str,
    complete: CompletionFn,
    model: str,
    settings: ModelSettings | None = None,
) -> EngineerReply:
    """One engineering turn: record the exchange, return the reply and any prompt.

    The user's turn is recorded before the call and the engineer's only after a
    reply that could be read, so a failed call leaves no half-exchange behind to
    be replayed as context next time.
    """
    system = "\n\n".join(
        [load_system_prompt("prompt_engineer"), briefing(use_case, prompt_under_test)]
    )
    store.append(thread_id, "user", user_message)
    messages = [{"role": "system", "content": system}, *store.provider_messages(thread_id)]

    raw = complete(messages, model=model, settings=settings)
    payload = parse_json_object(raw, what="prompt-engineer reply")

    reply = str(payload.get("reply", "")).strip()
    if not reply:
        raise GenerationError("The prompt engineer's reply contained no message")

    store.append(thread_id, "assistant", reply)
    return EngineerReply(
        reply=reply, proposed_prompt=str(payload.get("prompt", "") or "").strip()
    )
