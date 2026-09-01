"""The clarification chat that turns a rough request into a structured brief.

Two things come back from every turn: a message for the user, and the model's
current best reading of the eight brief fields. Keeping them together is what
makes the brief visibly track the conversation — the user watches fields fill in
as they answer, instead of discovering at the end what was inferred from them.

The chat is the one place with history. The brief itself is still passed in
explicitly on every call rather than being reconstructed from the transcript, so
a hand-edit the user makes in the Define area is authoritative and is not
overwritten by what the model remembers saying three turns ago.
"""

from dataclasses import dataclass

from prompt_workbench.core.chat_memory import ThreadStore
from prompt_workbench.core.generation import GenerationError, parse_json_object
from prompt_workbench.core.prompt_registry import load_system_prompt
from prompt_workbench.models.brief import BRIEF_FIELDS, FIELD_LABELS, PromptBrief
from prompt_workbench.models.model_settings import ModelSettings
from prompt_workbench.models.protocols import CompletionFn


@dataclass(frozen=True)
class DiscoveryReply:
    """One clarification turn: what to show, and what it learned."""

    reply: str
    brief: PromptBrief
    ready: bool


def _brief_state(brief: PromptBrief) -> str:
    """What the model is told about the brief so far.

    Filled fields are quoted back so it builds on them; missing ones are listed
    by name so it knows what is left to ask about.
    """
    lines = ["Brief so far:"]
    if brief.is_empty:
        lines.append("(nothing recorded yet)")
    else:
        lines += [
            f"- {FIELD_LABELS[name]}: {getattr(brief, name).strip()}"
            for name in brief.filled_fields()
        ]
    missing = brief.missing_fields()
    lines.append(
        "Still blank: " + (", ".join(missing) if missing else "nothing — all eight are filled")
    )
    return "\n".join(lines)


def clarify(
    *,
    store: ThreadStore,
    thread_id: str,
    user_message: str,
    brief: PromptBrief,
    platform_instruction: str,
    complete: CompletionFn,
    model: str,
    settings: ModelSettings,
) -> DiscoveryReply:
    """Send one user message, record the exchange, return the reply and brief.

    The user's turn is recorded before the call and the assistant's only after a
    reply that could be read. A failed call therefore leaves the thread showing
    what the user said and no answer, rather than a half-turn that would be sent
    again as context next time.
    """
    system = "\n\n".join(
        [platform_instruction, load_system_prompt("discovery"), _brief_state(brief)]
    )
    store.append(thread_id, "user", user_message)
    messages = [{"role": "system", "content": system}, *store.provider_messages(thread_id)]

    raw = complete(messages, model=model, settings=settings)
    payload = parse_json_object(raw, what="clarification reply")

    reply = str(payload.get("reply", "")).strip()
    if not reply:
        raise GenerationError("The clarification reply contained no message for the user")

    updates = payload.get("brief") or {}
    if not isinstance(updates, dict):
        raise GenerationError("The clarification reply's brief was not an object")
    known = {name: str(updates.get(name, "") or "") for name in BRIEF_FIELDS}

    store.append(thread_id, "assistant", reply)
    return DiscoveryReply(
        reply=reply,
        brief=brief.merged_with(known),
        ready=bool(payload.get("ready", False)),
    )
