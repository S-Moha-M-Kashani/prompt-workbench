"""How much of a model's context window a conversation has used.

An estimate, and honest about it: the exact count depends on the provider's
tokenizer, and calling one just to draw a progress bar would mean shipping a
tokenizer per model family. Four characters per token is the long-standing rule
of thumb for English text, and it is close enough for the only decision made
here — warn early, block before the wall.

The estimate is deliberately conservative (it rounds up, and it charges for the
per-message overhead the provider adds), because being told "you have room" and
then getting a hard provider error is the failure this module exists to prevent.
"""

from prompt_workbench.models.protocols import Message

CHARACTERS_PER_TOKEN = 4

# Roles, separators, and message framing the provider adds around every turn.
TOKENS_PER_MESSAGE = 4

# Fractions of the window at which the UI changes its tone.
WARN_AT = 0.75
BLOCK_AT = 0.95


def estimate_tokens(text: str) -> int:
    """A conservative token estimate for one piece of text."""
    return -(-len(text) // CHARACTERS_PER_TOKEN)  # ceiling division


def estimate_messages(messages: list[Message]) -> int:
    """The estimated cost of a whole conversation, framing included."""
    return sum(
        estimate_tokens(message.get("content", "")) + TOKENS_PER_MESSAGE
        for message in messages
    )


def usage_fraction(used_tokens: int, budget: int) -> float:
    """How full the window is, as ``0.0``–``1.0``+; an unknown budget reads as empty."""
    return 0.0 if budget <= 0 else used_tokens / budget
