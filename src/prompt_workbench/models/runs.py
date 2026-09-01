"""One recorded run of a prompt under test.

Deliberately flat. An earlier design carried a chain of snapshot references —
brief, dataset, candidate — because artifacts were generated from each other and
provenance had to be reconstructed. With ready use cases there is no chain: a
run is a use case, the exact prompt text that was sent, a model, a message, and
what came back. Storing the prompt *text* rather than a pointer to it means a
run stays readable after the prompt is edited ten more times.
"""

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class PromptRun:
    """A single one-shot execution: what was sent, and what came back."""

    id: str
    use_case_key: str
    prompt_revision: int
    system_prompt: str
    model_id: str
    user_message: str
    response: str
    created_at: datetime
    total_tokens: int = 0

    @property
    def is_scoreable(self) -> bool:
        """Whether there is a response here worth asking a judge about."""
        return bool(self.response.strip())

    @property
    def label(self) -> str:
        first_line = self.user_message.strip().splitlines()[0] if self.user_message.strip() else ""
        return f"r{self.prompt_revision} · {first_line[:48]}"
