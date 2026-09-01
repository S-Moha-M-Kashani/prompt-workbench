"""One candidate prompt, written with one approach to one kind of job.

The approach is recorded rather than inferred, because the point of generating
several is to find out which *approach* wins — and a prompt that has been edited
by hand still belongs to the approach it started from, which is what makes the
comparison hold up after a round of editing.
"""

from dataclasses import dataclass, replace
from datetime import datetime


@dataclass(frozen=True)
class PromptVariant:
    """A prompt under test, and the approach it came from."""

    id: str
    approach_key: str
    approach_label: str
    task_type_key: str
    system_prompt: str
    revision: int
    created_at: datetime
    edited: bool = False

    def __post_init__(self) -> None:
        if not self.system_prompt.strip():
            raise ValueError(f"Variant {self.approach_label!r} has an empty prompt")

    @property
    def label(self) -> str:
        return f"{self.approach_label}{' (edited)' if self.edited else ''}"

    def with_text(self, system_prompt: str, *, edited_at: datetime) -> "PromptVariant":
        return replace(
            self,
            system_prompt=system_prompt,
            revision=self.revision + 1,
            edited=True,
            created_at=edited_at,
        )
