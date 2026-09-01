"""Candidate system prompts — the artifacts actually under test.

Five techniques, one candidate each. They exist as separate artifacts rather
than as one prompt with switches because the whole point of the workbench is to
put them side by side: same brief, same dataset, same model, different technique,
and a grade for each.

A candidate is immutable. Editing produces a new revision and flags the artifact
as hand-edited, so a later regeneration can warn instead of quietly overwriting
work.
"""

from dataclasses import dataclass, replace
from datetime import datetime
from enum import StrEnum

from prompt_workbench.models.provenance import SourceRef


class PromptTechnique(StrEnum):
    """The prompt-writing techniques the workbench generates one candidate for."""

    DIRECT = "direct"
    ROLE_BASED = "role_based"
    FEW_SHOT = "few_shot"
    STRUCTURED_OUTPUT = "structured_output"
    REASONING_GUIDED = "reasoning_guided"

    @property
    def label(self) -> str:
        return {
            "direct": "Direct / zero-shot",
            "role_based": "Role-based",
            "few_shot": "Few-shot",
            "structured_output": "Structured output",
            "reasoning_guided": "Reasoning-guided",
        }[self.value]

    @property
    def description(self) -> str:
        return {
            "direct": "States the task plainly with no persona and no examples.",
            "role_based": "Casts the model in an expert role whose habits carry the task.",
            "few_shot": "Teaches by demonstration, using cases from the visible dataset.",
            "structured_output": "Fixes the shape of the answer and holds the model to it.",
            "reasoning_guided": "Prescribes the steps to work through before answering.",
        }[self.value]

    @property
    def template_name(self) -> str:
        """The prompt template that generates this technique's candidate."""
        return f"candidate_{self.value}"


@dataclass(frozen=True)
class CandidatePrompt:
    """One generated system prompt, its technique, and where it came from."""

    id: str
    technique: PromptTechnique
    system_prompt: str
    revision: int
    source_brief: SourceRef
    created_at: datetime
    source_dataset: SourceRef | None = None
    edited: bool = False

    def __post_init__(self) -> None:
        if not self.system_prompt.strip():
            raise ValueError("A candidate needs a non-empty system prompt")

    @property
    def ref(self) -> SourceRef:
        return SourceRef(id=self.id, revision=self.revision)

    @property
    def label(self) -> str:
        suffix = " (edited)" if self.edited else ""
        return f"{self.technique.label}{suffix}"

    def with_text(self, system_prompt: str, *, edited_at: datetime) -> "CandidatePrompt":
        """A new revision carrying hand-edited text."""
        return replace(
            self,
            system_prompt=system_prompt,
            revision=self.revision + 1,
            edited=True,
            created_at=edited_at,
        )
