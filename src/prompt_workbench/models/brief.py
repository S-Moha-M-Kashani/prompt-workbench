"""The use-case brief: what the system prompt under construction must do.

Eight fields, one per discovery topic, all free text. They are deliberately
plain strings rather than richer structures: the brief is written by a person in
a chat and read by a model, and every schema imposed between those two ends is
a place where a real requirement gets rounded off to fit.

``PromptBrief`` is the mutable-in-spirit draft the user edits; ``BriefSnapshot``
is what confirming one records — an immutable, numbered version that every
downstream artifact can point at.
"""

from dataclasses import dataclass, fields, replace
from datetime import datetime
from typing import Mapping

from prompt_workbench.models.provenance import SourceRef

# Order matters: it fixes how the brief is shown in the UI and rendered into a
# prompt, so two generations from the same brief read identically.
BRIEF_FIELDS: tuple[str, ...] = (
    "purpose",
    "audience",
    "inputs",
    "desired_behaviour",
    "constraints",
    "output_format",
    "examples",
    "failure_cases",
)

# Human labels, used both in the UI and in the text handed to a model.
FIELD_LABELS: dict[str, str] = {
    "purpose": "Purpose",
    "audience": "Audience",
    "inputs": "Inputs",
    "desired_behaviour": "Desired behaviour",
    "constraints": "Constraints",
    "output_format": "Output format",
    "examples": "Examples",
    "failure_cases": "Failure cases",
}

# What the discovery chat asks about when a field is still blank.
FIELD_QUESTIONS: dict[str, str] = {
    "purpose": "what the prompt is for and what a good outcome looks like",
    "audience": "who reads the output and what they already know",
    "inputs": "what the prompt will receive on each call",
    "desired_behaviour": "how the assistant should behave, in specifics",
    "constraints": "rules, limits, tone, and anything it must never do",
    "output_format": "the shape of the answer: prose, JSON, sections, length",
    "examples": "a concrete example of an input and a good response",
    "failure_cases": "the ways this could go wrong that matter most",
}


@dataclass(frozen=True)
class PromptBrief:
    """The eight answers that describe the prompt to be engineered."""

    purpose: str = ""
    audience: str = ""
    inputs: str = ""
    desired_behaviour: str = ""
    constraints: str = ""
    output_format: str = ""
    examples: str = ""
    failure_cases: str = ""

    def filled_fields(self) -> tuple[str, ...]:
        """Field names carrying real text, in ``BRIEF_FIELDS`` order."""
        return tuple(name for name in BRIEF_FIELDS if getattr(self, name).strip())

    def missing_fields(self) -> tuple[str, ...]:
        """Field names still blank — what discovery has left to ask about."""
        filled = set(self.filled_fields())
        return tuple(name for name in BRIEF_FIELDS if name not in filled)

    @property
    def is_empty(self) -> bool:
        return not self.filled_fields()

    def merged_with(self, updates: Mapping[str, str]) -> "PromptBrief":
        """A copy with ``updates`` applied, ignoring blank incoming values.

        Blank is treated as "the model had nothing to add here", not as "clear
        this field". A discovery reply that only learned the audience must not
        wipe the purpose the user typed a minute ago; clearing stays a
        deliberate edit in the UI.
        """
        known = {field.name for field in fields(self)}
        unknown = set(updates) - known
        if unknown:
            raise KeyError(f"Unknown brief field(s): {', '.join(sorted(unknown))}")
        kept = {name: value for name, value in updates.items() if value and value.strip()}
        return replace(self, **kept)


@dataclass(frozen=True)
class BriefSnapshot:
    """A confirmed brief, numbered and frozen so results stay reproducible."""

    id: str
    revision: int
    brief: PromptBrief
    created_at: datetime

    @property
    def ref(self) -> SourceRef:
        return SourceRef(id=self.id, revision=self.revision)

    def as_context(self) -> str:
        """The brief as labelled lines, blank fields omitted.

        Omitting rather than sending ``Audience:`` with nothing after it: an
        empty label reads to a model as "no audience", which is a different
        claim from "not specified".
        """
        return "\n".join(
            f"{FIELD_LABELS[name]}: {getattr(self.brief, name).strip()}"
            for name in self.brief.filled_fields()
        )


__all__ = ["BRIEF_FIELDS", "FIELD_LABELS", "FIELD_QUESTIONS", "BriefSnapshot", "PromptBrief"]
