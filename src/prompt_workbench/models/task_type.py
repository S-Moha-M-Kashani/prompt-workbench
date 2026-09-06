"""What kind of job the prompt does — the one choice everything else follows from.

An earlier design offered the same five prompt "techniques" for every prompt. It
could not do better, because it never asked what the prompt was *for*: few-shot
is the obvious move for a classifier over a fixed label set and close to
pointless for open drafting, and no amount of comparing them side by side
recovers that.

So the task type is chosen first, and it proposes four things that are otherwise
guesswork: which prompt approaches are worth trying, what settings the job wants,
which models can plausibly do it, and what a good answer would even be measured
by. It also carries the answer to a question a prompt workbench is otherwise
structurally unable to ask — whether this job should be a prompt at all, or a
fine-tuned small model.

What it does *not* decide is how the call is made. The framework, the tools and
the answer shape are properties of a round and belong to the user. The task type
is a starting kit that fills the workspace in with something editable — which is
why a tool call stopped being a kind of job: it was never a category of work,
only a property any round can have.
"""

from dataclasses import dataclass
from enum import StrEnum

from prompt_workbench.models.model_settings import ModelSettings


class FineTuneVerdict(StrEnum):
    """Whether prompting is the right tool for this job at volume."""

    LIKELY = "likely"
    SOMETIMES = "sometimes"
    UNLIKELY = "unlikely"

    @property
    def label(self) -> str:
        return {
            "likely": "A fine-tuned small model would likely win",
            "sometimes": "A fine-tune is sometimes worth it",
            "unlikely": "Prompting is the right tool here",
        }[self.value]


@dataclass(frozen=True)
class VariantApproach:
    """One way of writing the prompt for this kind of job."""

    key: str
    label: str
    instruction: str


@dataclass(frozen=True)
class TaskType:
    """A kind of job, and everything that follows from it."""

    key: str
    label: str
    description: str
    variants: tuple[VariantApproach, ...]
    metric_keys: tuple[str, ...]
    suggested_settings: ModelSettings
    settings_note: str
    fine_tune: FineTuneVerdict
    fine_tune_note: str
    needs_strong_model: bool = False
    keywords: tuple[str, ...] = ()
    #: The starting kit this job fills the workspace in with. A pointer rather
    #: than the content: the catalogue holds structural facts, and prompt text
    #: belongs in a data file where it can be read and reviewed as prose.
    preset_key: str = ""

    def variant(self, key: str) -> VariantApproach:
        for approach in self.variants:
            if approach.key == key:
                return approach
        raise KeyError(f"No variant {key!r} for task type {self.key!r}")
