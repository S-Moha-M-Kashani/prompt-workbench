"""A ready use case: one real prompt-engineering situation, with its world mocked.

The workbench's central idea. A prompt is only judgeable inside a situation —
"what should this classifier's prompt say?" has no answer until you know what
the candidate list looks like, how near the distractors are, and what happens
when the list comes back empty. Those surroundings are normally supplied by a
whole application, which is why prompts usually get written blind.

So each use case carries its own world as mocked blocks: a knowledge base, a
candidate list, tool schemas, prior state. With those in place the prompt can be
run and its answer read, which is the only way to tell whether it works.

A use case also carries the failure it is hunting — the ``trap``. That is the
part a technique-comparison misses: every one of these situations has a specific
way of going wrong, and a prompt is good exactly insofar as it closes that door.
"""

import re
from dataclasses import dataclass, field

from prompt_workbench.models.ground_truth import GroundTruthCase

# A `{name}` placeholder. Deliberately narrow: prompts under test are full of
# JSON braces, and anything looser would treat `{"score": 1}` as a placeholder.
_PLACEHOLDER = re.compile(r"\{([a-z_][a-z0-9_]*)\}")


@dataclass(frozen=True)
class MockBlock:
    """One piece of the world, standing in for real infrastructure."""

    name: str
    label: str
    content: str
    note: str = ""

    @property
    def placeholder(self) -> str:
        """What this block is called inside a prompt."""
        return "{" + self.name + "}"


@dataclass(frozen=True)
class UseCase:
    """A situation worth engineering a prompt for, and how to tell if it worked."""

    key: str
    label: str
    family: str
    situation: str
    trap: str
    system_prompt: str
    mocks: tuple[MockBlock, ...]
    example_message: str
    criteria: tuple[str, ...]
    forbidden: tuple[str, ...] = ()
    origin: str = ""
    notes: str = ""
    task_type: str = ""

    def __post_init__(self) -> None:
        if not self.system_prompt.strip():
            raise ValueError(f"Use case {self.key!r} needs a starting system prompt")
        if not any(c.strip() for c in self.criteria):
            raise ValueError(
                f"Use case {self.key!r} needs at least one criterion — without one "
                "there is no way to say whether a response was any good"
            )

    def filled_prompt(self, prompt_text: str) -> str:
        """``prompt_text`` with this use case's mocks substituted in.

        Plain replacement rather than ``str.format``: the prompts under test
        contain JSON shapes, and formatting would raise on every brace that is
        not a placeholder. An unknown placeholder is left visible rather than
        blanked, so a prompt referring to something that does not exist shows up
        in the output instead of silently becoming an empty string.
        """
        for block in self.mocks:
            prompt_text = prompt_text.replace(block.placeholder, block.content)
        return prompt_text

    def unfilled_placeholders(self, prompt_text: str) -> tuple[str, ...]:
        """Placeholder names in ``prompt_text`` that no mock block supplies."""
        supplied = {block.name for block in self.mocks}
        return tuple(
            name for name in dict.fromkeys(_PLACEHOLDER.findall(prompt_text))
            if name not in supplied
        )

    def as_ground_truth_case(self, case_id: str, *, user_message: str) -> GroundTruthCase:
        """This use case's own expectations, as something the evaluator can score.

        The reason no dataset has to be authored: the criteria that make the use
        case worth trying are the same criteria a response is judged against.
        """
        return GroundTruthCase(
            id=case_id,
            test_message=user_message,
            required_criteria=self.criteria,
            forbidden_behaviours=self.forbidden,
            tags=(self.family,),
        )
