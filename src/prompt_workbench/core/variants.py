"""Writing one prompt per approach, for the job the user actually has.

The approaches come from the task type, so a classifier gets strict enumeration
and few-shot over its real labels while a drafting job gets outline-then-write
and named anti-patterns. That is the whole difference from the earlier design,
which offered the same five "techniques" for every prompt and could therefore
only ever compare wordings.

Each variant is generated independently from the same brief, so the only thing
that differs between them is the approach — which is what makes putting their
scores side by side mean anything.
"""

from collections.abc import Callable, Sequence
from datetime import datetime

from prompt_workbench.core.generation import GenerationError
from prompt_workbench.core.prompt_registry import load_system_prompt
from prompt_workbench.models.case import CaseBrief
from prompt_workbench.models.identifiers import IdFactory
from prompt_workbench.models.model_settings import ModelSettings
from prompt_workbench.models.protocols import CompletionFn
from prompt_workbench.models.task_type import TaskType, VariantApproach
from prompt_workbench.models.variant import PromptVariant

# How many of the user's cases a variant writer is shown. Enough to ground the
# vocabulary and the shape; few enough that the prompt does not become a copy of
# the test set, which would score well and generalise to nothing.
MAX_CASES_SHOWN = 3


def _brief(case: CaseBrief, task: TaskType, approach: VariantApproach) -> str:
    blocks = [
        f"WHAT THE PROMPT MUST DO\n{case.description}",
        f"KIND OF JOB\n{task.label}: {task.description}",
        f"APPROACH TO USE — {approach.label}\n{approach.instruction}",
    ]
    if case.output_format.strip():
        blocks.append(f"REQUIRED OUTPUT FORMAT\n{case.output_format.strip()}")
    if case.cases:
        shown = "\n\n".join(
            f"Input: {c.input}"
            + (f"\nExpected: {c.expected_output}" if c.expected_output else "")
            for c in case.cases[:MAX_CASES_SHOWN]
        )
        blocks.append(f"THE USER'S OWN TEST CASES\n{shown}")
    if case.notes.strip():
        blocks.append(f"NOTES\n{case.notes.strip()}")
    return "\n\n".join(blocks)


def generate(
    *,
    case: CaseBrief,
    task: TaskType,
    approaches: Sequence[VariantApproach],
    complete: CompletionFn,
    model: str,
    settings: ModelSettings | None,
    new_id: IdFactory,
    clock: Callable[[], datetime],
) -> tuple[PromptVariant, ...]:
    """One complete prompt per approach, in the order given."""
    if not approaches:
        raise GenerationError("Pick at least one approach to write a prompt for.")
    if not case.description.strip():
        raise GenerationError("Describe what the prompt must do first.")

    variants: list[PromptVariant] = []
    for approach in approaches:
        text = complete(
            [
                {"role": "system", "content": load_system_prompt("variant_writer")},
                {"role": "user", "content": _brief(case, task, approach)},
            ],
            model=model,
            settings=settings,
        ).strip()
        if not text:
            raise GenerationError(
                f"The model returned an empty prompt for the {approach.label} approach."
            )
        variants.append(
            PromptVariant(
                id=new_id("variant"),
                approach_key=approach.key,
                approach_label=approach.label,
                task_type_key=task.key,
                system_prompt=text,
                revision=1,
                created_at=clock(),
            )
        )
    return tuple(variants)
