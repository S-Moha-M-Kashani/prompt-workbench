"""Generating one candidate system prompt per technique.

Each technique has its own template, so a candidate is the product of exactly
two things: the brief snapshot and the technique's instruction. That is what
makes the five results a fair comparison rather than five drafts.

The rule worth stating out loud is which candidates see the dataset. Only
few-shot does. Showing the test cases to the direct candidate would make it
quietly few-shot as well, and the difference between the two techniques — the
thing the workbench exists to measure — would stop meaning anything. Few-shot in
turn may see *only* the visible dataset, never invented examples, so its
advantage is one the user can read on screen and edit.
"""

from collections.abc import Callable, Sequence
from datetime import datetime

from prompt_workbench.core.generation import GenerationError
from prompt_workbench.core.prompt_registry import load_system_prompt
from prompt_workbench.models.brief import BriefSnapshot
from prompt_workbench.models.candidates import CandidatePrompt, PromptTechnique
from prompt_workbench.models.ground_truth import GroundTruthDataset
from prompt_workbench.models.identifiers import IdFactory
from prompt_workbench.models.model_settings import ModelSettings
from prompt_workbench.models.protocols import CompletionFn

# The only technique that is shown the dataset.
TECHNIQUES_NEEDING_EXAMPLES = frozenset({PromptTechnique.FEW_SHOT})

# How many cases a few-shot candidate is offered. Enough to demonstrate a
# pattern, few enough that the candidate does not become a copy of the test set.
MAX_DEMONSTRATIONS = 3


def _demonstrations(dataset: GroundTruthDataset) -> str:
    """The visible cases, rendered for a few-shot candidate to learn from."""
    blocks: list[str] = []
    for case in dataset.cases[:MAX_DEMONSTRATIONS]:
        block = [f"Input: {case.test_message}"]
        if case.has_reference:
            block.append(f"Ideal response: {case.reference_answer}")
        block.append(case.as_expectations())
        blocks.append("\n".join(block))
    return "\n\n---\n\n".join(blocks)


def generate(
    *,
    brief: BriefSnapshot,
    dataset: GroundTruthDataset | None,
    techniques: Sequence[PromptTechnique],
    platform_instruction: str,
    complete: CompletionFn,
    model: str,
    settings: ModelSettings,
    new_id: IdFactory,
    clock: Callable[[], datetime],
) -> tuple[CandidatePrompt, ...]:
    """One candidate per technique, in the order the techniques were given."""
    needs_examples = [t for t in techniques if t in TECHNIQUES_NEEDING_EXAMPLES]
    if needs_examples and (dataset is None or not len(dataset)):
        names = ", ".join(t.label for t in needs_examples)
        raise GenerationError(
            f"{names} needs visible examples to learn from. Generate or write at "
            "least one ground-truth test case first, or untick that technique."
        )

    candidates: list[CandidatePrompt] = []
    for technique in techniques:
        system = "\n\n".join(
            [platform_instruction, load_system_prompt(technique.template_name)]
        )
        request = f"--- Brief ---\n{brief.as_context()}"
        if technique in TECHNIQUES_NEEDING_EXAMPLES and dataset is not None:
            request += f"\n\n--- Visible test cases ---\n{_demonstrations(dataset)}"

        text = complete(
            [{"role": "system", "content": system}, {"role": "user", "content": request}],
            model=model,
            settings=settings,
        ).strip()
        if not text:
            raise GenerationError(
                f"The model returned an empty system prompt for the "
                f"{technique.label} candidate."
            )

        candidates.append(
            CandidatePrompt(
                id=new_id("candidate"),
                technique=technique,
                system_prompt=text,
                revision=1,
                source_brief=brief.ref,
                source_dataset=(
                    dataset.ref
                    if dataset is not None and technique in TECHNIQUES_NEEDING_EXAMPLES
                    else None
                ),
                created_at=clock(),
            )
        )
    return tuple(candidates)
