"""Generating the hybrid ground-truth dataset from a confirmed brief.

Stateless by construction. It receives a ``BriefSnapshot`` and nothing else —
no chat history, no earlier dataset, no workspace handle — so the same snapshot
and the same model always describe the same request. That is what makes a
regenerated dataset comparable with the one before it.

Validation is strict. A case the model returns without a checkable criterion is
reported, not quietly dropped: silently discarding two of six generated cases
would leave the user with a thinner dataset than they asked for and no idea why.
"""

from collections.abc import Callable
from datetime import datetime

from prompt_workbench.core.generation import GenerationError, as_text_tuple, parse_json_object
from prompt_workbench.core.prompt_registry import load_system_prompt
from prompt_workbench.models.brief import BriefSnapshot
from prompt_workbench.models.ground_truth import (
    CaseCategory,
    GroundTruthCase,
    GroundTruthDataset,
)
from prompt_workbench.models.identifiers import IdFactory
from prompt_workbench.models.model_settings import ModelSettings
from prompt_workbench.models.protocols import CompletionFn

DEFAULT_CASE_COUNT = 6


def _as_category(value: object) -> CaseCategory:
    """A category name from the model, defaulting to normal rather than failing.

    The category is a label for filtering, not evidence. An unrecognized one is
    not worth losing five good cases over.
    """
    try:
        return CaseCategory(str(value).strip().lower())
    except ValueError:
        return CaseCategory.NORMAL


def generate(
    *,
    brief: BriefSnapshot,
    count: int,
    platform_instruction: str,
    complete: CompletionFn,
    model: str,
    settings: ModelSettings,
    new_id: IdFactory,
    clock: Callable[[], datetime],
) -> GroundTruthDataset:
    """Ask for ``count`` test cases about ``brief`` and return them as a dataset."""
    system = "\n\n".join([platform_instruction, load_system_prompt("dataset_generation")])
    request = (
        f"Write {count} test cases for this brief.\n\n"
        f"--- Brief ---\n{brief.as_context()}"
    )

    raw = complete(
        [{"role": "system", "content": system}, {"role": "user", "content": request}],
        model=model,
        settings=settings,
    )
    payload = parse_json_object(raw, what="generated dataset")

    entries = payload.get("cases")
    if not isinstance(entries, list) or not entries:
        raise GenerationError(
            "The model returned no test cases. Try again, or add a case by hand in "
            "the Ground Truth area."
        )

    cases: list[GroundTruthCase] = []
    for position, entry in enumerate(entries, start=1):
        if not isinstance(entry, dict):
            raise GenerationError(f"Generated case {position} was not an object")
        reference = entry.get("reference_answer")
        try:
            cases.append(
                GroundTruthCase(
                    id=new_id("case"),
                    test_message=str(entry.get("test_message", "")),
                    required_criteria=as_text_tuple(entry.get("required_criteria")),
                    forbidden_behaviours=as_text_tuple(entry.get("forbidden_behaviours")),
                    tags=as_text_tuple(entry.get("tags")),
                    reference_answer=str(reference).strip() if reference else None,
                    category=_as_category(entry.get("category")),
                )
            )
        except ValueError as error:
            raise GenerationError(f"Generated case {position} is unusable: {error}") from error

    return GroundTruthDataset(
        id=new_id("dataset"),
        revision=1,
        source_brief=brief.ref,
        cases=tuple(cases),
        created_at=clock(),
    )
