"""Whether a shape is enforced by the provider or merely asked for in the prompt.

The difference is load-bearing, which is why it is decided in one place rather
than in each adapter. Where the model publishes a structured-output parameter
the shape is a guarantee; where it does not, the shape is an instruction in the
system prompt and a claim to be checked afterwards by a metric.

Presenting the second as the first would turn a measurement into an assumption —
so the request carries the answer as a flag, the interface states which is in
force, and nothing has to infer it twice.
"""

from dataclasses import replace

from prompt_workbench.models.call import CallRequest, OutputStructure

#: The metric that checks an unenforced shape after the fact.
SHAPE_METRIC_KEY = "json_correctness"


class StructureSupport:
    """Anything that can say whether a model enforces a shape.

    Named as a role rather than typed against ``ModelRegistry`` so ``core``
    keeps depending on behaviour rather than on a service class.
    """

    def can_enforce_structure(self, model_id: str) -> bool:  # pragma: no cover - a role
        raise NotImplementedError


def with_structure(
    request: CallRequest,
    structure: OutputStructure | None,
    support: StructureSupport,
) -> CallRequest:
    """``request`` carrying ``structure``, with the honest enforcement flag set."""
    if structure is None:
        return replace(request, output_structure=None, structure_is_enforced=False)
    return replace(
        request,
        output_structure=structure,
        structure_is_enforced=support.can_enforce_structure(request.model_id),
    )


def enforcement_note(is_enforced: bool) -> str:
    """The sentence the interface shows beside the shape."""
    if is_enforced:
        return (
            "The provider will enforce this shape: the model is held to the schema "
            "and a reply that does not match it is refused before it reaches you."
        )
    return (
        "This model publishes no structured-output parameter, so the shape is only "
        "asked for in the system prompt. Nothing enforces it — score the answers "
        f"with the {SHAPE_METRIC_KEY} metric to find out how often it was honoured."
    )
