"""Static capability facts about a single selectable model.

An aggregating gateway accepts every sampling parameter for every model, but
silently drops the ones the upstream provider does not implement — a request to
a reasoning model with ``temperature=2.0`` succeeds and changes nothing.
``ModelInfo`` records what a model *really* honours, taken from the
``supported_parameters`` field of the provider's model list, so callers can warn
instead of measuring noise.

``tunable`` holds knob names (see ``services.model_catalog.SETTING_NAMES``)
rather than one boolean per knob: a set stays correct as knobs are added.

``context_window`` is what the context-limit warning measures a conversation
against. It is the model's total window, so the estimate must leave room for the
reply as well as the history.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ModelInfo:
    """One selectable model and the sampling knobs it actually applies."""

    id: str
    note: str = ""
    tunable: frozenset[str] = field(default_factory=frozenset)
    context_window: int = 128_000
