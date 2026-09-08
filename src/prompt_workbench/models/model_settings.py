"""Tunable LLM sampling settings for one model call.

An artifact says *what* to ask for; ``ModelSettings`` says *how* to generate it.

Five common knobs stay typed fields, because the interface has real controls for
them and the sweep reads ``temperature`` by name. Everything else a model
publishes — ``seed``, ``stop``, ``top_k``, ``reasoning_effort``, whatever the
provider adds next — arrives in ``extra``. The published list is the provider's
and it changes weekly, so a field-per-knob design would need a release every
time it moved.

Two rules keep the bag honest, and both live at the boundary in
``services.model_registry`` because only the registry knows what a model
publishes: a key the model does not publish is never added, and switching models
drops the now-unsupported keys and reports which.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ModelSettings:
    """Optional sampling parameters; ``None`` means "leave at the model default"."""

    temperature: float | None = None
    top_p: float | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    max_tokens: int | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)

    def as_params(self) -> dict[str, Any]:
        """One merge: the named fields that are set, then everything in ``extra``.

        ``extra`` wins a collision. It is the later, more explicit statement —
        and refusing the collision instead would mean the caller has to know
        which five names happen to be fields.
        """
        params: dict[str, Any] = {
            name: value
            for name, value in (
                ("temperature", self.temperature),
                ("top_p", self.top_p),
                ("frequency_penalty", self.frequency_penalty),
                ("presence_penalty", self.presence_penalty),
                ("max_tokens", self.max_tokens),
            )
            if value is not None
        }
        params.update(self.extra)
        return params

    @property
    def parameter_names(self) -> tuple[str, ...]:
        """Every parameter this would send, named."""
        return tuple(self.as_params())
