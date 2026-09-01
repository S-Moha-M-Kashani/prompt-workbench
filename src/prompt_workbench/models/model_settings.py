"""Tunable LLM sampling settings for one model call.

A single bag of optional model parameters, kept separate from the artifacts
being tested: an artifact says *what* to ask for, ``ModelSettings`` says *how*
to generate it. Each field maps to the provider parameter of the same name and
is omitted from the request when ``None``, so the model keeps its own default.
Add a new knob by adding a field here.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSettings:
    """Optional sampling parameters; ``None`` means "leave at the model default"."""

    temperature: float | None = None
    top_p: float | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    max_tokens: int | None = None
