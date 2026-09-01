"""Typed contracts for the injectable completion seams.

A ``Protocol`` types the *shape* of the injected callable, so both the real
provider-client functions and the tests' fakes satisfy it structurally — no
inheritance, no coupling. Core workflows depend on these protocols and never on
a concrete provider SDK.
"""

from typing import Any, Protocol

from prompt_workbench.models.model_settings import ModelSettings

Message = dict[str, str]


class CompletionFn(Protocol):
    def __call__(
        self,
        messages: list[Message],
        *,
        model: str | None = None,
        settings: ModelSettings | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> str: ...


class CompletionWithUsageFn(Protocol):
    def __call__(
        self,
        messages: list[Message],
        *,
        model: str | None = None,
        settings: ModelSettings | None = None,
    ) -> tuple[str, int]: ...
