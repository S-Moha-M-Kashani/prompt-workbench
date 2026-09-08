"""Typed contracts for the injectable completion seams.

A ``Protocol`` types the *shape* of the injected callable, so both the real
provider-client functions and the tests' fakes satisfy it structurally — no
inheritance, no coupling. Core workflows depend on these protocols and never on
a concrete provider SDK.
"""

from typing import Any, Protocol

from prompt_workbench.models.call import CallRequest, CallResult
from prompt_workbench.models.model_settings import ModelSettings
from prompt_workbench.models.usage import TokenUsage

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
    """A completion that also reports what it cost.

    Separate from ``CompletionFn`` rather than replacing it: only the one-shot
    runner needs the numbers, and widening the common protocol would make every
    fake in the suite return a tuple it does not care about.
    """

    def __call__(
        self,
        messages: list[Message],
        *,
        model: str | None = None,
        settings: ModelSettings | None = None,
    ) -> tuple[str, TokenUsage]: ...


class CallRunner(Protocol):
    """A framework that can run one round and report what it cost.

    ``core`` depends on this and never on a framework SDK: the adapters live in
    ``llm_call/``, which is the only ring allowed to import one. ``describe()``
    returns the framework's key, so a result can name what produced it.
    """

    def run(self, request: CallRequest) -> CallResult: ...

    def describe(self) -> str: ...
