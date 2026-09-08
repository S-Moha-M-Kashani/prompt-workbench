"""One measurement path, so a framework difference is the framework's own.

``LlmCall.run()`` validates the request, starts a monotonic clock, hands off to
the subclass's ``_invoke``, stops the clock, sums the usage across every model
call the round made, and builds the ``CallResult``. A subclass implements
``_invoke`` and names itself; it never times, never sums and never decides the
result shape.

Four adapters timing themselves would be four chances for one framework to look
faster because it started its clock later. The thing being compared must not be
measured by the thing being compared.

A failed round returns a result carrying its cause, with ``latency_ms`` of
``None`` and zero usage. Reporting zero milliseconds would be a lie in exactly
the shape of an unknown price shown as $0.00.
"""

import json
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from time import perf_counter
from typing import ClassVar

from prompt_workbench.models.call import CallRequest, CallResult, ToolInvocation
from prompt_workbench.models.usage import TokenUsage


@dataclass(frozen=True)
class InvocationOutcome:
    """What a subclass gives back: the answer, the usage per model call, the trace.

    ``usages`` is one entry per model call rather than a total, because the
    number of model calls a round took is itself a reported number — a tool
    round-trip is two calls, and a framework that takes more of them must not be
    able to hide that behind a single sum.
    """

    answer: str
    usages: tuple[TokenUsage, ...] = ()
    tool_calls: tuple[ToolInvocation, ...] = ()
    structure_was_enforced: bool = False
    extra_model_calls: int = 0


@dataclass
class LlmCall(ABC):
    """A framework that can run one round, measured identically to the others."""

    #: The registry key this adapter answers to; overridden by every subclass.
    #: A ``ClassVar`` rather than a field, so a subclass declares it once and no
    #: caller can construct an adapter that lies about which framework it is.
    framework: ClassVar[str] = "unnamed"

    clock: Callable[[], float] = field(default=perf_counter)

    def describe(self) -> str:
        return self.framework

    def run(self, request: CallRequest) -> CallResult:
        """One round: validated, timed, summed, and named."""
        request.validate()
        started = self.clock()
        try:
            outcome = self._invoke(request)
        except Exception as error:  # noqa: BLE001 - any framework failure is one failure
            return CallResult(
                answer="",
                framework=self.framework,
                model_id=request.model_id,
                usage=TokenUsage(),
                latency_ms=None,
                model_calls=0,
                error=f"{type(error).__name__}: {error}",
            )
        elapsed_ms = (self.clock() - started) * 1000.0
        return CallResult(
            answer=outcome.answer,
            framework=self.framework,
            model_id=request.model_id,
            usage=_summed(outcome.usages),
            latency_ms=elapsed_ms,
            tool_calls=outcome.tool_calls,
            model_calls=len(outcome.usages) + outcome.extra_model_calls,
            structure_was_enforced=outcome.structure_was_enforced,
        )

    @abstractmethod
    def _invoke(self, request: CallRequest) -> InvocationOutcome:
        """Run the round in this framework's own idiom. No timing, no summing."""


def _summed(usages: tuple[TokenUsage, ...]) -> TokenUsage:
    """Tokens across every model call the round made.

    Never estimated. Where a framework reports nothing this stays zero, and
    ``TokenUsage.is_reported`` already means "unknown" rather than "free".
    """
    return TokenUsage(
        tokens_in=sum(u.tokens_in for u in usages),
        tokens_out=sum(u.tokens_out for u in usages),
    )


def system_prompt_for(request: CallRequest) -> str:
    """The system prompt, with the answer shape appended when unenforced.

    Shared by every adapter rather than reimplemented per framework: an
    unenforced shape has to reach the model *somehow*, and two adapters
    wording that instruction differently would make a shape difference look
    like a framework difference.
    """
    if request.output_structure is None or request.structure_is_enforced:
        return request.system_prompt
    schema = json.dumps(dict(request.output_structure.schema), indent=2)
    return (
        f"{request.system_prompt}\n\n"
        f"Answer with JSON matching this schema and nothing else:\n{schema}"
    )
