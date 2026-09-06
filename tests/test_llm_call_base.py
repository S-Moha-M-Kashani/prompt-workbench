"""The template method: one clock, one usage sum, for every framework."""

import pytest

from prompt_workbench.llm_call.base import InvocationOutcome, LlmCall
from prompt_workbench.models import CallRequest, ModelSettings, TokenUsage
from prompt_workbench.models.call import ToolInvocation


class StubCall(LlmCall):
    """A framework that does nothing but report what it was told to."""

    framework = "stub"

    def __init__(self, outcome: InvocationOutcome | Exception, **kwargs: object) -> None:
        super().__init__(**kwargs)  # type: ignore[arg-type]
        self._outcome = outcome
        self.invoked_with: CallRequest | None = None

    def _invoke(self, request: CallRequest) -> InvocationOutcome:
        self.invoked_with = request
        if isinstance(self._outcome, Exception):
            raise self._outcome
        return self._outcome


def _request(**overrides: object) -> CallRequest:
    fields: dict[str, object] = {
        "system_prompt": "Be brief.",
        "user_prompt": "Hello.",
        "model_id": "some/model",
        "settings": ModelSettings(temperature=0.1),
    }
    fields.update(overrides)
    return CallRequest(**fields)  # type: ignore[arg-type]


def _ticking_clock(*readings: float):
    values = iter(readings)
    return lambda: next(values)


def test_a_completed_round_is_timed_with_the_injected_clock() -> None:
    call = StubCall(
        InvocationOutcome(answer="hi", usages=(TokenUsage(3, 1),)),
        clock=_ticking_clock(10.0, 10.25),
    )
    result = call.run(_request())
    assert result.latency_ms == pytest.approx(250.0)
    assert result.framework == "stub"
    assert result.model_id == "some/model"
    assert result.answer == "hi"


def test_usage_is_summed_across_every_model_call_the_round_made() -> None:
    call = StubCall(
        InvocationOutcome(
            answer="done",
            usages=(TokenUsage(100, 20), TokenUsage(140, 8)),
            tool_calls=(ToolInvocation(name="lookup", arguments={}, order=0),),
        ),
        clock=_ticking_clock(0.0, 0.5),
    )
    result = call.run(_request())
    assert result.usage == TokenUsage(tokens_in=240, tokens_out=28)
    assert result.model_calls == 2
    assert [c.name for c in result.tool_calls] == ["lookup"]


@pytest.mark.parametrize("field", ["system_prompt", "user_prompt"])
def test_run_refuses_a_blank_prompt_before_any_call(field: str) -> None:
    call = StubCall(InvocationOutcome(answer="hi", usages=()))
    request = _request()
    object.__setattr__(request, field, "   ")
    with pytest.raises(ValueError):
        call.run(request)
    assert call.invoked_with is None


def test_a_failed_round_reports_its_cause_and_no_latency() -> None:
    call = StubCall(RuntimeError("the provider refused"), clock=_ticking_clock(0.0, 9.0))
    result = call.run(_request())
    assert result.failed is True
    assert "the provider refused" in (result.error or "")
    assert result.latency_ms is None
    assert result.usage == TokenUsage()
