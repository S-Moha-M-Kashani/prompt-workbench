"""Running deepeval's metrics over our runs.

The bridge, and the only place that knows both vocabularies: our ``EvalCase``
and ``PromptRun`` on one side, deepeval's ``LLMTestCase`` and metric objects on
the other. Keeping the translation in one module is what lets everything else be
tested without the optional dependency installed.

Two rules carried over from the previous metric layer, because they matter more
than which library implements them. A metric that fails is recorded as a failure
and excluded from the score, never softened into a neutral number. And a metric
whose required fields the cases do not supply is refused before the run, not
discovered halfway through it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from prompt_workbench.models.call import ToolInvocation
from prompt_workbench.models.case import CaseBrief, EvalCase
from prompt_workbench.services.deepeval_metrics import MetricChoice, build, require


@dataclass(frozen=True)
class MetricOutcome:
    """One metric's verdict on one response."""

    metric_key: str
    metric_label: str
    score: float | None
    threshold: float
    reason: str = ""
    failed: bool = False
    failure: str = ""
    higher_is_better: bool = True

    @property
    def passed(self) -> bool:
        """Whether this cleared its threshold, in the right direction."""
        if self.score is None:
            return False
        return (
            self.score >= self.threshold
            if self.higher_is_better
            else self.score <= self.threshold
        )


def preflight(
    *, brief: CaseBrief, choices: Sequence[MetricChoice]
) -> tuple[str, ...]:
    """Everything that would stop these metrics scoring these cases.

    Checked offline and reported all at once, because each one is something the
    user fixes by editing rather than by retrying.
    """
    problems: list[str] = []
    enabled = [choice for choice in choices if choice.enabled]

    if not enabled:
        problems.append("No metric is enabled, so there is nothing to measure against.")
    if not brief.cases:
        problems.append("There are no test cases to run.")

    available = brief.supplied_fields()
    for choice in enabled:
        missing_fields = choice.missing_fields(available)
        if missing_fields:
            problems.append(
                f"{choice.spec.label} reads "
                + ", ".join(missing_fields)
                + ", which your cases do not all supply."
            )
        missing_config = choice.missing_config()
        if missing_config:
            problems.append(
                f"{choice.spec.label} still needs " + ", ".join(missing_config) + "."
            )
    return tuple(problems)


def available_fields(
    case: EvalCase, tool_calls: tuple[ToolInvocation, ...] | None
) -> set[str]:
    """Which metric inputs exist for this case *on this round*.

    A function rather than a method on ``EvalCase``, because whether
    ``tools_called`` exists is a fact about the run and not about the case —
    and making the case know about a ``CallResult`` would couple ground truth
    to measurement in the one direction that must not exist.

    An empty trace still counts as supplied: a round that called nothing is
    evidence, and treating it as a missing input would skip the metric exactly
    when it has something to say.
    """
    fields = case.supplied_fields()
    if tool_calls is not None:
        fields.add("tools_called")
    return fields


def as_llm_test_case(
    case: EvalCase,
    response: str,
    *,
    tool_calls: tuple[ToolInvocation, ...] | None = None,
) -> Any:
    """Our case plus a response, as the object deepeval scores.

    The tool trace comes from what the adapter recorded, never inferred from
    the assistant's prose — "I looked up customer 7" is a claim, and scoring a
    claim as evidence is how a tool-calling measurement stops being one.
    """
    require()
    from deepeval.test_case import LLMTestCase, ToolCall

    return LLMTestCase(
        input=case.input,
        actual_output=response,
        expected_output=case.expected_output,
        context=list(case.context) or None,
        retrieval_context=list(case.retrieval_context) or None,
        tools_called=(
            [ToolCall(name=call.name, input_parameters=dict(call.arguments)) for call in tool_calls]
            if tool_calls is not None
            else None
        ),
        expected_tools=(
            [ToolCall(name=name, input_parameters={}) for name in case.expected_tools]
            if case.expected_tools
            else None
        ),
    )


def score_one(
    *,
    case: EvalCase,
    response: str,
    choice: MetricChoice,
    judge: Any,
    tool_calls: tuple[ToolInvocation, ...] | None = None,
) -> MetricOutcome:
    """Score one response against one metric, turning every failure into a value.

    A judge that times out, a metric that cannot parse a reply, a schema that
    does not match — all of them come back as a recorded failure rather than an
    exception, so one bad metric never costs the rest of a sweep.
    """
    label = choice.spec.label
    try:
        metric = build(choice, judge=judge if choice.spec.uses_judge else None)
        metric.measure(as_llm_test_case(case, response, tool_calls=tool_calls))
        score = metric.score
        reason = getattr(metric, "reason", "") or ""
    except Exception as error:  # noqa: BLE001 - one metric, one failure
        return MetricOutcome(
            metric_key=choice.key,
            metric_label=label,
            score=None,
            threshold=choice.threshold,
            failed=True,
            failure=str(error) or error.__class__.__name__,
            higher_is_better=choice.spec.higher_is_better,
        )

    return MetricOutcome(
        metric_key=choice.key,
        metric_label=label,
        score=None if score is None else float(score),
        threshold=choice.threshold,
        reason=str(reason),
        higher_is_better=choice.spec.higher_is_better,
    )


def aggregate(outcomes: Sequence[MetricOutcome]) -> float | None:
    """The mean of the scores that exist, on a pass/fail-normalised scale.

    Metrics disagree about direction — faithfulness wants 0.9 and up,
    hallucination wants 0.2 and down — so averaging raw scores would add numbers
    that mean opposite things. Each score is first turned into its distance on
    the "good" side of its own threshold.
    """
    usable = [o for o in outcomes if o.score is not None]
    if not usable:
        return None
    normalised = [o.score if o.higher_is_better else 1.0 - (o.score or 0.0) for o in usable]
    return sum(n or 0.0 for n in normalised) / len(normalised)


def met_every_threshold(outcomes: Sequence[MetricOutcome]) -> bool:
    """Whether every metric that produced a score cleared its bar.

    A failed metric counts as not met: a configuration whose measurement broke
    has not been shown to work.
    """
    if not outcomes:
        return False
    return all(outcome.passed for outcome in outcomes)
