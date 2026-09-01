"""Turning one metric plus one piece of evidence into one score.

Two paths behind one function. ``score_metric`` is what callers use; it picks
the deterministic check or the judge based on the metric's own kind, so the
evaluation loop never branches on it and a custom metric goes through exactly
the same code as a built-in one.

The recurring rule: every way this can go wrong produces a ``MetricScore`` that
says so. Nothing returns a number it did not measure.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any, Protocol

from prompt_workbench.core.prompt_registry import load_system_prompt
from prompt_workbench.models.evaluation import EvaluationEvidence, MetricScore
from prompt_workbench.models.metrics import MetricDefinition, MetricKind
from prompt_workbench.models.protocols import Message


class JudgeFn(Protocol):
    """A judge: messages in, reply text out. Both backends satisfy this."""

    def __call__(self, messages: list[Message]) -> str: ...


# --- the deterministic path -----------------------------------------------

# Recognizing "this brief wants JSON" from free text. The brief is written by a
# person, so this looks for how people actually say it rather than for a schema.
_WANTS_JSON = re.compile(r"\bjson\b", re.I)
_WORD_LIMIT = re.compile(r"(?:at most|no more than|under|max(?:imum)?(?: of)?)\s+(\d+)\s+words", re.I)
_CHARACTER_LIMIT = re.compile(
    r"(?:at most|no more than|under|max(?:imum)?(?: of)?)\s+(\d+)\s+characters", re.I
)


def _json_check(response: str) -> tuple[bool, str]:
    try:
        json.loads(response.strip())
    except (json.JSONDecodeError, ValueError):
        return False, "the brief asks for JSON, but the response does not parse as JSON"
    return True, "parses as JSON"


def format_compliance(evidence: EvaluationEvidence) -> MetricScore:
    """Check the response against whatever the brief's output format states.

    Only mechanically checkable rules count. A brief asking for "a warm, concise
    tone" produces *no* applicable check, and this reports that rather than
    awarding a free 1.0 — a metric that always passes is worse than no metric,
    because it raises the grade while looking like evidence.
    """
    stated = evidence.output_format
    results: list[tuple[bool, str]] = []

    if _WANTS_JSON.search(stated):
        results.append(_json_check(evidence.response))

    if match := _WORD_LIMIT.search(stated):
        limit = int(match.group(1))
        count = len(evidence.response.split())
        results.append(
            (count <= limit, f"{count} words against a stated limit of {limit}")
        )

    if match := _CHARACTER_LIMIT.search(stated):
        limit = int(match.group(1))
        count = len(evidence.response)
        results.append(
            (count <= limit, f"{count} characters against a stated limit of {limit}")
        )

    if not results:
        return MetricScore.not_applicable(
            metric_id="format_compliance",
            metric_name="Format compliance",
            weight=0.0,
            reason=(
                "the brief's output format states nothing that can be checked "
                "mechanically, so this metric is skipped rather than passed"
            ),
        )

    passed = sum(1 for ok, _ in results if ok)
    return MetricScore(
        metric_id="format_compliance",
        metric_name="Format compliance",
        weight=0.0,
        score=passed / len(results),
        reason="; ".join(detail for _, detail in results),
    )


DETERMINISTIC_CHECKS: dict[str, Callable[[EvaluationEvidence], MetricScore]] = {
    "format_compliance": format_compliance,
}


# --- the judge path -------------------------------------------------------

# The judge is asked for JSON, but a model asked for JSON sometimes writes a
# sentence around it. This finds the object rather than failing the whole score
# over a "Here you go:" — while a reply with no object at all still fails.
_JSON_OBJECT = re.compile(r"\{.*\}", re.S)


def _parse_verdict(reply: str) -> tuple[float, str]:
    """``(score, reason)`` from a judge's reply, or ``ValueError`` saying why not."""
    match = _JSON_OBJECT.search(reply)
    if not match:
        raise ValueError(f"could not find a JSON verdict in the reply: {reply[:200]!r}")
    try:
        verdict: dict[str, Any] = json.loads(match.group(0))
    except json.JSONDecodeError as error:
        raise ValueError(f"the judge's verdict is not valid JSON: {error}") from error

    raw = verdict.get("score")
    if not isinstance(raw, (int, float)) or isinstance(raw, bool):
        raise ValueError(f"the judge returned no usable score: {verdict!r}")
    score = float(raw)
    if not 0.0 <= score <= 1.0:
        # Not clamped: a judge answering 4 has misread the scale, and clamping
        # to 1.0 would record its misreading as a perfect result.
        raise ValueError(
            f"the judge scored {score}, which is outside the 0-1 range the rubric asks for"
        )
    return score, str(verdict.get("reason", "")).strip()


def build_judge_messages(metric: MetricDefinition, evidence: EvaluationEvidence) -> list[Message]:
    """The evaluator base instruction, this metric's rubric, and the evidence."""
    return [
        {"role": "system", "content": load_system_prompt("evaluator_base")},
        {
            "role": "user",
            "content": (
                f"Metric: {metric.name}\n"
                f"Rubric:\n{metric.rubric}\n\n"
                f"--- Brief ---\n{evidence.brief_context}\n\n"
                f"--- Candidate system prompt under test ---\n{evidence.candidate_prompt}\n\n"
                f"--- Test case expectations ---\n{evidence.case.as_expectations()}\n\n"
                f"--- User message sent ---\n{evidence.user_message}\n\n"
                f"--- Response to score ---\n{evidence.response}"
            ),
        },
    ]


def score_metric(
    metric: MetricDefinition, evidence: EvaluationEvidence, *, judge: JudgeFn
) -> MetricScore:
    """Score one response against one metric, whichever kind it is."""
    if metric.requires_reference and not evidence.case.has_reference:
        return MetricScore.not_applicable(
            metric_id=metric.id,
            metric_name=metric.name,
            weight=metric.weight,
            reason="this test case has no reference answer, so the metric is skipped",
        )

    if metric.kind is MetricKind.DETERMINISTIC:
        check = DETERMINISTIC_CHECKS.get(metric.check)
        if check is None:
            return MetricScore.failed_with(
                metric_id=metric.id,
                metric_name=metric.name,
                weight=metric.weight,
                failure=(
                    f"this metric names a deterministic check, {metric.check!r}, that "
                    "the workbench does not implement"
                ),
            )
        result = check(evidence)
        # The check does not know the metric's weight or id; graft them on.
        return MetricScore(
            metric_id=metric.id,
            metric_name=metric.name,
            weight=metric.weight,
            score=result.score,
            reason=result.reason,
            failed=result.failed,
            failure=result.failure,
            applicable=result.applicable,
        )

    try:
        reply = judge(build_judge_messages(metric, evidence))
    except Exception as error:  # noqa: BLE001 - any backend failure is one failure
        return MetricScore.failed_with(
            metric_id=metric.id,
            metric_name=metric.name,
            weight=metric.weight,
            failure=str(error) or error.__class__.__name__,
        )

    try:
        score, reason = _parse_verdict(reply)
    except ValueError as error:
        return MetricScore.failed_with(
            metric_id=metric.id,
            metric_name=metric.name,
            weight=metric.weight,
            failure=str(error),
        )

    return MetricScore(
        metric_id=metric.id,
        metric_name=metric.name,
        weight=metric.weight,
        score=score,
        reason=reason,
    )
