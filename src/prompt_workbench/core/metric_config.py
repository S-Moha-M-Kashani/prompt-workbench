"""Checking a metric configuration before it can cost anything.

Every problem here is one a user can fix in the Evaluate area, and every one of
them is caught before the first judge call rather than after — an invalid
configuration discovered halfway through a run has already spent money on scores
that will be thrown away.

``validate`` returns problems rather than raising, because the UI wants to show
all of them at once instead of revealing them one call at a time.
"""

from collections.abc import Sequence

from prompt_workbench.models.metrics import MetricDefinition


def selected(metrics: Sequence[MetricDefinition]) -> tuple[MetricDefinition, ...]:
    """The metrics a run would actually use: the enabled ones."""
    return tuple(metric for metric in metrics if metric.enabled)


def validate(metrics: Sequence[MetricDefinition]) -> tuple[str, ...]:
    """Everything wrong with this configuration, in words a user can act on."""
    problems: list[str] = []
    chosen = selected(metrics)

    if not chosen:
        problems.append("No metric is enabled, so there is nothing to score against.")

    names = [metric.name.strip().casefold() for metric in metrics]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        problems.append(
            "Metric names must be unique so a result can be read without ambiguity. "
            "Repeated: " + ", ".join(duplicates)
        )

    if chosen and sum(metric.weight for metric in chosen) <= 0:
        problems.append(
            "The enabled metrics have no positive total weight, so a weighted "
            "grade cannot be calculated. Give at least one of them a weight above zero."
        )

    return tuple(problems)
