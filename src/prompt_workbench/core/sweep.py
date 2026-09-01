"""Running every variant against every model, and reading the result.

The question a prompt workbench exists to answer is not "which prompt scored
highest" — it is "what is the cheapest thing that is good enough", and those
have different answers surprisingly often. A variant that scores 0.95 on a model
costing thirty times more than one scoring 0.88 is usually the wrong choice, and
a ranking by score alone hides that.

So results rank by score with cost breaking ties, and separately report the
cheapest configuration that cleared every threshold. That second number is the
one you act on.

Nothing here calls a model. The runner is passed in, so the matrix logic — which
is where the arithmetic errors would live — is tested without spending anything.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from prompt_workbench.models.model_settings import ModelSettings
from prompt_workbench.models.usage import TokenUsage
from prompt_workbench.services.model_registry import (
    ASSUMED_TOKENS_IN,
    ASSUMED_TOKENS_OUT,
    ModelRegistry,
)

# What a judged metric is assumed to cost when estimating, before anything has
# run. Judges read the prompt, the case and the response, so their input is
# larger than the call being judged; their output is a score and a sentence.
JUDGE_TOKENS_IN = 1500
JUDGE_TOKENS_OUT = 120


@dataclass(frozen=True)
class SweepPlan:
    """What a sweep would do, priced, before it is allowed to do it."""

    variant_keys: tuple[str, ...]
    model_ids: tuple[str, ...]
    case_count: int
    judged_metric_count: int
    model_calls: int
    judge_calls: int
    estimated_cost: float | None
    has_unpriced_model: bool

    @property
    def is_runnable(self) -> bool:
        return bool(self.variant_keys and self.model_ids and self.case_count > 0)

    @property
    def total_calls(self) -> int:
        return self.model_calls + self.judge_calls

    def summary(self) -> str:
        variants = len(self.variant_keys)
        models = len(self.model_ids)
        cost = (
            "cost unknown — some models publish no price"
            if self.estimated_cost is None
            else f"est. ${self.estimated_cost:.2f}"
        )
        return (
            f"{variants} variant(s) × {models} model(s) × {self.case_count} case(s) "
            f"= {self.model_calls} model calls and {self.judge_calls} judge calls · {cost}"
        )


def plan(
    *,
    variant_keys: Sequence[str],
    model_ids: Sequence[str],
    case_count: int,
    judged_metric_count: int,
    registry: ModelRegistry,
    judge_price_per_million: float = 0.6,
) -> SweepPlan:
    """Price a sweep before running it.

    The estimate uses assumed token shapes, so it is approximate — but it is
    approximate in dollars, which is the unit the decision is actually made in.
    """
    cells = len(variant_keys) * len(model_ids)
    model_calls = cells * case_count
    judge_calls = model_calls * judged_metric_count

    unpriced = False
    total = 0.0
    for model_id in model_ids:
        entry = registry.find(model_id)
        if entry is None or not entry.has_price:
            unpriced = True
            continue
        per_call = (
            ASSUMED_TOKENS_IN * entry.price_in_per_million
            + ASSUMED_TOKENS_OUT * entry.price_out_per_million
        ) / 1_000_000
        total += per_call * len(variant_keys) * case_count

    total += (
        judge_calls
        * (JUDGE_TOKENS_IN + JUDGE_TOKENS_OUT)
        * judge_price_per_million
        / 1_000_000
    )

    return SweepPlan(
        variant_keys=tuple(variant_keys),
        model_ids=tuple(model_ids),
        case_count=case_count,
        judged_metric_count=judged_metric_count,
        model_calls=model_calls,
        judge_calls=judge_calls,
        estimated_cost=None if unpriced else total,
        has_unpriced_model=unpriced,
    )


@dataclass(frozen=True)
class SweepCell:
    """One variant on one model: what it scored and what it cost."""

    variant_key: str
    model_id: str
    settings: ModelSettings
    score: float | None
    usage: TokenUsage
    failed: bool = False
    failure: str = ""
    met_every_threshold: bool = False

    def cost_per_thousand(self, registry: ModelRegistry) -> float | None:
        entry = registry.find(self.model_id)
        return None if entry is None else entry.cost_per_thousand(self.usage)


def _sort_key(cell: SweepCell, registry: ModelRegistry) -> tuple:
    """Failures last; then best score; then cheapest.

    Cost sorts as infinity when unknown, so a configuration whose price cannot
    be established never wins a tie by default.
    """
    cost = cell.cost_per_thousand(registry)
    return (
        cell.failed,
        -(cell.score if cell.score is not None else -1.0),
        cost if cost is not None else float("inf"),
        cell.model_id,
    )


def rank(cells: Sequence[SweepCell], *, registry: ModelRegistry) -> tuple[SweepCell, ...]:
    """Best first, with cost breaking ties and failures kept but sorted last."""
    return tuple(sorted(cells, key=lambda cell: _sort_key(cell, registry)))


def cheapest_passing(
    cells: Sequence[SweepCell], *, registry: ModelRegistry
) -> SweepCell | None:
    """The least expensive configuration that cleared every threshold.

    Usually the answer, and usually not the top of the ranking.
    """
    passing = [cell for cell in cells if cell.met_every_threshold and not cell.failed]
    if not passing:
        return None
    return min(
        passing,
        key=lambda cell: (
            cell.cost_per_thousand(registry)
            if cell.cost_per_thousand(registry) is not None
            else float("inf"),
            -(cell.score or 0.0),
        ),
    )


CellRunner = Callable[[str, str], SweepCell]


def run(
    *,
    variant_keys: Sequence[str],
    model_ids: Sequence[str],
    run_cell: CellRunner,
) -> tuple[SweepCell, ...]:
    """Run every combination, keeping going when one of them fails.

    A single failed cell must not cost the user the whole sweep — the other
    combinations have already been paid for by the time it happens.
    """
    cells: list[SweepCell] = []
    for variant_key in variant_keys:
        for model_id in model_ids:
            try:
                cells.append(run_cell(variant_key, model_id))
            except Exception as error:  # noqa: BLE001 - one cell, one failure
                cells.append(
                    SweepCell(
                        variant_key=variant_key,
                        model_id=model_id,
                        settings=ModelSettings(),
                        score=None,
                        usage=TokenUsage(),
                        failed=True,
                        failure=str(error) or error.__class__.__name__,
                    )
                )
    return tuple(cells)
