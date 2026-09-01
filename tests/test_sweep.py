"""The sweep: what it would cost, then what it found."""

import pytest

from prompt_workbench.core import sweep
from prompt_workbench.models import ModelSettings, TokenUsage
from prompt_workbench.services import model_registry

SAMPLE = {
    "data": [
        {"id": "cheap/model", "pricing": {"prompt": "0.0000001", "completion": "0.0000004"},
         "context_length": 128000, "supported_parameters": ["temperature", "max_tokens"]},
        {"id": "dear/model", "pricing": {"prompt": "0.000003", "completion": "0.000015"},
         "context_length": 200000, "supported_parameters": ["temperature", "max_tokens"]},
    ]
}


def a_registry() -> model_registry.ModelRegistry:
    return model_registry.ModelRegistry(fetch=lambda: SAMPLE)


# --- planning -------------------------------------------------------------


def test_a_plan_counts_every_combination_before_anything_runs() -> None:
    plan = sweep.plan(
        variant_keys=("a", "b", "c"),
        model_ids=("cheap/model", "dear/model"),
        case_count=8,
        judged_metric_count=3,
        registry=a_registry(),
    )
    assert plan.model_calls == 3 * 2 * 8
    assert plan.judge_calls == 3 * 2 * 8 * 3


def test_a_plan_estimates_cost_from_real_prices() -> None:
    plan = sweep.plan(
        variant_keys=("a",),
        model_ids=("cheap/model",),
        case_count=10,
        judged_metric_count=0,
        registry=a_registry(),
    )
    assert plan.estimated_cost is not None
    assert plan.estimated_cost > 0


def test_the_dearer_model_makes_the_plan_dearer() -> None:
    def cost_of(model_id: str) -> float:
        estimate = sweep.plan(
            variant_keys=("a",), model_ids=(model_id,), case_count=10,
            judged_metric_count=0, registry=a_registry(),
        ).estimated_cost
        assert estimate is not None
        return estimate

    assert cost_of("dear/model") > cost_of("cheap/model")


def test_an_unpriced_model_makes_the_estimate_unknown_not_zero() -> None:
    registry = model_registry.ModelRegistry(
        fetch=lambda: {"data": [{"id": "free/model", "pricing": {"prompt": "0", "completion": "0"}}]}
    )
    plan = sweep.plan(
        variant_keys=("a",), model_ids=("free/model",), case_count=4,
        judged_metric_count=1, registry=registry,
    )
    assert plan.estimated_cost is None
    assert plan.has_unpriced_model


def test_an_empty_plan_is_refused() -> None:
    for variants, models, cases in ((), ("m",), 3), (("a",), (), 3), (("a",), ("m",), 0):
        plan = sweep.plan(
            variant_keys=variants, model_ids=models, case_count=cases,
            judged_metric_count=1, registry=a_registry(),
        )
        assert not plan.is_runnable


# --- ranking --------------------------------------------------------------


def cell(variant: str, model: str, score: float | None, usage: TokenUsage, *,
         failed: bool = False, passed_all: bool = True) -> sweep.SweepCell:
    return sweep.SweepCell(
        variant_key=variant, model_id=model, settings=ModelSettings(),
        score=score, usage=usage, failed=failed, failure="boom" if failed else "",
        met_every_threshold=passed_all,
    )


def test_results_rank_by_score_first() -> None:
    ranked = sweep.rank(
        (
            cell("a", "cheap/model", 0.7, TokenUsage(1000, 100)),
            cell("b", "cheap/model", 0.9, TokenUsage(1000, 100)),
        ),
        registry=a_registry(),
    )
    assert [c.variant_key for c in ranked] == ["b", "a"]


def test_cost_breaks_a_tie_so_the_cheapest_equal_option_wins() -> None:
    ranked = sweep.rank(
        (
            cell("a", "dear/model", 0.9, TokenUsage(1000, 100)),
            cell("a", "cheap/model", 0.9, TokenUsage(1000, 100)),
        ),
        registry=a_registry(),
    )
    assert ranked[0].model_id == "cheap/model"


def test_the_cheapest_configuration_clearing_every_threshold_is_identified() -> None:
    """The actual question: not which scored highest, but which is good enough
    and costs least."""
    winner = sweep.cheapest_passing(
        (
            cell("a", "dear/model", 0.95, TokenUsage(1000, 100)),
            cell("b", "cheap/model", 0.85, TokenUsage(1000, 100)),
            cell("c", "cheap/model", 0.60, TokenUsage(1000, 100), passed_all=False),
        ),
        registry=a_registry(),
    )
    assert winner is not None
    assert (winner.variant_key, winner.model_id) == ("b", "cheap/model")


def test_no_winner_when_nothing_clears_the_thresholds() -> None:
    assert sweep.cheapest_passing(
        (cell("a", "cheap/model", 0.4, TokenUsage(1000, 100), passed_all=False),),
        registry=a_registry(),
    ) is None


def test_a_failed_combination_is_kept_visible_but_never_ranked_or_chosen() -> None:
    cells = (
        cell("a", "cheap/model", None, TokenUsage(), failed=True, passed_all=False),
        cell("b", "cheap/model", 0.8, TokenUsage(1000, 100)),
    )
    ranked = sweep.rank(cells, registry=a_registry())
    assert ranked[-1].failed, "failures sort last rather than disappearing"
    assert sweep.cheapest_passing(cells, registry=a_registry()).variant_key == "b"  # type: ignore[union-attr]


def test_a_cell_reports_its_measured_cost_per_thousand_calls() -> None:
    ranked = sweep.rank(
        (cell("a", "cheap/model", 0.8, TokenUsage(tokens_in=1000, tokens_out=200)),),
        registry=a_registry(),
    )
    # (1000 * 0.10 + 200 * 0.40) / 1e6 * 1000 == 0.18
    assert ranked[0].cost_per_thousand(a_registry()) == pytest.approx(0.18)


def test_cost_is_unknown_when_the_provider_reported_no_usage() -> None:
    ranked = sweep.rank((cell("a", "cheap/model", 0.8, TokenUsage()),), registry=a_registry())
    assert ranked[0].cost_per_thousand(a_registry()) is None
