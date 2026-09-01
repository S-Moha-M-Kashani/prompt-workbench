"""Model prices and capabilities, from the provider rather than by hand.

No test reaches the network: the fetcher is injected, and the committed
snapshot is what the suite reads.
"""

import json

import pytest

from prompt_workbench.models import ModelSettings, TokenUsage
from prompt_workbench.services import model_registry

SAMPLE = {
    "data": [
        {
            "id": "openai/gpt-4o-mini",
            "name": "GPT-4o mini",
            "context_length": 128000,
            "pricing": {"prompt": "0.00000015", "completion": "0.0000006"},
            "supported_parameters": ["temperature", "top_p", "max_tokens", "response_format"],
        },
        {
            "id": "openai/gpt-5-mini",
            "name": "GPT-5 mini",
            "context_length": 400000,
            "pricing": {"prompt": "0.00000025", "completion": "0.000002"},
            "supported_parameters": ["max_tokens", "reasoning_effort", "response_format"],
        },
        {
            "id": "expensive/model",
            "name": "Expensive",
            "context_length": 8000,
            "pricing": {"prompt": "0.00006", "completion": "0.00012"},
            "supported_parameters": ["temperature", "max_tokens"],
        },
    ]
}


def a_registry(payload: dict | None = None, *, fails: bool = False) -> model_registry.ModelRegistry:
    def fetch() -> dict:
        if fails:
            raise RuntimeError("network unreachable")
        return payload if payload is not None else SAMPLE

    return model_registry.ModelRegistry(fetch=fetch)


# --- prices ---------------------------------------------------------------


def test_prices_are_read_per_million_tokens_not_per_token() -> None:
    """The provider quotes dollars per token, which is unreadable. Everything
    downstream works in dollars per million."""
    entry = a_registry().get("openai/gpt-4o-mini")
    assert entry.price_in_per_million == pytest.approx(0.15)
    assert entry.price_out_per_million == pytest.approx(0.60)


def test_capabilities_come_from_the_provider_not_a_hand_written_table() -> None:
    registry = a_registry()
    assert registry.supports("openai/gpt-4o-mini", "temperature")
    assert not registry.supports("openai/gpt-5-mini", "temperature")
    assert registry.supports("openai/gpt-5-mini", "max_tokens")


def test_context_window_comes_from_the_provider() -> None:
    assert a_registry().get("openai/gpt-5-mini").context_window == 400000


def test_models_can_be_ordered_cheapest_first() -> None:
    """Ordering is by blended cost, since a model can be cheap in and dear out."""
    ordered = [entry.id for entry in a_registry().cheapest_first()]
    assert ordered[0] == "openai/gpt-4o-mini"
    assert ordered[-1] == "expensive/model"


def test_an_unknown_model_is_not_invented() -> None:
    with pytest.raises(KeyError):
        a_registry().get("nobody/no-such-model")


# --- the snapshot fallback ------------------------------------------------


def test_a_failed_fetch_falls_back_to_the_committed_snapshot() -> None:
    registry = a_registry(fails=True)
    assert registry.all_models(), "the snapshot must carry the app when the network is gone"
    assert registry.is_stale, "and must say the prices may be out of date"


def test_a_successful_fetch_is_not_marked_stale() -> None:
    registry = a_registry()
    registry.all_models()
    assert not registry.is_stale


def test_the_committed_snapshot_is_valid_and_covers_the_recommended_models() -> None:
    """Shipped so the app works offline on a first run."""
    snapshot = json.loads(model_registry.SNAPSHOT_PATH.read_text())
    ids = {entry["id"] for entry in snapshot["data"]}
    assert len(ids) >= 8
    for required in ("openai/gpt-4o-mini", "openai/gpt-5-mini"):
        assert required in ids, required


def test_a_malformed_entry_is_skipped_rather_than_breaking_the_catalogue() -> None:
    payload = {"data": [{"id": "good/model", "pricing": {"prompt": "0.000001", "completion": "0.000002"}}, {"no_id": True}]}
    registry = a_registry(payload)
    assert [entry.id for entry in registry.all_models()] == ["good/model"]


# --- cost -----------------------------------------------------------------


def test_cost_per_thousand_calls_comes_from_measured_tokens() -> None:
    entry = a_registry().get("openai/gpt-4o-mini")
    # 1,000 in and 200 out, a thousand times over.
    cost = entry.cost_per_thousand(TokenUsage(tokens_in=1000, tokens_out=200))
    # (1000 * 0.15 + 200 * 0.60) / 1e6 * 1000 == 0.27
    assert cost == pytest.approx(0.27)


def test_cost_is_unknown_rather_than_zero_when_usage_was_not_reported() -> None:
    entry = a_registry().get("openai/gpt-4o-mini")
    assert entry.cost_per_thousand(TokenUsage()) is None


def test_a_model_with_no_published_price_reports_unknown_cost() -> None:
    payload = {"data": [{"id": "free/model", "pricing": {"prompt": "0", "completion": "0"}}]}
    entry = a_registry(payload).get("free/model")
    assert entry.has_price is False
    assert entry.cost_per_thousand(TokenUsage(tokens_in=10, tokens_out=10)) is None


# --- settings filtering ---------------------------------------------------


def test_settings_a_model_ignores_are_named() -> None:
    ignored = a_registry().ignored_settings(
        "openai/gpt-5-mini", ModelSettings(temperature=0.7, top_p=0.9, max_tokens=64)
    )
    assert set(ignored) == {"temperature", "top_p"}


def test_an_unknown_model_is_treated_permissively() -> None:
    """The provider is the real enforcer; a model we have no facts about must
    not be blocked locally."""
    assert a_registry().supports("nobody/no-such-model", "temperature")


def test_the_shortlist_is_the_snapshot_not_the_whole_live_list() -> None:
    """The live list is hundreds deep and its cheapest entry is cheap because it
    is tiny — a bad default for writing prompts. The snapshot is the curation."""
    registry = a_registry()
    assert len(registry.all_models()) == 3
    curated = set(registry.curated_ids())
    assert len(curated) >= 8
    assert "openai/gpt-4o-mini" in curated


def test_the_shortlist_keeps_live_prices_where_they_are_known() -> None:
    """Curated by the snapshot, priced by the live fetch."""
    payload = {
        "data": [
            {"id": "openai/gpt-4o-mini", "pricing": {"prompt": "0.000009", "completion": "0.00001"}},
            {"id": "some/other-model", "pricing": {"prompt": "0.0000001", "completion": "0.0000001"}},
        ]
    }
    shortlist = a_registry(payload).recommended()
    ids = [entry.id for entry in shortlist]
    assert ids == ["openai/gpt-4o-mini"], "an uncurated model must not be offered"
    assert shortlist[0].price_in_per_million == pytest.approx(9.0), "priced from the live fetch"


def test_the_shortlist_is_ordered_cheapest_first() -> None:
    registry = a_registry(fails=True)
    prices = [entry.blended_price() for entry in registry.recommended()]
    assert prices == sorted(prices)
