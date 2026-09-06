"""Anthropic's own models and prices, kept apart from the provider's list.

The workbench's provider serves an OpenAI-compatible endpoint and publishes a
live catalogue with real prices. It does not serve the Anthropic Messages API,
so the Anthropic adapter reaches a different provider with a different
credential — and its models cannot be priced from the workbench's list without
inventing a number that looks measured.

So this is a second catalogue, labelled as one. It is a committed table rather
than a live fetch: Anthropic publishes no unauthenticated price endpoint, and a
price nobody can refresh has to say when it was written down.
"""

from dataclasses import dataclass

#: When these prices were last checked against Anthropic's published rates.
AS_OF = "2026-06-24"

CATALOGUE_NOTE = (
    "Anthropic is a separate provider: it needs its own credential, and its "
    "models and prices are its own. They are never merged into the workbench's "
    "provider catalogue, because a Claude model priced from that list would be "
    "a fabricated number wearing the look of a measured one."
)


@dataclass(frozen=True)
class AnthropicModel:
    """One Claude model: what it costs and how much it can hold."""

    id: str
    label: str
    price_in_per_million: float
    price_out_per_million: float
    context_window: int
    max_output_tokens: int

    @property
    def has_price(self) -> bool:
        return self.price_in_per_million > 0

    def cost_per_thousand(self, tokens_in: int, tokens_out: int) -> float | None:
        if tokens_in <= 0 and tokens_out <= 0:
            return None
        per_call = (
            tokens_in * self.price_in_per_million
            + tokens_out * self.price_out_per_million
        ) / 1_000_000
        return per_call * 1_000


MODELS: tuple[AnthropicModel, ...] = (
    AnthropicModel("claude-haiku-4-5", "Claude Haiku 4.5", 1.00, 5.00, 200_000, 64_000),
    AnthropicModel("claude-sonnet-5", "Claude Sonnet 5", 2.00, 10.00, 1_000_000, 128_000),
    AnthropicModel("claude-opus-5", "Claude Opus 5", 5.00, 25.00, 1_000_000, 128_000),
    AnthropicModel("claude-opus-4-8", "Claude Opus 4.8", 5.00, 25.00, 1_000_000, 128_000),
    AnthropicModel("claude-fable-5-1", "Claude Fable 5.1", 10.00, 50.00, 1_000_000, 128_000),
)

_BY_ID = {model.id: model for model in MODELS}


def all_models() -> tuple[AnthropicModel, ...]:
    """Cheapest first, like the provider catalogue, for the same reason."""
    return tuple(sorted(MODELS, key=lambda m: (m.price_in_per_million, m.id)))


def get(model_id: str) -> AnthropicModel:
    try:
        return _BY_ID[model_id]
    except KeyError:
        raise KeyError(f"No Anthropic model {model_id!r} in this catalogue") from None


def find(model_id: str) -> AnthropicModel | None:
    return _BY_ID.get(model_id)


def staleness_note() -> str:
    """What the interface says beside these prices."""
    return (
        f"These prices are a committed table checked on {AS_OF}, not a live fetch — "
        "Anthropic publishes no unauthenticated price list. Treat them as "
        "possibly out of date and confirm before relying on a cost figure."
    )
