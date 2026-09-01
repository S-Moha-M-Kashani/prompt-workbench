"""Model prices and capabilities, taken from the provider rather than by hand.

A hand-maintained table of "which model honours temperature" is wrong the week
after it is written, and a hand-ordered "cheapest first" is an opinion. The
provider publishes both facts — per-token prices and the parameters each model
actually accepts — on an endpoint that needs no credential, so this reads them.

Two properties make that safe to depend on. The response is cached to a
committed snapshot, so a first run with no network still has a catalogue; and
the fetch is injected, so no test ever reaches out. A registry running on the
snapshot says so, because a stale price presented as current is worse than an
absent one.
"""

from __future__ import annotations

import json
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from prompt_workbench.models.model_settings import ModelSettings
from prompt_workbench.models.usage import TokenUsage

MODELS_URL = "https://openrouter.ai/api/v1/models"
SNAPSHOT_PATH = Path(__file__).resolve().parent.parent / "data" / "model_snapshot.json"

FETCH_TIMEOUT_SECONDS = 10

# The knobs this project exposes, in the order they are reported.
SETTING_NAMES: tuple[str, ...] = (
    "temperature",
    "top_p",
    "frequency_penalty",
    "presence_penalty",
    "max_tokens",
)

# Assumed when ordering by price before anything has been measured: prompts here
# carry mocked context and answers are short, so input dominates. Replaced by
# real measurements as soon as a run has happened.
ASSUMED_TOKENS_IN = 1200
ASSUMED_TOKENS_OUT = 250

DEFAULT_CONTEXT_WINDOW = 32_000

# The curated shortlist, read once from the snapshot.
_CURATED_IDS: tuple[str, ...] | None = None


@dataclass(frozen=True)
class ModelEntry:
    """One model: what it costs, what it accepts, how much it can hold."""

    id: str
    name: str = ""
    price_in_per_million: float = 0.0
    price_out_per_million: float = 0.0
    context_window: int = DEFAULT_CONTEXT_WINDOW
    parameters: frozenset[str] = frozenset()

    @property
    def has_price(self) -> bool:
        """Whether the provider published a price at all.

        A zero price is treated as unpublished rather than free: presenting an
        unknown as $0.00 is how a sweep's estimate silently becomes fiction.
        """
        return self.price_in_per_million > 0 or self.price_out_per_million > 0

    def blended_price(self) -> float:
        """A single number for ordering, on assumed traffic shape."""
        return (
            ASSUMED_TOKENS_IN * self.price_in_per_million
            + ASSUMED_TOKENS_OUT * self.price_out_per_million
        ) / 1_000_000

    def cost_per_thousand(self, usage: TokenUsage) -> float | None:
        """Dollars per 1,000 calls at this measured usage, or ``None``.

        ``None`` where either the usage or the price is unknown — the two ways
        this number can be a guess dressed as a measurement.
        """
        if not usage.is_reported or not self.has_price:
            return None
        per_call = (
            usage.tokens_in * self.price_in_per_million
            + usage.tokens_out * self.price_out_per_million
        ) / 1_000_000
        return per_call * 1_000

    def supports(self, setting_name: str) -> bool:
        return setting_name in self.parameters


def fetch_live() -> dict[str, Any]:
    """The provider's public model list. No credential; it is a catalogue."""
    with urllib.request.urlopen(MODELS_URL, timeout=FETCH_TIMEOUT_SECONDS) as response:
        payload: dict[str, Any] = json.loads(response.read())
    return payload


def _price(pricing: Any, key: str) -> float:
    """Dollars per million tokens, from the provider's per-token string."""
    try:
        return float(pricing.get(key, 0) or 0) * 1_000_000
    except (AttributeError, TypeError, ValueError):
        return 0.0


def _as_entry(raw: dict[str, Any]) -> ModelEntry | None:
    """One catalogue entry, or ``None`` if it is not usable.

    Skipped rather than raised: a single malformed row among four hundred must
    not cost the user their whole catalogue.
    """
    model_id = raw.get("id")
    if not isinstance(model_id, str) or not model_id:
        return None
    pricing = raw.get("pricing") or {}
    parameters = raw.get("supported_parameters") or []
    return ModelEntry(
        id=model_id,
        name=str(raw.get("name", model_id)),
        price_in_per_million=_price(pricing, "prompt"),
        price_out_per_million=_price(pricing, "completion"),
        context_window=int(raw.get("context_length") or DEFAULT_CONTEXT_WINDOW),
        parameters=frozenset(str(p) for p in parameters),
    )


class ModelRegistry:
    """The model catalogue for one session, live if it can be, snapshot if not."""

    def __init__(self, *, fetch: Callable[[], dict[str, Any]] | None = None) -> None:
        # Resolved at call time rather than bound as a default, so a test can
        # replace the module-level fetcher and actually be obeyed.
        self._fetch = fetch
        self._entries: dict[str, ModelEntry] | None = None
        self.is_stale = False

    def _load(self) -> dict[str, ModelEntry]:
        if self._entries is not None:
            return self._entries
        fetch = self._fetch if self._fetch is not None else fetch_live
        try:
            payload = fetch()
            self.is_stale = False
        except Exception:  # noqa: BLE001 - any failure to reach the list is one failure
            payload = json.loads(SNAPSHOT_PATH.read_text())
            self.is_stale = True
        entries = [_as_entry(raw) for raw in payload.get("data", [])]
        self._entries = {entry.id: entry for entry in entries if entry is not None}
        return self._entries

    def all_models(self) -> tuple[ModelEntry, ...]:
        """Every model the provider lists — several hundred when live."""
        return tuple(self._load().values())

    def curated_ids(self) -> tuple[str, ...]:
        """The shortlist the workbench offers, from the committed snapshot.

        The live list is four hundred models deep and most of them have no place
        in a comparison — the cheapest of them is cheap because it is tiny, and
        offering it as a default for *writing* prompts would be actively bad
        advice. So the snapshot is not just an offline fallback: it is the
        curation, and the live fetch exists to keep its prices honest.
        """
        global _CURATED_IDS
        if _CURATED_IDS is None:
            snapshot = json.loads(SNAPSHOT_PATH.read_text())
            _CURATED_IDS = tuple(entry["id"] for entry in snapshot.get("data", []))
        return _CURATED_IDS

    def recommended(self) -> tuple[ModelEntry, ...]:
        """The curated shortlist, cheapest first, priced from live data if any."""
        known = self._load()
        return self.cheapest_first(
            tuple(mid for mid in self.curated_ids() if mid in known)
        )

    def get(self, model_id: str) -> ModelEntry:
        try:
            return self._load()[model_id]
        except KeyError:
            raise KeyError(f"No model {model_id!r} in the catalogue") from None

    def find(self, model_id: str) -> ModelEntry | None:
        return self._load().get(model_id)

    def cheapest_first(self, model_ids: tuple[str, ...] | None = None) -> tuple[ModelEntry, ...]:
        """Priced models, cheapest first; unpriced ones last."""
        entries = (
            [self._load()[m] for m in model_ids if m in self._load()]
            if model_ids is not None
            else list(self._load().values())
        )
        return tuple(
            sorted(entries, key=lambda e: (not e.has_price, e.blended_price(), e.id))
        )

    def supports(self, model_id: str, setting_name: str) -> bool:
        """Whether ``model_id`` honours ``setting_name``; unknown models are permissive."""
        entry = self.find(model_id)
        return True if entry is None else entry.supports(setting_name)

    def ignored_settings(self, model_id: str, settings: ModelSettings) -> tuple[str, ...]:
        """Names of knobs set on ``settings`` that ``model_id`` would drop."""
        return tuple(
            name
            for name in SETTING_NAMES
            if getattr(settings, name) is not None and not self.supports(model_id, name)
        )

    def context_window(self, model_id: str) -> int:
        entry = self.find(model_id)
        return DEFAULT_CONTEXT_WINDOW if entry is None else entry.context_window
