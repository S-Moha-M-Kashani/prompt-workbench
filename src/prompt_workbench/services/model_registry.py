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
from dataclasses import dataclass, replace
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

# The parameters a provider publishes when it can hold a model to a shape, and
# when it will accept a tool surface. Read from the model's own published list
# rather than a hand-written capability table, for the same reason the prices are.
STRUCTURE_PARAMETERS: tuple[str, ...] = ("response_format", "structured_outputs")
TOOL_PARAMETERS: tuple[str, ...] = ("tools", "tool_choice")

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

    def selectable(self) -> tuple[ModelEntry, ...]:
        """The whole catalogue, with the curated shortlist pinned on top.

        The full list is several hundred models deep, and most of them have no
        place in a comparison — but excluding them is the workbench deciding for
        the user. So everything is reachable and the curation is an ordering,
        not a filter.
        """
        known = self._load()
        curated = tuple(
            known[mid] for mid in self.curated_ids() if mid in known
        )
        curated_ids = {entry.id for entry in curated}
        rest = self.cheapest_first(
            tuple(mid for mid in known if mid not in curated_ids)
        )
        return self.cheapest_first(tuple(e.id for e in curated)) + rest

    def search(self, query: str) -> tuple[ModelEntry, ...]:
        """Catalogue entries whose identifier or display name contains ``query``.

        Substring rather than fuzzy: someone typing "gpt-5" wants the gpt-5
        models, and a ranked guess that also returns gpt-4o is a worse answer
        than a short exact one. An empty query is the whole catalogue.
        """
        needle = query.strip().lower()
        listed = self.selectable()
        if not needle:
            return listed
        return tuple(
            entry
            for entry in listed
            if needle in entry.id.lower() or needle in entry.name.lower()
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
        """Names of parameters set on ``settings`` that ``model_id`` would drop.

        Covers the five named knobs and everything in the open bag, because a
        dropped ``seed`` is exactly as silent as a dropped ``temperature``.
        """
        return tuple(
            name
            for name in settings.parameter_names
            if not self.supports(model_id, name)
        )

    def can_enforce_structure(self, model_id: str) -> bool:
        """Whether this model publishes a parameter that holds it to a schema.

        A model that does not is not refused a shape — the shape moves into the
        system prompt and the interface says it is asked for rather than
        enforced. Silently dropping it, or claiming it was enforced, are the two
        ways this number stops being a measurement.
        """
        entry = self.find(model_id)
        if entry is None:
            return False
        return any(name in entry.parameters for name in STRUCTURE_PARAMETERS)

    def supports_tools(self, model_id: str) -> bool:
        """Whether this model publishes a tool-use parameter."""
        entry = self.find(model_id)
        if entry is None:
            return False
        return any(name in entry.parameters for name in TOOL_PARAMETERS)

    def published_parameters(self, model_id: str) -> tuple[str, ...]:
        """Every parameter the provider says this model accepts, sorted.

        Empty for a model the catalogue does not know — which ``supports()``
        already treats permissively, so nothing is offered rather than
        everything being claimed.
        """
        entry = self.find(model_id)
        return () if entry is None else tuple(sorted(entry.parameters))

    def addable_parameters(self, model_id: str, settings: ModelSettings) -> tuple[str, ...]:
        """Published parameters not already shown as a control or set in the bag."""
        shown = set(SETTING_NAMES) | set(settings.extra)
        return tuple(
            name for name in self.published_parameters(model_id) if name not in shown
        )

    def with_parameter(
        self, model_id: str, settings: ModelSettings, name: str, value: Any
    ) -> ModelSettings:
        """``settings`` plus one more published parameter, or a refusal.

        Refusing rather than dropping quietly: the user asked for this key by
        name, and a parameter that vanishes between the control and the request
        is how a measurement ends up describing a call nobody made.
        """
        if not self.supports(model_id, name):
            raise ValueError(f"{model_id} does not publish the parameter {name!r}.")
        return replace(settings, extra={**settings.extra, name: value})

    def without_parameter(self, settings: ModelSettings, name: str) -> ModelSettings:
        """``settings`` with one bag entry removed."""
        return replace(
            settings, extra={k: v for k, v in settings.extra.items() if k != name}
        )

    def strip_unsupported(
        self, model_id: str, settings: ModelSettings
    ) -> tuple[ModelSettings, tuple[str, ...]]:
        """``settings`` narrowed to what ``model_id`` publishes, and what was dropped."""
        dropped = self.ignored_settings(model_id, settings)
        if not dropped:
            return settings, ()
        named = {
            name: (None if name in dropped else getattr(settings, name))
            for name in SETTING_NAMES
        }
        kept_extra = {k: v for k, v in settings.extra.items() if k not in dropped}
        return replace(settings, extra=kept_extra, **named), dropped

    def context_window(self, model_id: str) -> int:
        entry = self.find(model_id)
        return DEFAULT_CONTEXT_WINDOW if entry is None else entry.context_window
