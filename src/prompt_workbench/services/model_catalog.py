"""Curated catalog of selectable models and the settings each one honours.

Single source of truth for "can this model do X?". The UI and the orchestration
layer ask this module instead of hardcoding model ids, so the answer is stated
once. Facts mirror the ``supported_parameters`` field of the provider's model
list and are refreshed by hand — the provider remains the enforcer.

The first supported-model list is deliberately provisional; it is reviewed at
the model refinement gate before manual testing is implemented.
"""

from prompt_workbench.models import ModelSettings
from prompt_workbench.models.model_info import ModelInfo


# The sampling knobs this project exposes; mirrors ``ModelSettings`` fields and
# fixes the order used when reporting ignored settings.
SETTING_NAMES: tuple[str, ...] = (
    "temperature",
    "top_p",
    "frequency_penalty",
    "presence_penalty",
    "max_tokens",
)

_EVERY_KNOB = frozenset(SETTING_NAMES)

# Reasoning models apply no sampling knobs at all; only the output cap survives.
_OUTPUT_CAP_ONLY = frozenset({"max_tokens"})

# Hand-curated, and deliberately short. An id that does not exist on the
# provider fails at call time with a 404 that reads like a workbench bug, so the
# list is trimmed to models that can be named with confidence rather than padded
# for choice. The recommended default comes first; the fully tunable options
# follow, cheapest first.
#
# Context windows are the provider's published totals, used only to warn before
# a conversation overflows.
MODELS: tuple[ModelInfo, ...] = (
    ModelInfo(
        "openai/gpt-5-mini",
        "recommended default; a reasoning model, so sampling knobs do nothing",
        _OUTPUT_CAP_ONLY,
        context_window=400_000,
    ),
    ModelInfo(
        "openai/gpt-5-nano",
        "cheapest reasoning model; sampling knobs do nothing",
        _OUTPUT_CAP_ONLY,
        context_window=400_000,
    ),
    ModelInfo(
        "openai/gpt-5",
        "highest capability; sampling knobs do nothing",
        _OUTPUT_CAP_ONLY,
        context_window=400_000,
    ),
    ModelInfo(
        "openai/gpt-4o-mini",
        "tunable and cheap; the one to pick when comparing sampling settings",
        _EVERY_KNOB,
        context_window=128_000,
    ),
    ModelInfo(
        "openai/gpt-4o",
        "tunable; pricier",
        _EVERY_KNOB,
        context_window=128_000,
    ),
    ModelInfo(
        "anthropic/claude-sonnet-4.5",
        "tunable; a different model family, which makes a candidate's "
        "portability visible",
        _EVERY_KNOB,
        context_window=200_000,
    ),
    ModelInfo(
        "anthropic/claude-haiku-4.5",
        "tunable; fast and cheap in the Claude family",
        _EVERY_KNOB,
        context_window=200_000,
    ),
    ModelInfo(
        "deepseek/deepseek-chat",
        "tunable; open-weights, useful as a low-cost third opinion",
        _EVERY_KNOB,
        context_window=64_000,
    ),
)

# What the sidebar starts on when nothing is configured.
DEFAULT_MODEL_ID = MODELS[0].id

# The window assumed for a model the catalog has no facts about. Small on
# purpose: warning too early costs a user one dismissed message, while assuming
# a large window costs them a failed call at the end of a long conversation.
FALLBACK_CONTEXT = 32_000

_BY_ID: dict[str, ModelInfo] = {model.id: model for model in MODELS}


def all_models() -> tuple[ModelInfo, ...]:
    """Every selectable model, in display order."""
    return MODELS


def fully_tunable_models() -> tuple[ModelInfo, ...]:
    """Models honouring every knob in ``SETTING_NAMES``."""
    return tuple(model for model in MODELS if _EVERY_KNOB <= model.tunable)


def get(model_id: str) -> ModelInfo:
    """Look up one model; raises ``KeyError`` if it is not in the catalog."""
    return _BY_ID[model_id]


def context_window(model_id: str) -> int:
    """The model's total context window, or a conservative floor if unknown."""
    info = _BY_ID.get(model_id)
    return FALLBACK_CONTEXT if info is None else info.context_window


def supports(model_id: str, setting_name: str) -> bool:
    """Whether ``model_id`` honours ``setting_name``; unknown models are permissive."""
    info = _BY_ID.get(model_id)
    return True if info is None else setting_name in info.tunable


def ignored_settings(model_id: str, settings: ModelSettings) -> tuple[str, ...]:
    """Names of knobs set on ``settings`` that ``model_id`` will silently drop.

    A knob left at ``None`` is not being requested, so it is never reported.
    """
    tunable = get(model_id).tunable
    return tuple(
        name
        for name in SETTING_NAMES
        if getattr(settings, name) is not None and name not in tunable
    )
