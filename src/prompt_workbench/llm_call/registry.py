"""Which frameworks this installation can actually run a round through.

Availability is data, decided at call time rather than at module import: an
absent optional library must disable only itself, name its install command, and
cost nothing to the rest of the workbench. Importing each SDK at the top of this
file to find out would make an unused framework a startup cost everyone pays and
would make this module's import order load-bearing.

Adding a framework is one entry here plus one adapter module. Nothing about the
call shape, the measurement, the sweep or the UI changes.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from prompt_workbench.llm_call.base import LlmCall
from prompt_workbench.llm_call.openai_call import OpenAiCall


@dataclass(frozen=True)
class FrameworkEntry:
    """One way of reaching a model, and what it needs to be usable."""

    key: str
    label: str
    install_hint: str
    #: Import names checked for presence. Empty means "always available".
    import_names: tuple[str, ...]
    builder: Callable[..., LlmCall]
    #: What this adapter carries through from the request, for the picker to say.
    forwards: tuple[str, ...] = ("tools", "output structure", "model parameters")
    note: str = ""

    def is_available(self) -> bool:
        return all(
            importlib.util.find_spec(name) is not None for name in self.import_names
        )

    def unavailable_reason(self) -> str:
        return (
            f"{self.label} is not installed in this environment. "
            f"Install it with: {self.install_hint}"
        )

    def build(self, **kwargs: Any) -> LlmCall:
        """Construct the adapter, or refuse with the install command."""
        if not self.is_available():
            raise ModuleNotFoundError(self.unavailable_reason())
        return self.builder(**kwargs)


def _langchain_builder(**kwargs: Any) -> LlmCall:
    """Imported inside the builder so an absent extra costs nothing at import."""
    from prompt_workbench.llm_call.langchain_call import LangChainCall

    return LangChainCall(**kwargs)


def _langgraph_builder(**kwargs: Any) -> LlmCall:
    from prompt_workbench.llm_call.langgraph_call import LangGraphCall

    return LangGraphCall(**kwargs)


_ENTRIES: tuple[FrameworkEntry, ...] = (
    FrameworkEntry(
        key="openai",
        label="OpenAI SDK (no framework)",
        install_hint="already installed — it is a required dependency",
        import_names=(),
        builder=OpenAiCall,
        note="The baseline: the provider's API called directly, nothing in between.",
    ),
    FrameworkEntry(
        key="langchain",
        label="LangChain agent",
        install_hint="uv sync --extra langchain",
        import_names=("langchain", "langchain_openai"),
        builder=_langchain_builder,
        forwards=("tools", "model parameters"),
        note=(
            "LangChain's own `create_agent`. It runs its own tool loop, so a round "
            "with tools may make more model calls than the bare SDK does. An answer "
            "shape it cannot enforce is asked for in the system prompt."
        ),
    ),
    FrameworkEntry(
        key="langgraph",
        label="LangGraph (model node + tool node)",
        install_hint="uv sync --extra langgraph",
        import_names=("langgraph", "langchain_openai"),
        builder=_langgraph_builder,
        forwards=("tools", "model parameters"),
        note=(
            "A two-node graph built here rather than a prebuilt agent, so what is "
            "measured is the graph runtime and not LangChain's agent again."
        ),
    ),
)


def all_frameworks() -> tuple[FrameworkEntry, ...]:
    """Every framework the workbench knows, installed or not."""
    return _ENTRIES


def get(key: str) -> FrameworkEntry:
    for entry in _ENTRIES:
        if entry.key == key:
            return entry
    raise KeyError(f"No framework {key!r} in the registry")


def available_keys() -> tuple[str, ...]:
    return tuple(entry.key for entry in _ENTRIES if entry.is_available())


def build(key: str, **kwargs: Any) -> LlmCall:
    """The adapter for ``key``, or a refusal naming the install command."""
    return get(key).build(**kwargs)
