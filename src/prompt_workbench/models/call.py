"""One round of an LLM call, as a request shape and a result shape.

The unit this workbench measures is a *round*: a system prompt, a user prompt,
optionally a set of tools the model may ask for, optionally a shape the answer
must take, and one answer back. An agent that takes five rounds is brought here
one round at a time, because a loop's behaviour lives in state carried between
rounds and this bench deliberately keeps none.

Every framework — the bare provider SDK, a LangChain agent, a LangGraph node —
is reached through these two types and nothing else. That is what makes a
latency or a token difference between two frameworks a property of the
frameworks rather than of how each one was asked or timed.

``latency_ms`` is ``None`` for a round that failed, never zero: a fast round and
a round that never happened must not read the same, for the same reason an
unknown price is not shown as $0.00.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field

from prompt_workbench.models.model_settings import ModelSettings
from prompt_workbench.models.usage import TokenUsage


@dataclass(frozen=True)
class ToolSpec:
    """A tool the model may ask for, as the user defined it.

    Execution is always mocked, so a tool is only its name, its description and
    optionally the argument shape the model should produce. What is measured is
    whether the model asked for the right tool, not what a real tool returned.
    """

    name: str
    description: str
    input_schema: Mapping[str, object] | None = None


@dataclass(frozen=True)
class ToolInvocation:
    """One recorded tool call: which tool, with what, and when in the sequence."""

    name: str
    arguments: Mapping[str, object]
    order: int


@dataclass(frozen=True)
class OutputStructure:
    """A shape asked of the answer.

    Whether the provider *enforces* this shape or the model was merely asked for
    it in the prompt is a property of the selected model, not of the structure —
    see ``CallRequest.structure_is_enforced``.
    """

    name: str
    schema: Mapping[str, object]


@dataclass(frozen=True)
class CallRequest:
    """Everything one round needs, framework-independent.

    Both prompts are mandatory. A round without a system prompt is not a
    configuration anyone ships, so it is refused here rather than at the
    provider, before any call is made.
    """

    system_prompt: str
    user_prompt: str
    model_id: str
    settings: ModelSettings = field(default_factory=ModelSettings)
    tools: tuple[ToolSpec, ...] = ()
    output_structure: OutputStructure | None = None
    structure_is_enforced: bool = False

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Raise if the round could not honestly be run.

        Called from ``__post_init__`` and again by the call layer, so the
        guarantee holds at the boundary that actually spends money.
        """
        if not self.system_prompt.strip():
            raise ValueError("A round needs a system prompt; this one is blank.")
        if not self.user_prompt.strip():
            raise ValueError("A round needs a user prompt; this one is blank.")

    @property
    def sends_tools(self) -> bool:
        return bool(self.tools)

    @property
    def tool_names(self) -> tuple[str, ...]:
        return tuple(tool.name for tool in self.tools)


@dataclass(frozen=True)
class CallResult:
    """What one round produced, measured the same way for every framework."""

    answer: str
    framework: str
    model_id: str
    usage: TokenUsage = field(default_factory=TokenUsage)
    latency_ms: float | None = None
    tool_calls: tuple[ToolInvocation, ...] = ()
    model_calls: int = 1
    structure_was_enforced: bool = False
    error: str | None = None

    @property
    def failed(self) -> bool:
        return self.error is not None

    @property
    def is_measured(self) -> bool:
        """Whether these numbers describe a round that actually completed."""
        return not self.failed and self.latency_ms is not None
