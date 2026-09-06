"""What one browser session is holding as it works toward a configuration.

The state is a sequence, and it is worth naming because the interface follows
it: describe the case, settle the task type, get test cases, write variants,
choose metrics, sweep, keep the cheapest thing that passed. Each step is only
meaningful once the one before it exists, so the session exposes that as
``stage`` rather than leaving each widget to work it out.

Changing the task type resets what came from the old one. A variant written for
a classifier and scored by a classifier's metrics means nothing once the job is
declared to be summarization, and keeping it around invites exactly that
mistake.

Sweep results are held to the same rule, and for the same reason. A score is a
statement about one exact prompt, on one model, over one set of cases, judged by
one set of metrics at one set of settings. Change any of those and the number on
screen is describing something that no longer exists — so it is discarded rather
than left to be read as if it still applied.
"""

from collections.abc import Callable, Sequence
from dataclasses import replace
from datetime import UTC, datetime

from prompt_workbench.core import output_structure as shapes
from prompt_workbench.core import presets
from prompt_workbench.core.sweep import SweepCell
from prompt_workbench.models.call import CallRequest, CallResult, OutputStructure, ToolSpec
from prompt_workbench.models.case import CaseBrief, EvalCase
from prompt_workbench.models.identifiers import IdFactory, random_id
from prompt_workbench.models.model_settings import ModelSettings
from prompt_workbench.models.task_type import TaskType
from prompt_workbench.models.variant import PromptVariant
from prompt_workbench.services import deepeval_metrics, task_catalog
from prompt_workbench.services.deepeval_metrics import MetricChoice
from prompt_workbench.services.model_registry import ModelRegistry

STAGES: tuple[str, ...] = ("describe", "cases", "variants", "metrics", "sweep")

# The approach a prompt the user wrote themselves belongs to. It is a real
# approach — "whatever you are doing today" — and the one every generated
# variant has to beat to be worth adopting.
HAND_WRITTEN_KEY = "your_own"
HAND_WRITTEN_LABEL = "Your own prompt"

# The framework a round uses until the user picks another: the provider's API
# called directly, which is the behaviour that existed before this layer.
DEFAULT_FRAMEWORK = "openai"


def _now() -> datetime:
    return datetime.now(UTC)


class Session:
    """One person's work on one case."""

    def __init__(
        self,
        *,
        new_id: IdFactory = random_id,
        clock: Callable[[], datetime] = _now,
        registry: ModelRegistry | None = None,
    ) -> None:
        self._new_id = new_id
        self._clock = clock
        self.registry = registry if registry is not None else ModelRegistry()

        self.description: str = ""
        self.task_type: TaskType | None = None
        self.output_format: str = ""
        self.cases: tuple[EvalCase, ...] = ()
        self.variants: tuple[PromptVariant, ...] = ()
        self.metrics: tuple[MetricChoice, ...] = ()
        self.sweep_models: tuple[str, ...] = ()
        self._settings: ModelSettings = ModelSettings()
        self.sweep_results: tuple[SweepCell, ...] = ()

        # --- the round under test -----------------------------------------
        # The unit this workbench measures. Held here rather than in Streamlit
        # state so every rule about what invalidates a result is a plain test.
        self.framework_keys: tuple[str, ...] = (DEFAULT_FRAMEWORK,)
        self.round_model_id: str = ""
        self.system_prompt: str = ""
        self.user_prompt: str = ""
        self.tools: tuple[ToolSpec, ...] = ()
        self.output_structure: OutputStructure | None = None
        self.last_result: CallResult | None = None
        self.dropped_parameters: tuple[str, ...] = ()
        # What the last applied kit put on screen, so an edit to it is visible.
        self._kit_baseline: tuple[str, str, tuple[ToolSpec, ...], OutputStructure | None] | None = None

    @property
    def new_id(self) -> IdFactory:
        return self._new_id

    @property
    def clock(self) -> Callable[[], datetime]:
        return self._clock

    # --- results, and what invalidates them -------------------------------

    def _discard_results(self) -> None:
        """Drop any sweep results, because what produced them has moved."""
        self.sweep_results = ()

    @property
    def settings(self) -> ModelSettings:
        return self._settings

    @settings.setter
    def settings(self, settings: ModelSettings) -> None:
        """Adopt sampling settings, discarding results if they actually changed.

        The sidebar reassigns this on every rerun, so an assignment of the same
        values must not count as a change — otherwise results would vanish the
        instant they were drawn.
        """
        if settings == self._settings:
            return
        self._settings = settings
        self._discard_results()

    # --- the case ---------------------------------------------------------

    def describe(self, description: str) -> TaskType | None:
        """Record the description and propose a task type for it."""
        self.description = description
        return task_catalog.propose(description)

    def set_task_type(self, task: TaskType) -> None:
        """Adopt a task type, discarding anything shaped by a previous one.

        Variants written for one kind of job and metrics chosen for it do not
        transfer, and keeping them would let a classifier's prompt be judged by
        a summarizer's metrics without anyone noticing.
        """
        if self.task_type is not None and self.task_type.key == task.key:
            return
        self.task_type = task
        self.variants = ()
        self.sweep_models = ()
        self.metrics = deepeval_metrics.suggested_for(task.metric_keys)
        self.settings = task.suggested_settings
        self._discard_results()

    @property
    def brief(self) -> CaseBrief | None:
        """The case as one value, once there is enough of it to be one."""
        if self.task_type is None or not self.description.strip():
            return None
        return CaseBrief(
            description=self.description.strip(),
            task_type_key=self.task_type.key,
            created_at=self._clock(),
            cases=self.cases,
            output_format=self.output_format,
        )

    def set_cases(self, cases: Sequence[EvalCase]) -> None:
        self.cases = tuple(cases)
        self._discard_results()

    def add_case(
        self,
        *,
        input: str,
        expected_output: str | None = None,
        context: Sequence[str] = (),
        notes: str = "",
    ) -> EvalCase:
        """Write a case down rather than generating one.

        Generation needs a provider key; this does not. A workbench that cannot
        be used at all until someone has paid for a key is not a workbench, and
        the case you already know you care about is usually the one worth
        writing first.
        """
        case = EvalCase(
            id=self._new_id("case"),
            input=input,
            expected_output=(expected_output or "").strip() or None,
            context=tuple(c for c in context if c.strip()),
            notes=notes,
        )
        self.cases = self.cases + (case,)
        self._discard_results()
        return case

    def remove_case(self, case_id: str) -> None:
        self.cases = tuple(case for case in self.cases if case.id != case_id)
        self._discard_results()

    def replace_case(self, case_id: str, case: EvalCase) -> None:
        self.cases = tuple(case if c.id == case_id else c for c in self.cases)
        self._discard_results()

    # --- variants ---------------------------------------------------------

    def set_variants(
        self, variants: Sequence[PromptVariant], *, keep_hand_written: bool = False
    ) -> None:
        """Replace the generated variants.

        ``keep_hand_written`` preserves anything the user wrote themselves,
        because that prompt is usually the baseline the whole comparison exists
        to beat — losing it to a regeneration would defeat the point.
        """
        kept = (
            tuple(v for v in self.variants if v.approach_key == HAND_WRITTEN_KEY)
            if keep_hand_written
            else ()
        )
        self.variants = kept + tuple(variants)
        self._discard_results()

    def add_variant(self, system_prompt: str) -> PromptVariant:
        """Add a prompt you already have, as a variant to compare against.

        The obvious thing to want from a prompt workbench and the thing it could
        not previously do: bring your current prompt and find out whether any of
        the generated approaches actually beats it.
        """
        if not system_prompt.strip():
            raise ValueError("A variant cannot have an empty prompt")
        variant = PromptVariant(
            id=self._new_id("variant"),
            approach_key=HAND_WRITTEN_KEY,
            approach_label=HAND_WRITTEN_LABEL,
            task_type_key=self.task_type.key if self.task_type else "",
            system_prompt=system_prompt,
            revision=1,
            created_at=self._clock(),
        )
        self.variants = self.variants + (variant,)
        self._discard_results()
        return variant

    def variant(self, variant_id: str) -> PromptVariant:
        for variant in self.variants:
            if variant.id == variant_id:
                return variant
        raise KeyError(f"No variant {variant_id!r}")

    def edit_variant(self, variant_id: str, system_prompt: str) -> PromptVariant:
        edited = self.variant(variant_id).with_text(system_prompt, edited_at=self._clock())
        self.variants = tuple(edited if v.id == variant_id else v for v in self.variants)
        self._discard_results()
        return edited

    @property
    def has_edited_variants(self) -> bool:
        return any(variant.edited for variant in self.variants)

    # --- metrics ----------------------------------------------------------

    def metric(self, key: str) -> MetricChoice:
        for choice in self.metrics:
            if choice.key == key:
                return choice
        raise KeyError(f"No metric {key!r} selected")

    def update_metric(self, key: str, choice: MetricChoice) -> None:
        self.metrics = tuple(choice if c.key == key else c for c in self.metrics)
        self._discard_results()

    def add_metric(self, key: str) -> None:
        if any(c.key == key for c in self.metrics):
            return
        self.metrics = self.metrics + deepeval_metrics.suggested_for((key,))
        self._discard_results()

    def remove_metric(self, key: str) -> None:
        self.metrics = tuple(c for c in self.metrics if c.key != key)
        self._discard_results()

    @property
    def enabled_metrics(self) -> tuple[MetricChoice, ...]:
        return tuple(choice for choice in self.metrics if choice.enabled)

    @property
    def judged_metric_count(self) -> int:
        """How many enabled metrics will actually call a judge — the cost driver."""
        return sum(1 for choice in self.enabled_metrics if choice.spec.uses_judge)

    # --- the round under test ---------------------------------------------

    def set_framework_keys(self, keys: Sequence[str]) -> None:
        """Choose which frameworks the round runs through.

        A result names the framework that produced it, so changing the set
        changes what the results are about — and they go.
        """
        chosen = tuple(dict.fromkeys(keys))
        if chosen == self.framework_keys:
            return
        self.framework_keys = chosen
        self._discard_results()

    def set_round_model(self, model_id: str) -> None:
        """Select the model under test, dropping parameters it does not publish."""
        if model_id == self.round_model_id:
            return
        self.round_model_id = model_id
        kept, dropped = self.registry.strip_unsupported(model_id, self._settings)
        self._settings = kept
        self.dropped_parameters = dropped
        self._discard_results()

    def set_prompts(self, *, system_prompt: str, user_prompt: str) -> None:
        if (system_prompt, user_prompt) == (self.system_prompt, self.user_prompt):
            return
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self._discard_results()

    def set_tools(self, tools: Sequence[ToolSpec]) -> None:
        chosen = tuple(tools)
        if chosen == self.tools:
            return
        self.tools = chosen
        self._discard_results()

    def set_output_structure(self, structure: OutputStructure | None) -> None:
        if structure == self.output_structure:
            return
        self.output_structure = structure
        self._discard_results()

    @property
    def structure_is_enforced(self) -> bool:
        """Whether the selected model would hold itself to the shape."""
        return self.output_structure is not None and self.registry.can_enforce_structure(
            self.round_model_id
        )

    @property
    def structure_note(self) -> str:
        """The sentence saying which of the two is in force."""
        return shapes.enforcement_note(self.structure_is_enforced)

    @property
    def round_is_runnable(self) -> bool:
        return bool(
            self.system_prompt.strip() and self.user_prompt.strip() and self.round_model_id
        )

    def call_request(self) -> CallRequest:
        """The round as one value, with the enforcement flag honestly set."""
        if not self.round_is_runnable:
            raise ValueError(
                "A round needs a system prompt, a user prompt and a selected model."
            )
        return CallRequest(
            system_prompt=self.system_prompt,
            user_prompt=self.user_prompt,
            model_id=self.round_model_id,
            settings=self._settings,
            tools=self.tools,
            output_structure=self.output_structure,
            structure_is_enforced=self.structure_is_enforced,
        )

    # --- starting kits -----------------------------------------------------

    class EditsWouldBeLost(RuntimeError):
        """Applying a kit would discard work the user did themselves."""

    @property
    def kit_fields_edited(self) -> bool:
        """Whether anything a kit filled in has since been changed by hand.

        Compared against what the kit actually wrote rather than against a
        "dirty" flag, so re-typing the same text does not count as an edit and
        an undo genuinely undoes.
        """
        if self._kit_baseline is None:
            return False
        return self._kit_baseline != (
            self.system_prompt,
            self.user_prompt,
            self.tools,
            self.output_structure,
        )

    def apply_kit(self, task: TaskType, *, overwrite: bool = False) -> presets.StartingKit:
        """Fill the round in from ``task``'s starting kit.

        Every field it writes stays editable — the kit knows the shape of the
        job and nothing about the user's case. It writes the prompts, the tools,
        the answer shape, the metrics and the temperature, and it decides
        nothing about how the call is made: the framework, whether tools are
        sent and whether a shape is asked for stay the user's choices.

        Raises ``EditsWouldBeLost`` rather than overwriting hand-edited fields,
        because only the caller has a screen to ask on.
        """
        if self.kit_fields_edited and not overwrite:
            raise self.EditsWouldBeLost(
                "Applying this starting kit would overwrite prompts, tools or the "
                "answer shape you edited. Apply it with overwrite=True to replace them."
            )
        kit = presets.load(task.preset_key)
        self.system_prompt = kit.system_prompt
        self.user_prompt = kit.user_prompt
        self.tools = kit.tools
        self.output_structure = kit.output_structure
        if kit.metrics:
            self.metrics = deepeval_metrics.suggested_for(tuple(kit.metrics))
            self.metrics = tuple(
                replace(choice, threshold=kit.metrics[choice.key])
                if choice.key in kit.metrics
                else choice
                for choice in self.metrics
            )
        if kit.temperature is not None:
            self._settings = replace(self._settings, temperature=kit.temperature)
        self._kit_baseline = (
            self.system_prompt,
            self.user_prompt,
            self.tools,
            self.output_structure,
        )
        self._discard_results()
        return kit

    # --- where we are -----------------------------------------------------

    @property
    def stage(self) -> str:
        """The furthest step this session has reached."""
        if self.task_type is None or not self.description.strip():
            return "describe"
        if not self.cases:
            return "cases"
        if not self.variants:
            return "variants"
        if not self.enabled_metrics:
            return "metrics"
        return "sweep"

    def is_past(self, stage: str) -> bool:
        """Whether the session has got at least as far as ``stage``."""
        return STAGES.index(self.stage) >= STAGES.index(stage)
