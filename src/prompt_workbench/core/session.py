"""What one browser session is holding: a use case, a prompt, and what it did.

Small on purpose. The previous design had a workspace of interdependent
artifacts, each with revisions and staleness rules, because artifacts were
generated from one another. A session here holds one selected use case, the
prompt being worked on, the runs it produced, and the evaluations of those runs.

The one rule worth stating: selecting a use case resets the prompt and clears
the runs. Carrying a run from one situation into another would attach a response
to criteria it was never measured against.
"""

from collections.abc import Callable
from datetime import UTC, datetime

from prompt_workbench.models.evaluation import EvaluationRun
from prompt_workbench.models.identifiers import IdFactory, random_id
from prompt_workbench.models.metrics import MetricDefinition
from prompt_workbench.models.runs import PromptRun
from prompt_workbench.models.use_case import UseCase
from prompt_workbench.services import metric_catalog

ENGINEER_MODE = "engineer"
USER_MODE = "user"
MODES: tuple[str, ...] = (ENGINEER_MODE, USER_MODE)

MODE_LABELS = {
    ENGINEER_MODE: "Prompt engineer",
    USER_MODE: "End user (one-shot)",
}


def _now() -> datetime:
    return datetime.now(UTC)


class Session:
    """One person's work on one use case at a time."""

    def __init__(
        self,
        *,
        new_id: IdFactory = random_id,
        clock: Callable[[], datetime] = _now,
    ) -> None:
        self._new_id = new_id
        self._clock = clock

        self.use_case: UseCase | None = None
        self.prompt_under_test: str = ""
        self.prompt_revision: int = 0
        self.mode: str = ENGINEER_MODE
        self.runs: tuple[PromptRun, ...] = ()
        self.evaluations: tuple[EvaluationRun, ...] = ()
        self.metrics: tuple[MetricDefinition, ...] = metric_catalog.built_in_metrics()

    @property
    def new_id(self) -> IdFactory:
        return self._new_id

    @property
    def clock(self) -> Callable[[], datetime]:
        return self._clock

    # --- the selected use case -------------------------------------------

    def select(self, use_case: UseCase) -> None:
        """Load a use case, resetting everything that belonged to the last one."""
        self.use_case = use_case
        self.prompt_under_test = use_case.system_prompt
        self.prompt_revision = 1
        self.runs = ()
        self.evaluations = ()

    @property
    def is_ready(self) -> bool:
        return self.use_case is not None

    # --- the prompt under test -------------------------------------------

    def update_prompt(self, text: str) -> bool:
        """Record an edit as a new revision. Returns whether anything changed."""
        if not text.strip():
            raise ValueError("The prompt under test cannot be empty")
        if text == self.prompt_under_test:
            return False
        self.prompt_under_test = text
        self.prompt_revision += 1
        return True

    def reset_prompt(self) -> None:
        """Go back to the use case's original prompt, as a further revision."""
        if self.use_case is None:
            raise ValueError("No use case is selected")
        self.update_prompt(self.use_case.system_prompt)

    @property
    def prompt_is_modified(self) -> bool:
        return self.use_case is not None and self.prompt_under_test != self.use_case.system_prompt

    # --- results ----------------------------------------------------------

    def record_run(self, record: PromptRun) -> None:
        self.runs = self.runs + (record,)

    def record_evaluation(self, evaluation: EvaluationRun) -> None:
        self.evaluations = self.evaluations + (evaluation,)

    @property
    def latest_run(self) -> PromptRun | None:
        return self.runs[-1] if self.runs else None

    @property
    def latest_evaluation(self) -> EvaluationRun | None:
        return self.evaluations[-1] if self.evaluations else None

    def scoreable_runs(self) -> tuple[PromptRun, ...]:
        return tuple(record for record in self.runs if record.is_scoreable)
