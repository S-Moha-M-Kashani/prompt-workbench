"""Everything one session has built, and how those pieces relate.

This is where the workbench's state lives — deliberately here and not in the UI
module, so that "what happens when the brief changes after candidates exist?" is
a question with a tested answer rather than one that depends on which Streamlit
widget ran first.

Two ideas shape it:

**Revisions, not overwrites.** Editing an artifact makes a new revision. The old
one is gone from the workspace, but every artifact generated *from* it still
records which revision it read, so ``stale_artifacts`` can point at the gap
instead of pretending it does not exist.

**Edits are protected.** Hand-written text is the most valuable thing in the
workspace and the easiest to lose to a stray button press, so regenerating over
an edited candidate requires saying so explicitly.
"""

from collections.abc import Callable, Sequence
from dataclasses import replace
from datetime import UTC, datetime

from prompt_workbench.models.brief import BriefSnapshot, PromptBrief
from prompt_workbench.models.candidates import CandidatePrompt
from prompt_workbench.models.evaluation import EvaluationRun
from prompt_workbench.models.execution import ExecutionRecord
from prompt_workbench.models.ground_truth import GroundTruthCase, GroundTruthDataset
from prompt_workbench.models.identifiers import IdFactory, random_id
from prompt_workbench.models.metrics import MetricDefinition, MetricKind
from prompt_workbench.services import metric_catalog


def _now() -> datetime:
    return datetime.now(UTC)


class Workspace:
    """One session's artifacts, and the rules for moving between them."""

    def __init__(
        self,
        *,
        new_id: IdFactory = random_id,
        clock: Callable[[], datetime] = _now,
    ) -> None:
        self._new_id = new_id
        self._clock = clock

        self.draft_brief: PromptBrief = PromptBrief()
        self.confirmed_brief: BriefSnapshot | None = None
        self.dataset: GroundTruthDataset | None = None
        self.candidates: tuple[CandidatePrompt, ...] = ()
        self.metrics: tuple[MetricDefinition, ...] = metric_catalog.built_in_metrics()
        self.executions: tuple[ExecutionRecord, ...] = ()
        self.evaluations: tuple[EvaluationRun, ...] = ()
        self.platform_instruction_revision: int = 1

    # The injected clock and id factory, exposed so a caller generating an
    # artifact outside the workspace stamps it from the same sources — a second
    # `uuid4` or `datetime.now` would make test snapshots non-deterministic for
    # no gain.
    @property
    def new_id(self) -> IdFactory:
        return self._new_id

    @property
    def clock(self) -> Callable[[], datetime]:
        return self._clock

    # --- brief ------------------------------------------------------------

    def update_draft(self, brief: PromptBrief) -> None:
        self.draft_brief = brief

    @property
    def brief_is_dirty(self) -> bool:
        """Whether the draft has moved on from the last confirmed snapshot."""
        return self.confirmed_brief is not None and self.draft_brief != self.confirmed_brief.brief

    @property
    def can_generate(self) -> bool:
        """Whether there is a confirmed brief to generate anything from."""
        return self.confirmed_brief is not None

    def confirm_brief(self) -> BriefSnapshot:
        """Freeze the draft as the next revision and return it."""
        if self.draft_brief.is_empty:
            raise ValueError(
                "An empty brief cannot be confirmed — describe at least one thing "
                "the prompt should do first."
            )
        previous = self.confirmed_brief
        snapshot = BriefSnapshot(
            id=previous.id if previous else self._new_id("brief"),
            revision=previous.revision + 1 if previous else 1,
            brief=self.draft_brief,
            created_at=self._clock(),
        )
        self.confirmed_brief = snapshot
        return snapshot

    # --- ground truth -----------------------------------------------------

    def set_dataset(self, dataset: GroundTruthDataset) -> None:
        self.dataset = dataset

    def _require_brief(self) -> BriefSnapshot:
        if self.confirmed_brief is None:
            raise ValueError("Confirm a brief before creating ground-truth cases.")
        return self.confirmed_brief

    def _mutate_cases(self, cases: tuple[GroundTruthCase, ...]) -> None:
        """Store ``cases`` as the next dataset revision, creating one if needed."""
        if self.dataset is None:
            self.dataset = GroundTruthDataset(
                id=self._new_id("dataset"),
                revision=1,
                source_brief=self._require_brief().ref,
                cases=cases,
                created_at=self._clock(),
            )
        else:
            self.dataset = self.dataset.with_cases(cases)

    def new_case_id(self) -> str:
        return self._new_id("case")

    def add_case(self, case: GroundTruthCase) -> None:
        existing = self.dataset.cases if self.dataset else ()
        self._mutate_cases(existing + (replace(case, id=self._new_id("case")),))

    def replace_case(self, case_id: str, case: GroundTruthCase) -> None:
        if self.dataset is None:
            raise KeyError(case_id)
        self._mutate_cases(
            tuple(
                replace(case, id=case_id) if existing.id == case_id else existing
                for existing in self.dataset.cases
            )
        )

    def duplicate_case(self, case_id: str) -> None:
        if self.dataset is None:
            raise KeyError(case_id)
        original = self.dataset.case(case_id)
        self._mutate_cases(self.dataset.cases + (replace(original, id=self._new_id("case")),))

    def remove_case(self, case_id: str) -> None:
        if self.dataset is None:
            raise KeyError(case_id)
        self._mutate_cases(tuple(c for c in self.dataset.cases if c.id != case_id))

    # --- candidates -------------------------------------------------------

    @property
    def has_edited_candidates(self) -> bool:
        return any(candidate.edited for candidate in self.candidates)

    def set_candidates(
        self, candidates: Sequence[CandidatePrompt], *, overwrite: bool = False
    ) -> None:
        """Replace the candidate set, refusing to discard hand edits by accident."""
        if self.has_edited_candidates and not overwrite:
            raise ValueError(
                "Some candidates have been edited by hand. Regenerating would "
                "discard those edits — confirm the overwrite to continue."
            )
        self.candidates = tuple(candidates)

    def candidate(self, candidate_id: str) -> CandidatePrompt:
        for candidate in self.candidates:
            if candidate.id == candidate_id:
                return candidate
        raise KeyError(f"No candidate {candidate_id!r}")

    def edit_candidate(self, candidate_id: str, system_prompt: str) -> CandidatePrompt:
        """Record a hand edit as a new revision of that one candidate."""
        edited = self.candidate(candidate_id).with_text(system_prompt, edited_at=self._clock())
        self.candidates = tuple(
            edited if c.id == candidate_id else c for c in self.candidates
        )
        return edited

    # --- metrics ----------------------------------------------------------

    def metric(self, metric_id: str) -> MetricDefinition:
        for metric in self.metrics:
            if metric.id == metric_id:
                return metric
        raise KeyError(f"No metric {metric_id!r}")

    def add_metric(self, *, name: str, rubric: str, weight: float = 1.0) -> MetricDefinition:
        metric = MetricDefinition(
            id=self._new_id("metric"),
            name=name,
            rubric=rubric,
            kind=MetricKind.RUBRIC,
            weight=weight,
        )
        self.metrics = self.metrics + (metric,)
        return metric

    def _replace_metric(self, metric: MetricDefinition) -> None:
        self.metrics = tuple(metric if m.id == metric.id else m for m in self.metrics)

    def update_metric(self, metric_id: str, *, name: str, rubric: str) -> MetricDefinition:
        updated = replace(self.metric(metric_id), name=name, rubric=rubric)
        self._replace_metric(updated)
        return updated

    def set_metric_enabled(self, metric_id: str, enabled: bool) -> None:
        self._replace_metric(self.metric(metric_id).with_enabled(enabled))

    def set_metric_weight(self, metric_id: str, weight: float) -> None:
        self._replace_metric(self.metric(metric_id).with_weight(weight))

    def remove_metric(self, metric_id: str) -> None:
        """Remove a custom metric. Built-ins are disabled instead of deleted, so
        the shipped catalog is always recoverable without restarting."""
        if self.metric(metric_id).builtin:
            raise ValueError(
                "A built-in metric cannot be removed. Disable it instead — that "
                "excludes it from grades and keeps it available later."
            )
        self.metrics = tuple(m for m in self.metrics if m.id != metric_id)

    # --- results ----------------------------------------------------------

    def record_execution(self, record: ExecutionRecord) -> None:
        self.executions = self.executions + (record,)

    def record_evaluation(self, run: EvaluationRun) -> None:
        self.evaluations = self.evaluations + (run,)

    @property
    def latest_evaluation(self) -> EvaluationRun | None:
        return self.evaluations[-1] if self.evaluations else None

    def executions_for(self, candidate_id: str) -> tuple[ExecutionRecord, ...]:
        return tuple(r for r in self.executions if r.candidate.id == candidate_id)

    # --- staleness --------------------------------------------------------

    def stale_artifacts(self) -> tuple[str, ...]:
        """Artifacts built from a snapshot that has since moved on."""
        if self.confirmed_brief is None:
            return ()
        current = self.confirmed_brief.ref
        stale: list[str] = []

        if self.dataset is not None and self.dataset.source_brief.is_stale_against(current):
            stale.append(
                f"The ground truth was generated from brief revision "
                f"{self.dataset.source_brief.revision}, and the brief is now at "
                f"revision {current.revision}."
            )
        outdated = [c for c in self.candidates if c.source_brief.is_stale_against(current)]
        if outdated:
            stale.append(
                f"{len(outdated)} candidate(s) were generated from an earlier brief "
                f"revision than the current one ({current.revision})."
            )
        return tuple(stale)
