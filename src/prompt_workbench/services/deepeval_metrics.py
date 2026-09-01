"""The metric layer: deepeval, behind an optional dependency.

Why deepeval rather than something written here. The point of measuring a prompt
in a workbench is to keep measuring it afterwards, in the project that ships. If
the workbench invents its own metrics, everything learned here has to be
translated on the way out, and a translated threshold is a new threshold. Using
the same library on both sides means the configuration arrived at here is the
configuration written into the other project's test suite — same classes, same
criteria, same numbers.

It is optional because it costs thirty-one dependencies, including telemetry.
The base install stays small; the metric work asks for the extra and says how to
get it. Telemetry is switched off before deepeval is imported, not after.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

INSTALL_HINT = "uv sync --extra deepeval"

# Set before the first import: deepeval reads this into its settings at import
# time, and a workbench has no business reporting the user's usage anywhere.
os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")


class DeepEvalMissing(RuntimeError):
    """deepeval is not installed, and the metric layer needs it."""


class MetricNotConfigured(RuntimeError):
    """A metric takes a required argument the user has not supplied yet."""


class JudgeRequired(RuntimeError):
    """A judged metric was built without a judge.

    Raised rather than allowed, because deepeval's own default is to reach for
    an ambient ``OPENAI_API_KEY``. This project passes every credential
    explicitly, and a metric quietly billing a key nobody chose is exactly the
    failure that rule exists to prevent.
    """


@dataclass(frozen=True)
class ConfigField:
    """Something a metric needs before it can be constructed at all.

    Four of the sixteen metrics take required constructor arguments — a schema,
    the prompt's instructions, a domain, a list of advice types. Modelling them
    means the interface can ask for them, instead of the metric failing to build
    at the moment the user presses evaluate.
    """

    name: str
    label: str
    kind: str  # "text" | "lines" | "json_example"
    help: str


def is_available() -> bool:
    """Whether the optional dependency is present."""
    try:
        import deepeval  # noqa: F401
    except ImportError:
        return False
    return True


def require() -> None:
    """Raise with the install command rather than an ImportError traceback."""
    if not is_available():
        raise DeepEvalMissing(
            "The metric layer needs deepeval, which is an optional extra so the "
            f"base install stays small. Install it with:  {INSTALL_HINT}"
        )


@dataclass(frozen=True)
class MetricSpec:
    """One metric the workbench knows how to build and explain.

    ``needs`` names the ``LLMTestCase`` fields the metric reads. It is what lets
    the workbench say "this metric cannot run on your cases, they have no
    retrieval context" before spending anything, rather than after.
    """

    key: str
    label: str
    deepeval_class: str
    purpose: str
    default_threshold: float
    needs: tuple[str, ...] = ("input", "actual_output")
    # Whether a model must be supplied. Distinct from how the score is reached:
    # two metrics compute their score in code and still ask a model to write the
    # reason, so they need a judge without being judged.
    uses_judge: bool = True
    deterministic_score: bool = False
    custom_criteria: bool = False
    # Every deepeval 4.x metric passes on `score >= threshold`, safety metrics
    # included — 4.2 flipped bias, misuse and hallucination to match the rest.
    # The field stays because the direction is a fact about the library rather
    # than a constant, and `test_directions_match_deepeval` fails the moment it
    # changes again. Reading one of these backwards inverts a verdict, and a
    # threshold of 0.2 meant to cap violations becomes a floor almost anything
    # clears.
    higher_is_better: bool = True
    note: str = ""
    config_fields: tuple[ConfigField, ...] = ()

    @property
    def is_deterministic(self) -> bool:
        """Whether the *score* is computed in code and so cannot drift."""
        return self.deterministic_score


METRIC_SPECS: dict[str, MetricSpec] = {
    "g_eval_criteria": MetricSpec(
        key="g_eval_criteria",
        label="Custom criteria (GEval)",
        deepeval_class="GEval",
        purpose=(
            "Scores against criteria you write yourself. The workhorse: anything "
            "specific to your case that no packaged metric names."
        ),
        default_threshold=0.8,
        custom_criteria=True,
        note=(
            "Prefer explicit evaluation_steps over a single criteria sentence — "
            "steps are markedly more stable between runs."
        ),
    ),
    "answer_relevancy": MetricSpec(
        key="answer_relevancy",
        label="Answer relevancy",
        deepeval_class="AnswerRelevancyMetric",
        purpose="Whether the answer actually addresses the question asked.",
        default_threshold=0.8,
    ),
    "faithfulness": MetricSpec(
        key="faithfulness",
        label="Faithfulness",
        deepeval_class="FaithfulnessMetric",
        purpose="Whether every claim is supported by the retrieved context.",
        default_threshold=0.9,
        needs=("input", "actual_output", "retrieval_context"),
    ),
    "contextual_relevancy": MetricSpec(
        key="contextual_relevancy",
        label="Contextual relevancy",
        deepeval_class="ContextualRelevancyMetric",
        purpose="Whether what retrieval returned was relevant — a retriever check, not a prompt one.",
        default_threshold=0.7,
        needs=("input", "actual_output", "retrieval_context"),
    ),
    "hallucination": MetricSpec(
        key="hallucination",
        label="Hallucination",
        deepeval_class="HallucinationMetric",
        purpose="Whether the output stays consistent with the supplied context.",
        default_threshold=0.9,
        needs=("input", "actual_output", "context"),
    ),
    "summarization": MetricSpec(
        key="summarization",
        label="Summarization",
        deepeval_class="SummarizationMetric",
        purpose="Whether the summary keeps what matters and adds nothing.",
        default_threshold=0.8,
    ),
    "json_correctness": MetricSpec(
        key="json_correctness",
        label="JSON correctness",
        deepeval_class="JsonCorrectnessMetric",
        purpose="Whether the output parses and matches the schema.",
        default_threshold=1.0,
        deterministic_score=True,
        note=(
            "The score is computed in code, so it cannot drift — but deepeval still "
            "asks a model to write the reason, so a judge is required."
        ),
        config_fields=(
            ConfigField(
                name="expected_schema",
                label="An example of the expected JSON output",
                kind="json_example",
                help=(
                    "Paste one example object. A schema is derived from its keys and "
                    "value types; nested objects are checked as objects, not deeply."
                ),
            ),
        ),
    ),
    "exact_match": MetricSpec(
        key="exact_match",
        label="Exact match",
        deepeval_class="ExactMatchMetric",
        purpose="Output equals the expected output exactly. The only metric that calls no model at all.",
        default_threshold=1.0,
        needs=("input", "actual_output", "expected_output"),
        uses_judge=False,
        deterministic_score=True,
    ),
    "prompt_alignment": MetricSpec(
        key="prompt_alignment",
        label="Prompt alignment",
        deepeval_class="PromptAlignmentMetric",
        purpose="Whether the output obeys the instructions the prompt actually gave.",
        default_threshold=0.9,
        config_fields=(
            ConfigField(
                name="prompt_instructions",
                label="The instructions to hold it to, one per line",
                kind="lines",
                help="Usually the imperative lines of the prompt under test.",
            ),
        ),
    ),
    "tool_correctness": MetricSpec(
        key="tool_correctness",
        label="Tool correctness",
        deepeval_class="ToolCorrectnessMetric",
        purpose="Whether the tools called match the ones expected.",
        default_threshold=1.0,
        needs=("input", "actual_output", "tools_called", "expected_tools"),
        deterministic_score=True,
        note="Scored in code; a judge is still required for the written reason.",
    ),
    "task_completion": MetricSpec(
        key="task_completion",
        label="Task completion",
        deepeval_class="TaskCompletionMetric",
        purpose="Whether the agent actually finished what it set out to do.",
        default_threshold=0.8,
    ),
    "misuse": MetricSpec(
        key="misuse",
        label="Misuse",
        deepeval_class="MisuseMetric",
        purpose="Whether the model was steered outside its intended domain.",
        default_threshold=0.8,
        config_fields=(
            ConfigField(
                name="domain",
                label="The domain this assistant is for",
                kind="text",
                help="Two or three words, e.g. \"relationship coaching\".",
            ),
        ),
    ),
    "non_advice": MetricSpec(
        key="non_advice",
        label="Non-advice",
        deepeval_class="NonAdviceMetric",
        purpose="Whether it gave professional advice it should have withheld.",
        default_threshold=0.8,
        config_fields=(
            ConfigField(
                name="advice_types",
                label="Kinds of advice it must not give, one per line",
                kind="lines",
                help="For example: medical, legal, financial.",
            ),
        ),
    ),
    "pii_leakage": MetricSpec(
        key="pii_leakage",
        label="PII leakage",
        deepeval_class="PIILeakageMetric",
        purpose="Whether personal data appeared in the output.",
        default_threshold=0.9,
    ),
    "role_violation": MetricSpec(
        key="role_violation",
        label="Role violation",
        deepeval_class="RoleViolationMetric",
        purpose="Whether it broke the persona or role the prompt fixed.",
        default_threshold=0.8,
        config_fields=(
            ConfigField(
                name="role",
                label="The role the prompt fixes",
                kind="text",
                help='For example: "customer service agent", "relationship coach".',
            ),
        ),
    ),
    "bias": MetricSpec(
        key="bias",
        label="Bias",
        deepeval_class="BiasMetric",
        purpose="Whether the output is free of biased or unfair framing.",
        default_threshold=0.8,
    ),
}


@dataclass(frozen=True)
class MetricChoice:
    """One metric as the user has configured it for their case."""

    spec: MetricSpec
    threshold: float
    criteria: str = ""
    evaluation_steps: tuple[str, ...] = field(default_factory=tuple)
    enabled: bool = True
    config: dict[str, str] = field(default_factory=dict)

    @property
    def key(self) -> str:
        return self.spec.key

    def missing_config(self) -> tuple[str, ...]:
        """Required arguments this metric still lacks."""
        return tuple(
            field_spec.name
            for field_spec in self.spec.config_fields
            if not str(self.config.get(field_spec.name, "")).strip()
        )

    @property
    def is_ready(self) -> bool:
        """Whether this metric could be built right now."""
        return not self.missing_config()

    def missing_fields(self, available: set[str]) -> tuple[str, ...]:
        """Fields this metric needs that the user's cases do not supply."""
        return tuple(name for name in self.spec.needs if name not in available)

    def as_code(self, *, judge_expression: str = "judge") -> str:
        """The deepeval call that builds this metric.

        Shown so it can be copied into the other project's test suite — which is
        the whole reason for using deepeval rather than something local. It is
        text on screen, not a file written to disk.
        """
        parts: list[str] = []
        if self.spec.custom_criteria:
            parts.append('name="Custom criteria"')
            if self.evaluation_steps:
                steps = ", ".join(f'"{step}"' for step in self.evaluation_steps)
                parts.append(f"evaluation_steps=[{steps}]")
            elif self.criteria:
                parts.append(f'criteria="{self.criteria}"')
            parts.append(
                "evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT]"
            )
        for field_spec in self.spec.config_fields:
            raw = str(self.config.get(field_spec.name, "")).strip()
            if not raw:
                continue
            if field_spec.kind == "lines":
                values = ", ".join(f'"{line.strip()}"' for line in raw.splitlines() if line.strip())
                parts.append(f"{field_spec.name}=[{values}]")
            elif field_spec.kind == "json_example":
                parts.append(f"{field_spec.name}=ExpectedSchema")
            else:
                parts.append(f'{field_spec.name}="{raw}"')
        parts.append(f"threshold={self.threshold}")
        if self.spec.uses_judge:
            parts.append(f"model={judge_expression}")
        return f"{self.spec.deepeval_class}({', '.join(parts)})"


def suggested_for(metric_keys: tuple[str, ...]) -> tuple[MetricChoice, ...]:
    """The metrics a task type proposes, at their starting thresholds."""
    return tuple(
        MetricChoice(spec=METRIC_SPECS[key], threshold=METRIC_SPECS[key].default_threshold)
        for key in metric_keys
        if key in METRIC_SPECS
    )


def get(key: str) -> MetricSpec:
    return METRIC_SPECS[key]


def schema_from_example(example: str) -> Any:
    """A pydantic model derived from one example of the expected output.

    ``JsonCorrectnessMetric`` wants a model class, not a schema document, and
    asking a user to write pydantic in a text box would be a strange thing for a
    prompt workbench to do. So it takes an example of the output and infers the
    field names and types from it.

    Deliberately shallow: nested objects are checked as objects rather than
    recursed into, which is enough to catch the failure this metric exists for —
    an answer that is not the right shape at all.
    """
    require()
    import json as _json

    from pydantic import create_model

    try:
        parsed = _json.loads(example)
    except _json.JSONDecodeError as error:
        raise ValueError(f"That is not valid JSON: {error}") from error
    if not isinstance(parsed, dict):
        raise ValueError("Paste a JSON object — a single example of the expected output.")

    types: dict[str, Any] = {}
    for key, value in parsed.items():
        if isinstance(value, bool):
            annotation: Any = bool
        elif isinstance(value, int):
            annotation = int
        elif isinstance(value, float):
            annotation = float
        elif isinstance(value, list):
            annotation = list
        elif isinstance(value, dict):
            annotation = dict
        else:
            annotation = str
        types[key] = (annotation, ...)
    return create_model("ExpectedSchema", **types)


def build(choice: MetricChoice, *, judge: Any = None) -> Any:
    """Instantiate the real deepeval metric. Requires the optional dependency."""
    require()
    missing = choice.missing_config()
    if missing:
        raise MetricNotConfigured(
            f"{choice.spec.label} needs " + ", ".join(missing) + " before it can run."
        )
    import deepeval.metrics as metrics

    metric_class = getattr(metrics, choice.spec.deepeval_class, None)
    if metric_class is None:
        raise DeepEvalMissing(
            f"This deepeval version has no {choice.spec.deepeval_class}. "
            "Update the extra, or pick a different metric."
        )

    if choice.spec.uses_judge:
        if judge is None:
            raise JudgeRequired(
                f"{choice.spec.label} is judged by a model, so it needs one passed "
                "explicitly. Without it deepeval would fall back to an ambient "
                "OPENAI_API_KEY, which this workbench never does."
            )
        kwargs_model = judge
    else:
        kwargs_model = None

    kwargs: dict[str, Any] = {"threshold": choice.threshold}
    if kwargs_model is not None:
        kwargs["model"] = kwargs_model
    for field_spec in choice.spec.config_fields:
        raw = str(choice.config.get(field_spec.name, "")).strip()
        if field_spec.kind == "lines":
            kwargs[field_spec.name] = [
                line.strip() for line in raw.splitlines() if line.strip()
            ]
        elif field_spec.kind == "json_example":
            kwargs[field_spec.name] = schema_from_example(raw)
        else:
            kwargs[field_spec.name] = raw
    if choice.spec.custom_criteria:
        # `SingleTurnParams` in deepeval 4.2+, `LLMTestCaseParams` before it.
        # Resolved by name so both spellings work without a redefined import.
        import deepeval.test_case as test_case

        params = getattr(test_case, "SingleTurnParams", None) or test_case.LLMTestCaseParams

        kwargs["name"] = "Custom criteria"
        kwargs["evaluation_params"] = [params.INPUT, params.ACTUAL_OUTPUT]
        if choice.evaluation_steps:
            kwargs["evaluation_steps"] = list(choice.evaluation_steps)
        elif choice.criteria:
            kwargs["criteria"] = choice.criteria
    return metric_class(**kwargs)
