"""The metrics the workbench ships with.

Four, chosen so that each one answers a question the others cannot:

- **Criteria coverage** — did the reply do what this case requires?
- **Forbidden behaviour** — did it avoid what this case rules out? The mirror of
  the first, and a separate number because a reply can nail every requirement
  and still do the one thing it must not.
- **Format compliance** — is it the right shape? Deterministic, because a
  judge asked "is this valid JSON" is a slow, expensive, occasionally wrong
  ``json.loads``.
- **Reference similarity** — where a case has an ideal answer, how close is
  this? Skipped, visibly, where it does not.

Weights start equal. They are the user's to change, and the Evaluate area shows
the arithmetic, so a default here is a starting point rather than an opinion
about what matters.
"""

from prompt_workbench.models.metrics import MetricDefinition, MetricKind

BUILT_IN_METRICS: tuple[MetricDefinition, ...] = (
    MetricDefinition(
        id="criteria_coverage",
        name="Criteria coverage",
        kind=MetricKind.RUBRIC,
        rubric=(
            "Score how completely the response satisfies the required criteria for "
            "this test case. Consider each criterion separately and judge only "
            "whether the response actually meets it, not whether the response is "
            "good in general. Score 1.0 when every criterion is fully met, 0.0 when "
            "none are, and the proportion met in between."
        ),
        weight=1.0,
        builtin=True,
    ),
    MetricDefinition(
        id="forbidden_behaviour",
        name="Forbidden behaviour",
        kind=MetricKind.RUBRIC,
        rubric=(
            "Score whether the response avoided every forbidden behaviour listed for "
            "this test case. Score 1.0 when none of them occur. Reduce the score in "
            "proportion to how many occur, and score 0.0 when the response commits "
            "all of them. Judge only the listed behaviours; do not invent new rules."
        ),
        weight=1.0,
        builtin=True,
    ),
    MetricDefinition(
        id="format_compliance",
        name="Format compliance",
        kind=MetricKind.DETERMINISTIC,
        check="format_compliance",
        rubric=(
            "Checked in code, not by a judge: the response is tested against the "
            "output format stated in the brief — whether it parses as JSON, and "
            "whether it respects any stated word or character limit. Rules that "
            "cannot be checked mechanically are reported as not applicable rather "
            "than passed."
        ),
        weight=1.0,
        builtin=True,
    ),
    MetricDefinition(
        id="reference_similarity",
        name="Reference similarity",
        kind=MetricKind.RUBRIC,
        requires_reference=True,
        rubric=(
            "Score how closely the response matches the meaning of the reference "
            "answer for this test case. Judge agreement in substance, not wording: "
            "a differently phrased response that conveys the same content scores "
            "high. Score 1.0 for full agreement and 0.0 for a response that "
            "contradicts or omits the reference's substance."
        ),
        weight=1.0,
        builtin=True,
    ),
)

_BY_ID: dict[str, MetricDefinition] = {metric.id: metric for metric in BUILT_IN_METRICS}


def built_in_metrics() -> tuple[MetricDefinition, ...]:
    """Every shipped metric, in display order."""
    return BUILT_IN_METRICS


def get(metric_id: str) -> MetricDefinition:
    """One built-in metric by id; raises ``KeyError`` for an unknown one."""
    return _BY_ID[metric_id]
