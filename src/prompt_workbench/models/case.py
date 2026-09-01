"""The user's own case: what they need, and the examples to try it on.

A test case here is shaped by where it is going. deepeval scores an
``LLMTestCase`` with a fixed set of fields — input, actual output, expected
output, context, retrieval context, tools — and a metric reads some subset of
them. So a case carries those fields and nothing else, which means the workbench
can say "faithfulness needs retrieval context and your cases have none" before
anything is spent, rather than after.

The expected output is optional and usually absent. Most jobs have many
acceptable answers, and forcing a single reference turns a fair test into a
guess-my-wording test.
"""

from dataclasses import dataclass, field
from datetime import datetime


@dataclass(frozen=True)
class EvalCase:
    """One input to try the prompt on, and whatever is known about a good answer.

    Named ``EvalCase`` rather than ``TestCase`` for two reasons: pytest collects
    anything called ``Test*`` as a test class, and deepeval already owns
    ``LLMTestCase``, which this maps onto rather than replaces.
    """

    id: str
    input: str
    expected_output: str | None = None
    context: tuple[str, ...] = ()
    retrieval_context: tuple[str, ...] = ()
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.input.strip():
            raise ValueError("A test case needs an input")

    def supplied_fields(self) -> set[str]:
        """Which ``LLMTestCase`` fields this case actually populates."""
        fields = {"input", "actual_output"}
        if self.expected_output and self.expected_output.strip():
            fields.add("expected_output")
        if self.context:
            fields.add("context")
        if self.retrieval_context:
            fields.add("retrieval_context")
        return fields


@dataclass(frozen=True)
class CaseBrief:
    """What the user is trying to build a prompt for."""

    description: str
    task_type_key: str
    created_at: datetime
    cases: tuple[EvalCase, ...] = ()
    output_format: str = ""
    notes: str = ""

    def supplied_fields(self) -> set[str]:
        """Fields available across every case — the intersection, not the union.

        A metric that can score three cases out of eight is not a metric you can
        put a threshold on, so availability means all of them.
        """
        if not self.cases:
            return set()
        common = self.cases[0].supplied_fields()
        for case in self.cases[1:]:
            common &= case.supplied_fields()
        return common

    def case(self, case_id: str) -> EvalCase:
        for case in self.cases:
            if case.id == case_id:
                return case
        raise KeyError(f"No test case {case_id!r}")
