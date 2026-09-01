"""One recorded model response and everything needed to attribute it.

An evaluation is only meaningful if every score can be traced to the exact
candidate revision, model, settings, brief, and test case that produced the
response. A response without those is a nice-looking paragraph with no
provenance, so ``is_attributable`` gates it out of evaluation rather than
letting it quietly average in.
"""

from dataclasses import dataclass
from datetime import datetime

from prompt_workbench.models.model_settings import ModelSettings
from prompt_workbench.models.provenance import SourceRef


@dataclass(frozen=True)
class ExecutionRecord:
    """A response, plus the full configuration that generated it."""

    id: str
    thread_id: str
    candidate: SourceRef
    source_brief: SourceRef
    case_id: str | None
    model_id: str
    settings: ModelSettings
    user_message: str
    response: str
    created_at: datetime
    total_tokens: int = 0

    def missing_evidence(self) -> tuple[str, ...]:
        """Why this response cannot be evaluated, in words a user can act on."""
        reasons: list[str] = []
        if not self.case_id:
            reasons.append(
                "no ground-truth test case — send a dataset case rather than "
                "free text if you want this response scored"
            )
        if not self.response.strip():
            reasons.append("the model returned an empty response")
        return tuple(reasons)

    @property
    def is_attributable(self) -> bool:
        """Whether this response carries enough evidence to be scored."""
        return not self.missing_evidence()
