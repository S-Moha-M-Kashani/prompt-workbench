"""What every generation step shares: one error type and one way to read JSON.

The workbench asks a model for structured data in three places, and all three
have the same problem — a model asked for JSON sometimes writes a sentence
around it, and sometimes ignores the request entirely. The tolerance is
deliberately narrow: find an object inside prose, but never guess at a reply
that has none. A generation step that half-succeeds leaves the user with
artifacts they did not write and cannot trace.
"""

import json
import re
from typing import Any

# The outermost {...} in the reply. Greedy on purpose: a nested object must not
# end the match early.
_JSON_OBJECT = re.compile(r"\{.*\}", re.S)


class GenerationError(RuntimeError):
    """A generation step did not produce something the workbench can use."""


def parse_json_object(reply: str, *, what: str) -> dict[str, Any]:
    """The JSON object in ``reply``, or a ``GenerationError`` naming ``what`` failed."""
    match = _JSON_OBJECT.search(reply)
    if not match:
        raise GenerationError(
            f"The model's {what} could not be read: it returned no JSON object. "
            f"It said: {reply.strip()[:300]!r}"
        )
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError as error:
        raise GenerationError(
            f"The model's {what} could not be read as JSON: {error}"
        ) from error
    if not isinstance(parsed, dict):
        raise GenerationError(f"The model's {what} was not a JSON object")
    return parsed


def as_text_tuple(value: Any) -> tuple[str, ...]:
    """A JSON list coerced to a tuple of non-empty strings."""
    if not isinstance(value, list):
        return ()
    return tuple(str(item).strip() for item in value if str(item).strip())
