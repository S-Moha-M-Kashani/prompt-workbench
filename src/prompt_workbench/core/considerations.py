"""What the numbers on screen do and do not support, in one place.

`docs/LIMITATIONS.md` exists and nobody reading a score opens it. So these
statements are rendered on the page that shows the numbers, and the document
quotes the same strings — a test asserts every one of them appears there, so
the two cannot drift.

Each statement names what it qualifies (``shown_beside``), so the page can put
the right sentence next to the right number instead of one paragraph of caveats
nobody reads.

Removing or weakening any of these while the corresponding number is still on
screen is the one edit this module exists to make difficult.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Consideration:
    """One honest limit, and where it belongs."""

    key: str
    title: str
    statement: str
    #: Names of the things on screen this qualifies, e.g. "latency", "score".
    shown_beside: tuple[str, ...]


CONSIDERATIONS: tuple[Consideration, ...] = (
    Consideration(
        key="single_run_latency",
        title="A single run's latency is an observation, not a benchmark",
        statement=(
            "A latency here is one run: it carries network variance, provider "
            "queueing and this machine's own overhead, and none of that is the "
            "framework's doing. Comparing two frameworks on one run each tells "
            "you almost nothing. Repeat runs and their spread are what turn this "
            "into a claim."
        ),
        shown_beside=("latency",),
    ),
    Consideration(
        key="judged_grades",
        title="A judged grade is one model's opinion",
        statement=(
            "Most metrics here are a model reading a rubric, and they move "
            "between runs and prefer longer, hedged answers. Exact match, JSON "
            "correctness and tool correctness decide in code; everything else "
            "is judgement. Compare variants against each other, and do not read "
            "a score as a percentage correct."
        ),
        shown_beside=("score", "grade"),
    ),
    Consideration(
        key="mocked_tools",
        title="Mocked tools measure the asking, not the handling",
        statement=(
            "Every tool here is mocked: it announces itself, returns that text, "
            "and records the call. So a tool trace measures whether the model "
            "asked for the right tool with the right arguments. It measures "
            "nothing about whether real tool output is handled, which is a "
            "different question needing a different bench."
        ),
        shown_beside=("tools", "trace"),
    ),
    Consideration(
        key="one_round",
        title="One round is not a multi-round agent",
        statement=(
            "The unit measured here is a single round: prompts in, one answer "
            "out, with whatever tool round-trips that answer needed. An agent's "
            "real behaviour lives in the state it carries between rounds, and "
            "this bench deliberately keeps none. Tune a workflow by bringing "
            "each of its rounds here separately."
        ),
        shown_beside=("latency", "tools", "trace", "score"),
    ),
    Consideration(
        key="unknown_cost",
        title="An unknown cost is shown as unknown, never as zero",
        statement=(
            "Where the provider published no price, or the framework reported "
            "no token counts, the cost is reported as unknown. It is never "
            "shown as $0.00, because a free call and a call nobody measured "
            "must not read the same."
        ),
        shown_beside=("cost",),
    ),
)


def all_considerations() -> tuple[Consideration, ...]:
    return CONSIDERATIONS


def beside(what: str) -> tuple[Consideration, ...]:
    """The statements that belong next to ``what`` on screen."""
    return tuple(item for item in CONSIDERATIONS if what in item.shown_beside)


def get(key: str) -> Consideration:
    for item in CONSIDERATIONS:
        if item.key == key:
            return item
    raise KeyError(f"No consideration {key!r}")
